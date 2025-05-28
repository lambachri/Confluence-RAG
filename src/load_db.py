import sys
import logging
import shutil
import os
import requests
from typing import List, Dict, Any, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

# Fix the logging format string - replace %s with %(levelname)s
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('../')
from src.config import (
    CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
    CONFLUENCE_USERNAME, CONFLUENCE_API_KEY, PERSIST_DIRECTORY
)

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
# Import Chroma from langchain_chroma instead of langchain_community
from langchain_chroma import Chroma

# Liste des IDs de pages spécifiques à charger par défaut
DEFAULT_PAGE_IDS = [
    "3376545896", "3710844929", "3985899883", "3729621164", "3712188435", 
            "3896606774", "4552491244", "4675077035", "3716349965", "3758030922", 
            "3758555439", "2384003194", "2640052240", "2722825033", "2639036983", 
            "2384691207", "2756640773","4444487681","4766369934","4055826672"
]

class CustomConfluenceLoader(ConfluenceLoader):
    """
    Extension de ConfluenceLoader pour charger récursivement des pages en-dessous d'un parent.
    Optionnel: vous pouvez utiliser la logique suivante si vous préférez
    charger via cette classe plutôt que via DataLoader.
    """
    
    def load_integration_section(self, integration_parent_id="2952528207", include_attachments=True) -> List[Document]:
        """
        Exemple d'utilisation de la logique `_get_all_integration_pages` 
        pour récupérer toutes les sous-pages récursivement, puis charger leur contenu.
        """
        integration_pages = self._get_all_integration_pages(integration_parent_id)
        
        documents = []
        for page_data in integration_pages:
            page_id = page_data.get("id")
            try:
                page_documents = self.load_document(page_id, include_attachments)
                documents.extend(page_documents)
            except Exception as e:
                logging.error(f"Error loading page {page_id}: {e}")
        
        return documents
    
    def _get_all_integration_pages(self, integration_parent_id: str) -> List[Dict[str, Any]]:
        """
        Récupère toutes les pages sous l'ID parent (integration_parent_id) de manière récursive.
        Renvoie la liste des objets JSON bruts.
        """
        all_pages = []
        pages_to_process = [integration_parent_id]
        processed_page_ids = set()
        
        username = self._username
        api_key = self._api_key
        base_url = self.base_url
        
        while pages_to_process:
            current_page_id = pages_to_process.pop(0)
            if current_page_id in processed_page_ids:
                continue
            processed_page_ids.add(current_page_id)
            
            endpoint = f"{base_url}/rest/api/content/{current_page_id}?expand=body.storage,version"
            response = requests.get(endpoint, auth=(username, api_key))
            
            if response.status_code == 200:
                page_data = response.json()
                all_pages.append(page_data)
                
                # Récupérer les enfants
                children_endpoint = f"{base_url}/rest/api/content/{current_page_id}/child/page"
                children_response = requests.get(children_endpoint, auth=(username, api_key))
                
                if children_response.status_code == 200:
                    children_data = children_response.json()
                    child_ids = [child["id"] for child in children_data.get("results", [])]
                    pages_to_process.extend(child_ids)
        
        return all_pages


class DataLoader():
    """Create, load, save the DB using the confluence Loader"""
    def __init__(
        self,
        confluence_url=CONFLUENCE_SPACE_NAME,
        username=CONFLUENCE_USERNAME,
        api_key=CONFLUENCE_API_KEY,
        space_key=CONFLUENCE_SPACE_KEY,
        persist_directory=PERSIST_DIRECTORY,
        integration_id="2952528207",  # ID of the Integration parent page
        page_ids=None,  # Nouveau paramètre pour les IDs de pages spécifiques
        use_default_pages=True  # Utiliser les pages par défaut si page_ids est None
    ):
        self.confluence_url = confluence_url
        self.username = username
        self.api_key = api_key
        self.space_key = space_key
        self.persist_directory = persist_directory
        self.integration_id = integration_id
        
        # Priorité: page_ids fournis > DEFAULT_PAGE_IDS si use_default_pages=True > liste vide
        if page_ids:
            # Valider et nettoyer les IDs pour éviter les erreurs
            self.page_ids = self._sanitize_page_ids(page_ids)
            logger.info(f"Utilisation des {len(self.page_ids)} pages IDs fournis")
        elif use_default_pages:
            self.page_ids = self._sanitize_page_ids(DEFAULT_PAGE_IDS)
            logger.info(f"Utilisation des {len(self.page_ids)} pages par défaut")
        else:
            self.page_ids = []
            logger.warning("Aucun ID de page spécifié et use_default_pages=False. Aucune page ne sera chargée.")

        # Un petit cache local (optionnel) pour éviter de retélécharger 
        # des pages inchangées (en se basant sur leur "version.number").
        # Peut être stocké sur disque, ou laissé en mémoire comme ici.
        self.page_cache_file = "confluence_page_cache.json"
        self._page_cache = self._load_page_cache()
        
        # Log important des IDs qui seront utilisés
        logger.info(f"IDs des pages qui seront chargées: {', '.join(self.page_ids)}")

    def _sanitize_page_ids(self, ids):
        """Nettoie et valide les IDs de page pour éviter les erreurs"""
        cleaned_ids = []
        for pid in ids:
            # Nettoyer l'ID pour assurer qu'il est valide (supprime espaces et caractères non numériques)
            pid = str(pid).strip()
            if not pid.isdigit():
                # Extraire uniquement les chiffres
                cleaned = ''.join(c for c in pid if c.isdigit())
                if cleaned:
                    cleaned_ids.append(cleaned)
                    if cleaned != pid:
                        logger.warning(f"ID de page nettoyé: '{pid}' -> '{cleaned}'")
                else:
                    logger.warning(f"ID de page ignoré car non numérique: '{pid}'")
            else:
                cleaned_ids.append(pid)
        
        # Retirer les doublons si nécessaire
        unique_ids = list(dict.fromkeys(cleaned_ids))
        if len(unique_ids) != len(cleaned_ids):
            logger.warning(f"Suppression de {len(cleaned_ids) - len(unique_ids)} IDs en double")
        
        return unique_ids

    def _load_page_cache(self) -> Dict[str, Any]:
        """Charge un cache JSON depuis un fichier local (optionnel)."""
        if os.path.exists(self.page_cache_file):
            try:
                with open(self.page_cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_page_cache(self):
        """Sauvegarde le cache en local (optionnel)."""
        try:
            with open(self.page_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._page_cache, f, indent=2)
        except Exception as e:
            logging.warning(f"Impossible d'écrire le cache: {e}")

    def load_from_confluence_loader(self):
        """
        Charge le contenu Confluence pour toutes les pages sous 'integration_id' 
        (y compris la page parente si nécessaire).
        """
        try:
            loader = ConfluenceLoader(
                url=self.confluence_url,
                username=self.username,
                api_key=self.api_key,
                space_key=self.space_key
            )
            auth = (self.username, self.api_key)
            base_url = self.confluence_url

            # 1) Récupérer récursivement tous les IDs
            logging.info(f"Recherche récursive des pages à partir de l'ID parent: {self.integration_id}")
            all_page_ids = self._get_all_page_ids_recursively(base_url, auth, self.integration_id)
            logging.info(f"Total des pages trouvées sous {self.integration_id}: {len(all_page_ids)}")

            # 2) Charger la page d'intégration (parent) si besoin
            integration_page = []
            try:
                logging.info(f"Chargement de la page parent (ID: {self.integration_id})...")
                page_data = self._fetch_page_info(self.integration_id)
                parent_title = page_data.get("title", f"Page ID: {self.integration_id}")
                logging.info(f"Page parent: {parent_title}")
                
                integration_page = loader.load(page_ids=[self.integration_id])
                logging.info(f"Page parent '{parent_title}' chargée: {len(integration_page)} documents")
            except Exception as e:
                logging.warning(f"Impossible de charger la page d'intégration: {e}")

            # 3) Charger les sous-pages en mode concurrent
            #    => Vous pouvez ajuster max_workers si vous voulez plus/moins de parallélisation
            logging.info(f"Début du chargement des {len(all_page_ids)} sous-pages...")
            all_docs = integration_page[:]
            all_docs += self._load_pages_concurrently(loader, all_page_ids, concurrency=5)

            logging.info(f"Chargement terminé. Total des documents: {len(all_docs)}")
            
            # Récapitulatif des pages chargées
            titles = [doc.metadata.get('title', 'Sans titre') for doc in all_docs]
            unique_titles = set(titles)
            logging.info(f"Pages uniques chargées: {len(unique_titles)}")
            
            # Afficher tous les titres uniques
            unique_titles_list = sorted(list(unique_titles))
            logging.info("=== LISTE COMPLÈTE DES PAGES CHARGÉES ===")
            for i, title in enumerate(unique_titles_list):
                logging.info(f"Page {i+1}/{len(unique_titles_list)}: {title}")

            # Sauvegarder le cache de pages pour usage futur
            self._save_page_cache()

            return all_docs

        except Exception as e:
            logging.error(f"Erreur lors du chargement: {e}")
            return self._load_all_and_filter()

    def _get_all_page_ids_recursively(self, base_url, auth, parent_id) -> List[str]:
        """
        Récupère tous les IDs de pages (et sous-pages) pour le parent_id donné, de manière récursive.
        """
        all_ids = []
        queue = [parent_id]
        visited = set()
        page_titles = {}  # Pour stocker les titres des pages
        
        logging.info("Début de la récupération récursive des IDs de pages...")
        logging.info("=== PAGES TROUVÉES (HIÉRARCHIE) ===")

        while queue:
            pid = queue.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            all_ids.append(pid)
            
            # Récupérer le titre de la page pour le logging
            try:
                page_endpoint = f"{base_url}/rest/api/content/{pid}?expand=version"
                page_resp = requests.get(page_endpoint, auth=auth)
                if page_resp.status_code == 200:
                    page_data = page_resp.json()
                    page_title = page_data.get("title", f"Page ID: {pid}")
                    page_titles[pid] = page_title
                    logging.info(f"Page trouvée: {page_title} (ID: {pid})")
            except Exception as e:
                logging.warning(f"Impossible de récupérer le titre pour la page {pid}: {e}")
                page_titles[pid] = f"Page ID: {pid}"

            # Récupérer les enfants
            children_endpoint = f"{base_url}/rest/api/content/{pid}/child/page"
            try:
                resp = requests.get(children_endpoint, auth=auth)
                if resp.status_code == 200:
                    data = resp.json()
                    child_pages = data.get("results", [])
                    
                    # Logging des enfants trouvés
                    if child_pages:
                        child_titles = [f"{child.get('title', '?')} (ID: {child['id']})" for child in child_pages]
                        logging.info(f"Sous-pages de '{page_titles[pid]}': {len(child_pages)}")
                        for i, child_title in enumerate(child_titles):
                            logging.info(f"  ├─ {i+1}/{len(child_titles)}: {child_title}")
                    
                    for child in child_pages:
                        queue.append(child["id"])
                        page_titles[child["id"]] = child.get("title", f"Page ID: {child['id']}")
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des enfants pour {page_titles[pid]}: {e}")

        logging.info(f"Récupération récursive terminée. {len(all_ids)} pages trouvées.")
        return all_ids

    def _load_pages_concurrently(self, loader: ConfluenceLoader, page_ids: List[str], concurrency=5) -> List[Document]:
        """
        Charge en parallèle les pages Confluence en utilisant ThreadPoolExecutor.
        Tire parti d'un petit cache si la version n'a pas changé.
        """
        documents = []
        # Découpage en batch, vous pouvez ajuster selon vos besoins.
        # (Cela peut limiter la taille de la file pour ne pas surcharger le serveur.)
        batch_size = 20
        
        # Précharger les titres des pages pour le logging
        page_titles = {}
        try:
            logging.info(f"Récupération des titres pour {len(page_ids)} pages...")
            for pid in page_ids:
                try:
                    page_data = self._fetch_page_info(pid)
                    page_titles[pid] = page_data.get("title", f"Page ID: {pid}")
                except Exception as e:
                    logging.warning(f"Impossible de récupérer le titre pour page ID {pid}: {e}")
                    page_titles[pid] = f"Page ID: {pid}"
            
            logging.info(f"Titres récupérés pour {len(page_titles)} pages. Début du chargement...")
            # Afficher la liste complète des pages qui vont être chargées
            logging.info("=== PAGES À CHARGER (LISTE COMPLÈTE) ===")
            for i, (pid, title) in enumerate(page_titles.items()):
                logging.info(f"Page {i+1}/{len(page_titles)}: {title} (ID: {pid})")
        except Exception as e:
            logging.warning(f"Erreur lors de la récupération des titres: {e}")

        def load_one_page(pid: str) -> List[Document]:
            page_title = page_titles.get(pid, f"Page ID: {pid}")
            logging.info(f"CHARGEMENT: {page_title} (ID: {pid})")
            
            # On peut checker la cache
            # => On va utiliser la version stockée si on la trouve, pour éviter de recharger.
            #    Ou alors on recharge si la page n'est pas en cache ou si la version a changé.
            if pid in self._page_cache:
                cache_info = self._page_cache[pid]
                # L'API de ConfluenceLoader.load ne donne pas toujours la version directement,
                # donc on peut recharger pour être sûr, OU on interroge l'API Confluence pour la version.
                # Simplifions: on interroge la version. Si inchangée, on réutilise le cache (doc).
                # => A vous de décider si vous stockez l'intégralité du texte, ou juste un résumé.
                try:
                    page_data = self._fetch_page_info(pid)
                    server_version = page_data["version"]["number"]
                    if server_version == cache_info["version"]:
                        # Page inchangée => on peut recréer le Document à partir du cache 
                        # (il faudrait stocker 'page_content' et 'metadata' dans le cache).
                        # Ici c'est un exemple minimaliste ; on recharge si absent ou si le code est simplifié.
                        logging.info(f"✓ CACHE: Utilisation de la version en cache pour: {page_title}")
                        return self._reload_docs_from_cache(cache_info)
                except Exception as e:
                    logging.warning(f"Erreur lors de la vérification du cache pour {page_title}: {e}")

            # Si pas de cache ou version changée => on charge
            try:
                # Surcharge du warning de langchain en créant un nouveau loader avec page_ids
                temp_loader = ConfluenceLoader(
                    url=self.confluence_url,
                    username=self.username,
                    api_key=self.api_key,
                    space_key=self.space_key,
                    page_ids=[pid]
                )
                docs = temp_loader.load()
                logging.info(f"✓ SUCCÈS: Page chargée: {page_title} ({len(docs)} chunks)")
                
                # On stocke dans le cache
                if docs:
                    try:
                        page_data = self._fetch_page_info(pid)
                        version_number = page_data["version"]["number"]
                        # Stocker 'version_number' + 'page_content' (simplifié)
                        # Pour un vrai cache, on stocke tout le contenu, plus la metadata...
                        self._page_cache[pid] = {
                            "version": version_number,
                            "content": [doc.page_content for doc in docs],
                            "metadata": [doc.metadata for doc in docs]
                        }
                    except Exception as e:
                        logging.warning(f"Erreur lors de la mise en cache de {page_title}: {e}")
                
                return docs
            except Exception as e:
                logging.error(f"❌ ÉCHEC: Erreur lors du chargement de {page_title}: {str(e)}")
                return []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(0, len(page_ids), batch_size):
                batch = page_ids[i:i+batch_size]
                logging.info(f"Soumission du batch {i//batch_size + 1}/{(len(page_ids) + batch_size - 1)//batch_size} ({len(batch)} pages)")
                for pid in batch:
                    futures.append(executor.submit(load_one_page, pid))

            loaded_pages_count = 0
            for future in as_completed(futures):
                try:
                    docs_result = future.result()
                    documents.extend(docs_result)
                    loaded_pages_count += 1
                    if loaded_pages_count % 10 == 0:
                        logging.info(f"PROGRESSION: {loaded_pages_count}/{len(page_ids)} pages traitées")
                except Exception as e:
                    logging.error(f"Erreur lors du traitement d'une page: {e}")

        logging.info(f"Chargement terminé: {loaded_pages_count}/{len(page_ids)} pages chargées avec succès")
        return documents

    def _fetch_page_info(self, pid: str) -> Dict[str, Any]:
        """Récupère des infos basiques sur une page (pour la version, etc.)."""
        endpoint = f"{self.confluence_url}/rest/api/content/{pid}?expand=version"
        resp = requests.get(endpoint, auth=(self.username, self.api_key))
        if resp.status_code != 200:
            raise ValueError(f"Impossible de récupérer la page {pid}")
        return resp.json()

    def _reload_docs_from_cache(self, cache_info) -> List[Document]:
        """Reconstruit des Documents depuis le cache (exemple simple)."""
        docs = []
        contents = cache_info.get("content", [])
        metadatas = cache_info.get("metadata", [])
        for content, md in zip(contents, metadatas):
            docs.append(
                Document(
                    page_content=content,
                    metadata=md
                )
            )
        return docs

    def _load_all_and_filter(self):
        """
        Méthode fallback : charge toutes les pages puis filtre.
        Evite de tout perdre si la méthode principale échoue.
        """
        try:
            loader = ConfluenceLoader(
                url=self.confluence_url,
                username=self.username,
                api_key=self.api_key,
                space_key=self.space_key
            )
            
            logging.info("Chargement de toutes les pages de l'espace...")
            all_docs = loader.load(limit=200)
            
            logging.info(f"Filtrage des {len(all_docs)} pages chargées...")
            filtered_docs = []
            for doc in all_docs:
                content = doc.page_content.lower()
                if any(keyword in content for keyword in ["opcon", "scheduler", "job", "monitoring"]):
                    filtered_docs.append(doc)
            
            logging.info(f"Pages pertinentes trouvées: {len(filtered_docs)}")
            return filtered_docs
            
        except Exception as e:
            logging.error(f"Erreur fallback: {e}")
            return []

    def split_docs(self, docs):
        """
        Découpe les documents en sous-chunks plus précis.
        Amélioré pour créer des chunks plus cohérents.
        """
        headers_to_split_on = [
            ("#", "Titre 1"),
            ("##", "Sous-titre 1"),
            ("###", "Sous-titre 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        md_docs = []
        for doc in docs:
            # Booster certains docs contenant des mots-clés pertinents
            boost_keywords = ["opcon", "scheduler", "job", "monitoring", "m3", "cegid", "contrôle", "article", "procédure","anaplan","talend","microsoft","azure","aws","google","cloud"]
            
            if any(kw in doc.page_content.lower() for kw in boost_keywords):
                # Déterminer le niveau de boost basé sur le nombre de mots-clés présents
                matches = sum(1 for kw in boost_keywords if kw in doc.page_content.lower())
                boost_factor = 1.0 + (matches * 0.05)  # +5% par mot-clé trouvé, max +45%
                doc.metadata["relevance_boost"] = min(1.5, boost_factor)  # Plafonner à +50%

            try:
                # Tenter de diviser avec le splitter Markdown
                splitted_md = markdown_splitter.split_text(doc.page_content)
                
                # Si aucun en-tête Markdown n'a été trouvé, créer un seul document
                if not splitted_md:
                    splitted_md = [doc]
                
                for chunk in splitted_md:
                    # Fusionner la metadata du chunk et du doc original
                    if hasattr(chunk, 'metadata'):
                        chunk.metadata = {**doc.metadata, **chunk.metadata}
                    else:
                        # Si ce n'est pas un Document mais du texte, créer un nouveau Document
                        from langchain_core.documents import Document
                        chunk = Document(
                            page_content=chunk,
                            metadata=doc.metadata
                        )
                
                md_docs.extend(splitted_md)
            except Exception as e:
                logger.warning(f"Erreur lors du découpage Markdown, conservation du document entier: {e}")
                md_docs.append(doc)
        
        # Ensuite, un RecursiveCharacterTextSplitter avec des paramètres optimisés
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Augmenté pour avoir plus de contexte par chunk
            chunk_overlap=200,  # Augmenté pour éviter de perdre le contexte entre chunks
            separators=["\n\n", "\n", ".", ";", ",", " ", ""]
        )
        
        try:
            splitted_docs = splitter.split_documents(md_docs)
            # Assurer que tous les documents ont un titre
            for doc in splitted_docs:
                if "title" not in doc.metadata:
                    # Extraire un titre du contenu si possible
                    first_line = doc.page_content.strip().split('\n')[0][:50]
                    doc.metadata["title"] = first_line + "..." if len(first_line) >= 50 else first_line
            return splitted_docs
        except Exception as e:
            logger.error(f"Erreur lors du découpage final: {e}")
            # En cas d'erreur, retourner les documents après le premier découpage
            return md_docs

    def save_to_db(self, splitted_docs, embeddings):
        """
        Sauvegarde les chunks dans Chroma DB.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
        
        # In newer versions of Chroma, data is automatically persisted when 
        # using persist_directory - no need to call persist() explicitly
        db = Chroma.from_documents(
            documents=splitted_docs,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        
        # No need to call db.persist() - it happens automatically
        # when using persist_directory parameter
        
        logger.info(f"Saved {len(splitted_docs)} documents to Chroma DB at {self.persist_directory}")
        return db

    def load_from_db(self, embeddings):
        """
        Charge la DB depuis le persist_directory (Chroma).
        """
        # No need to import Chroma here
        if not os.path.exists(self.persist_directory):
            logging.warning(f"Vector DB directory {self.persist_directory} absent.")
            os.makedirs(self.persist_directory, exist_ok=True)
            
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        return db

    def set_db(self, embeddings, test_mode=False):
        """Crée et persiste la DB (mode test ou complet) seulement si elle n'existe pas déjà"""
        # Vérifier si la base existe déjà
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"Base de données existante trouvée dans: {self.persist_directory}")
            
            # Log des fichiers présents dans le répertoire
            files = os.listdir(self.persist_directory)
            logger.info(f"Fichiers dans le répertoire: {files}")
            
            try:
                # Charger la base existante
                db = self.get_db(embeddings)
                
                # Tenter de compter les éléments
                try:
                    count = db._collection.count()
                    logger.info(f"Nombre d'éléments dans la base existante: {count}")
                except Exception as e:
                    logger.warning(f"Impossible de compter les éléments: {e}")
                
                return db
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la base existante: {e}")
                logger.warning("Création d'une nouvelle base...")
        else:
            logger.info(f"Aucune base existante trouvée dans: {self.persist_directory}")
            logger.info("Création d'une nouvelle base...")

        # S'assurer que self.page_ids contient toujours des valeurs
        if not self.page_ids:
            logger.warning(f"SÉCURITÉ: Aucun ID de page spécifié. Utilisation des {len(DEFAULT_PAGE_IDS)} pages par défaut")
            self.page_ids = DEFAULT_PAGE_IDS
            
        # Chargement limité aux pages spécifiques
        logger.info(f"Chargement limité à {len(self.page_ids)} pages spécifiques")
        logger.info(f"Liste des IDs: {self.page_ids}")
        docs = self.load_from_specific_pages()
        
        # Log pour vérifier combien de documents ont réellement été chargés
        logger.info(f"Documents chargés: {len(docs) if docs else 0}")
        
        # Log détaillé pour débugger
        if docs:
            titles = [doc.metadata.get('title', 'Sans titre') for doc in docs]
            unique_titles = set(titles)
            logger.info(f"Nombre de titres uniques: {len(unique_titles)}")
            logger.info(f"Premiers titres chargés (max 10): {list(unique_titles)[:10]}")
            if len(unique_titles) > 10:
                logger.info(f"... et {len(unique_titles) - 10} autres titres")
        else:
            logger.error("ERREUR CRITIQUE: Aucun document n'a été chargé!")
            
        # Si aucun document n'est chargé, utiliser des documents fictifs
        if not docs:
            logger.warning("Aucun document chargé. Utilisation de documents fictifs pour le test.")
            docs = self._create_fallback_test_docs()
        
        logger.info("Découpage des documents...")
        splitted_docs = self.split_docs(docs)
        logger.info(f"Nombre de chunks après découpage: {len(splitted_docs)}")
        
        logger.info("Création de la base vectorielle...")
        db = self.save_to_db(splitted_docs, embeddings)
        logger.info("Base de données créée avec succès!")
        
        return db

    def get_db(self, embeddings):
        """Charge la DB existante sans la recréer."""
        db = self.load_from_db(embeddings)
        return db

    def get_retriever(self, embeddings):
        """Crée un retriever simple sans réordonnancement"""
        db = self.get_db(embeddings)
        
        # Retriever simple sans reranking
        retriever = db.as_retriever(
            search_type="similarity",  # Revert to basic similarity search
            search_kwargs={"k": 20}  # Increase to get more documents for reranking
        )
        
        logger.info(f"Retriever configuré avec: search_type=similarity, k=20")
        return retriever

    def load_test_pages(self, page_ids=None):
        """Charge seulement 1 ou 2 pages spécifiques pour test rapide"""
        if page_ids is None:
            # Au lieu d'utiliser juste l'ID d'intégration, chargeons toutes les pages de l'espace
            try:
                print("Chargement de quelques pages de test depuis l'espace...")
                loader = ConfluenceLoader(
                    url=self.confluence_url,
                    username=self.username,
                    api_key=self.api_key,
                    space_key=self.space_key,
                    limit=5  # Augmenté à 10 pour avoir plus d'exemples
                )
                
                # Charger les 10 premières pages de l'espace
                test_docs = loader.load()
                print(f"Pages trouvées: {len(test_docs)}")
                
                # Afficher les titres des pages chargées
                print("\n=== Titres des pages chargées ===")
                for i, doc in enumerate(test_docs):
                    title = doc.metadata.get("title", "Sans titre")
                    print(f"{i+1}. {title}")
                    # Optionnel: afficher les premiers 100 caractères du contenu pour mieux comprendre
                    preview = doc.page_content[:100].replace("\n", " ").strip() + "..."
                    print(f"   Aperçu: {preview}\n")
                
                # Si aucune page n'est trouvée, créons des documents fictifs pour le test
                if not test_docs:
                    print("Aucune page trouvée. Création de documents fictifs pour le test...")
                    test_docs = self._create_fallback_test_docs()
                
                return test_docs
                
            except Exception as e:
                print(f"Erreur lors du chargement des pages de test: {e}")
                # En cas d'erreur, créer des documents fictifs
                return self._create_fallback_test_docs()
        
        # Le reste de votre code existant...

    def _create_fallback_test_docs(self):
        """Crée des documents fictifs en cas d'échec de chargement depuis Confluence"""
        print("Création de documents fictifs pour le test...")
        return [
            Document(
                page_content="Document de test pour OpCon. Ce document contient des informations sur le monitoring des schedulers.",
                metadata={"source": "test", "title": "Test OpCon"}
            ),
            Document(
                page_content="Guide d'utilisation OpCon. Ce document explique comment gérer les jobs et résoudre les problèmes courants.",
                metadata={"source": "test", "title": "Guide OpCon"}
            ),
            Document(
                page_content="Procédures de maintenance pour les schedulers. Document technique pour le support L2.",
                metadata={"source": "test", "title": "Maintenance Scheduler"}
            )
        ]

    def load_from_specific_pages(self):
        """
        Charge uniquement les pages spécifiées dans self.page_ids.
        Cette méthode permet de charger précisément les pages dont on a besoin.
        """
        if not self.page_ids:
            logging.error("ERREUR: Aucun ID de page spécifié!")
            return []

        try:
            # Créer le loader de base
            loader = ConfluenceLoader(
                url=self.confluence_url,
                username=self.username,
                api_key=self.api_key
            )
            
            logging.info(f"Chargement de {len(self.page_ids)} pages spécifiques...")
            logging.info("=== PAGES À CHARGER (LISTE D'IDS FOURNIE) ===")
            
            # Précharger les titres pour le logging
            page_titles = {}
            valid_page_ids = []
            
            for pid in self.page_ids:
                try:
                    # Nettoyer l'ID pour assurer qu'il est valide
                    pid = str(pid).strip()
                    if not pid.isdigit():
                        pid = ''.join(c for c in pid if c.isdigit())
                    
                    if not pid:
                        logging.warning(f"ID de page ignoré car non numérique")
                        continue
                        
                    # Vérifier si la page existe avant de la charger
                    try:
                        page_data = self._fetch_page_info(pid)
                        title = page_data.get("title", f"Page ID: {pid}")
                        page_titles[pid] = title
                        valid_page_ids.append(pid)
                        logging.info(f"Page trouvée: {title} (ID: {pid})")
                    except Exception as e:
                        logging.error(f"Erreur lors de la récupération de la page {pid}: {e}")
                        logging.warning(f"La page {pid} sera ignorée")
                except Exception as e:
                    logging.warning(f"Impossible de traiter l'ID de page {pid}: {e}")
            
            if not valid_page_ids:
                logging.error("ERREUR CRITIQUE: Aucune page valide trouvée parmi les IDs fournis!")
                return []
            
            logging.info(f"Pages valides à charger: {len(valid_page_ids)}/{len(self.page_ids)}")
            
            # Charger les pages avec la méthode de chargement concurrent
            all_docs = []
            
            # Méthode alternative pour charger directement les pages via l'API de ConfluenceLoader
            for pid in valid_page_ids:
                try:
                    logging.info(f"Chargement direct de la page {pid} ({page_titles.get(pid, 'Sans titre')})...")
                    page_docs = loader.load(page_ids=[pid])
                    if page_docs:
                        logging.info(f"✓ Page {pid} chargée avec succès: {len(page_docs)} chunks")
                        all_docs.extend(page_docs)
                    else:
                        logging.warning(f"⚠ Page {pid} chargée mais ne contient aucun document")
                except Exception as e:
                    logging.error(f"❌ Erreur lors du chargement de la page {pid}: {e}")
            
            logging.info(f"Chargement terminé. Total des documents: {len(all_docs)}")
            
            # Récapitulatif des pages chargées
            titles = [doc.metadata.get('title', 'Sans titre') for doc in all_docs]
            unique_titles = set(titles)
            logging.info(f"Pages uniques chargées: {len(unique_titles)}")
            
            # Afficher tous les titres uniques
            unique_titles_list = sorted(list(unique_titles))
            logging.info("=== LISTE COMPLÈTE DES PAGES CHARGÉES ===")
            for i, title in enumerate(unique_titles_list):
                logging.info(f"Page {i+1}/{len(unique_titles_list)}: {title}")
            
            return all_docs
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement des pages spécifiques: {e}")
            return []


if __name__ == "__main__":
    # Test rapide pour vérifier le chargement des pages par défaut
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    print("Test de chargement des pages spécifiques par défaut...")
    loader = DataLoader(use_default_pages=True)
    embeddings = OpenAIEmbeddings()
    
    # Vérifier si la base existe et la supprimer pour le test (optionnel)
    if "--reset" in sys.argv and os.path.exists(PERSIST_DIRECTORY):
        print(f"Suppression de la base existante: {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)
    
    db = loader.set_db(embeddings)
    
    try:
        doc_count = db._collection.count()
        print(f"Base créée avec succès. Nombre de documents: {doc_count}")
    except Exception as e:
        print(f"Erreur lors de la vérification de la base: {e}")
