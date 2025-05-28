# test_confluence.py
import os
import sys
sys.path.append('../')
from src.config import (CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
                    CONFLUENCE_USERNAME, CONFLUENCE_API_KEY)

from langchain_community.document_loaders import ConfluenceLoader
import re

def get_integration_pages():
    """Récupère uniquement les pages de la section Integration"""
    print("=== Récupération des pages de la section Integration ===")
    print(f"URL: {CONFLUENCE_SPACE_NAME}")
    print(f"Utilisateur: {CONFLUENCE_USERNAME}")
    print(f"Espace ciblé: {CONFLUENCE_SPACE_KEY}")
    
    try:
        # 1. Charger l'ensemble des pages de l'espace
        loader = ConfluenceLoader(
            url=CONFLUENCE_SPACE_NAME,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_KEY,
            space_key=CONFLUENCE_SPACE_KEY
        )
        
        print(f"Récupération des pages de l'espace {CONFLUENCE_SPACE_KEY}...")
        all_pages = loader.load()
        print(f"Nombre total de pages dans l'espace: {len(all_pages)}")
        
        if not all_pages:
            print("❌ Aucune page trouvée dans cet espace.")
            return []
        
        # 2. Trouver la page "Integration" directement
        integration_page = None
        for page in all_pages:
            if page.metadata.get("title") == "Integration":
                integration_page = page
                break
        
        if not integration_page:
            print("❌ Page 'Integration' non trouvée dans l'espace.")
            return []
        
        # Extraire l'ID de la page Integration
        integration_page_id = extract_page_id(integration_page.metadata.get("source"))
        print(f"Page 'Integration' trouvée (ID: {integration_page_id})")
        
        # 3. Récupérer toutes les sous-pages d'Integration
        print("Chargement des sous-pages de 'Integration'...")
        
        integration_loader = ConfluenceLoader(
            url=CONFLUENCE_SPACE_NAME,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_KEY,
            space_key=CONFLUENCE_SPACE_KEY  # Ajoutez ce paramètre obligatoire ici
        )
        
        # Vérifier la signature de la méthode load pour utiliser le bon paramètre
        try:
            # Essayer d'abord avec parent_id (versions plus récentes de langchain)
            integration_children = integration_loader.load(parent_id=integration_page_id)
        except TypeError:
            # Si cela échoue, essayer avec page_id (versions plus anciennes)
            try:
                integration_children = integration_loader.load(page_id=integration_page_id)
            except TypeError:
                # Si cela échoue aussi, essayer sans paramètre puis filtrer manuellement
                print("Attention: Impossible d'utiliser parent_id ou page_id, chargement de toutes les pages...")
                all_children = integration_loader.load()
                
                # Filtrer manuellement les pages qui ont Integration comme parent
                integration_children = []
                for page in all_children:
                    page_url = page.metadata.get("source", "")
                    # Vérifier si l'URL contient l'ID de la page Integration
                    if integration_page_id in page_url:
                        integration_children.append(page)
        
        if not integration_children:
            print("❌ Aucune sous-page trouvée sous 'Integration'.")
            return []
        
        print(f"Nombre de sous-pages trouvées sous 'Integration': {len(integration_children)}")
        
        # 4. Récupérer récursivement toutes les sous-pages
        all_integration_pages = []
        all_integration_pages.append(integration_page)  # Ajouter la page Integration elle-même
        all_integration_pages.extend(integration_children)
        
        # Utiliser une fonction récursive pour récupérer les sous-pages à tous les niveaux
        def get_all_children_recursive(parent_pages, visited_ids=None):
            if visited_ids is None:
                visited_ids = set()
            
            result = []
            
            for parent_page in parent_pages:
                parent_id = extract_page_id(parent_page.metadata.get("source"))
                
                # Éviter les boucles infinies en vérifiant si la page a déjà été visitée
                if parent_id and parent_id not in visited_ids:
                    visited_ids.add(parent_id)
                    
                    try:
                        child_loader = ConfluenceLoader(
                            url=CONFLUENCE_SPACE_NAME,
                            username=CONFLUENCE_USERNAME,
                            api_key=CONFLUENCE_API_KEY,
                            space_key=CONFLUENCE_SPACE_KEY  # Ajoutez ce paramètre obligatoire ici
                        )
                        
                        # Utiliser la même méthode de chargement que précédemment
                        try:
                            # Essayer d'abord avec parent_id
                            children = child_loader.load(parent_id=parent_id)
                        except TypeError:
                            try:
                                # Essayer avec page_id
                                children = child_loader.load(page_id=parent_id)
                            except TypeError:
                                # Sinon, charger toutes les pages et filtrer
                                all_pages = child_loader.load()
                                children = []
                                for page in all_pages:
                                    page_url = page.metadata.get("source", "")
                                    if parent_id in page_url:
                                        children.append(page)
                        
                        if children:
                            print(f"Trouvé {len(children)} sous-pages sous '{parent_page.metadata.get('title')}'")
                            result.extend(children)
                            
                            # Appel récursif pour récupérer les sous-pages des sous-pages
                            deeper_children = get_all_children_recursive(children, visited_ids)
                            result.extend(deeper_children)
                    except Exception as e:
                        print(f"Erreur lors de la récupération des sous-pages de {parent_page.metadata.get('title')}: {e}")
            
            return result
        
        # Récupérer toutes les sous-pages de manière récursive
        additional_pages = get_all_children_recursive(integration_children)
        all_integration_pages.extend(additional_pages)
        
        print(f"Nombre total de pages dans la section Integration: {len(all_integration_pages)}")
        return all_integration_pages
    
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return []

def get_integration_pages_fixed():
    """Récupère les pages de la section Integration avec la méthode corrigée"""
    print("=== Récupération des pages de la section Integration (version corrigée) ===")
    print(f"URL: {CONFLUENCE_SPACE_NAME}")
    print(f"Utilisateur: {CONFLUENCE_USERNAME}")
    print(f"Espace ciblé: {CONFLUENCE_SPACE_KEY}")
    
    try:
        # 1. Charger l'ensemble des pages de l'espace pour trouver la page Integration
        loader = ConfluenceLoader(
            url=CONFLUENCE_SPACE_NAME,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_KEY,
            space_key=CONFLUENCE_SPACE_KEY
        )
        
        print(f"Récupération des pages de l'espace {CONFLUENCE_SPACE_KEY}...")
        all_pages = loader.load()
        print(f"Nombre total de pages dans l'espace: {len(all_pages)}")
        
        if not all_pages:
            print("❌ Aucune page trouvée dans cet espace.")
            return []
        
        # 2. Trouver la page "Integration" directement
        integration_page = None
        for page in all_pages:
            if page.metadata.get("title") == "Integration":
                integration_page = page
                break
        
        if not integration_page:
            print("❌ Page 'Integration' non trouvée dans l'espace.")
            return []
        
        # Extraire l'ID de la page Integration
        integration_page_id = extract_page_id(integration_page.metadata.get("source"))
        print(f"Page 'Integration' trouvée (ID: {integration_page_id})")
        
        # 3. Utiliser une approche alternative pour récupérer les sous-pages
        # Au lieu d'utiliser parent_id, nous allons filtrer manuellement les pages
        print("Filtrage manuel des pages sous 'Integration'...")
        
        # Créer une liste pour stocker toutes les pages liées à Integration
        integration_pages = [integration_page]  # Inclure la page d'intégration elle-même
        
        # Utiliser une méthode de filtrage basée sur le contenu des URL ou les titres
        for page in all_pages:
            page_url = page.metadata.get("source", "")
            page_title = page.metadata.get("title", "")
            
            # Vérifier si l'URL contient le chemin vers Integration
            if f"/pages/{integration_page_id}/" in page_url:
                integration_pages.append(page)
                continue
                
            # Vérifier si le titre indique une sous-page
            if page_title.startswith("Integration/") or " - Integration" in page_title:
                integration_pages.append(page)
                continue
                
            # Vérifier si le contenu fait référence à la page d'intégration
            if integration_page_id in page.page_content:
                integration_pages.append(page)
        
        # Dédupliquer les pages (au cas où)
        unique_urls = set()
        unique_pages = []
        
        for page in integration_pages:
            url = page.metadata.get("source", "")
            if url not in unique_urls:
                unique_urls.add(url)
                unique_pages.append(page)
        
        print(f"Nombre de pages liées à Integration trouvées: {len(unique_pages)}")
        return unique_pages
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return []

def extract_page_id(url):
    """Extrait l'ID de page à partir de l'URL Confluence"""
    if not url:
        return None
    
    # Pattern pour une URL de page Confluence: .../pages/123456/titre-de-la-page
    match = re.search(r'/pages/(\d+)/', url)
    if match:
        return match.group(1)
    
    # Alternative: chercher simplement un nombre dans l'URL
    parts = url.split('/')
    for part in parts:
        if part.isdigit():
            return part
    
    return None

def filter_integration_pages(pages, keywords=None):
    """Filtrer les pages selon des mots-clés spécifiques"""
    if keywords is None:
        # Mots-clés par défaut liés à l'intégration
        keywords = ["integration", "api", "ETL", "foundation", "migration"]
    
    filtered_pages = []
    for page in pages:
        page_title = page.metadata.get("title", "").lower()
        page_content = page.page_content.lower()
        
        # Vérifier si un des mots-clés est dans le titre ou le contenu
        if any(kw.lower() in page_title or kw.lower() in page_content for kw in keywords):
            filtered_pages.append(page)
    
    return filtered_pages

def get_integration_pages_by_title():
    """Version alternative qui récupère toutes les pages et filtre par titre"""
    try:
        # Charger toutes les pages de l'espace
        loader = ConfluenceLoader(
            url=CONFLUENCE_SPACE_NAME,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_KEY,
            space_key=CONFLUENCE_SPACE_KEY
        )
        
        all_pages = loader.load()
        print(f"Nombre total de pages chargées: {len(all_pages)}")
        
        # Filtrer les pages dont le titre contient "Integration" ou qui sont
        # dans un chemin contenant "Integration"
        integration_pages = []
        
        for page in all_pages:
            title = page.metadata.get("title", "")
            url = page.metadata.get("source", "")
            
            # Recherche plus précise pour Intégration
            if "Integration" in title:
                print(f"Page trouvée par titre: {title}")
                integration_pages.append(page)
                continue
                
            # Vérifier si la page est dans le chemin de Integration
            # Chercher le motif "/Integration/" dans l'URL (indépendamment de la casse)
            if "/Integration/" in url or "/integration/" in url:
                print(f"Page trouvée par URL: {title} ({url})")
                integration_pages.append(page)
                continue
                
            # Vérifier également le chemin dans la structure de Confluence (si disponible)
            ancestor_titles = page.metadata.get("ancestor_titles", [])
            if ancestor_titles and "Integration" in ancestor_titles:
                print(f"Page trouvée par ancêtre: {title}")
                integration_pages.append(page)
                continue
            
            # Recherche dans le contenu pour les références à Integration
            if "Integration" in page.page_content and (
                "section Integration" in page.page_content.lower() or
                "space integration" in page.page_content.lower()
            ):
                print(f"Page trouvée par contenu: {title}")
                integration_pages.append(page)
        
        print(f"Nombre de pages liées à Integration: {len(integration_pages)}")
        
        # Afficher quelques statistiques sur les pages trouvées
        domains = {}
        for page in integration_pages:
            url = page.metadata.get("source", "")
            # Extraire le domaine de l'URL
            if "://" in url:
                domain = url.split("://")[1].split("/")[0]
                domains[domain] = domains.get(domain, 0) + 1
        
        print("Pages par domaine:")
        for domain, count in domains.items():
            print(f"  {domain}: {count} pages")
        
        return integration_pages
    
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return []

def display_page_info(pages):
    """Affiche les informations des pages"""
    for i, page in enumerate(pages):
        title = page.metadata.get("title", "Sans titre")
        url = page.metadata.get("source", "URL inconnue")
        
        print(f"{i+1}. {title}")
        print(f"   URL: {url}")
        
        # Aperçu du contenu
        content_preview = page.page_content[:150].replace('\n', ' ').strip() if page.page_content else ""
        if content_preview:
            print(f"   Aperçu: {content_preview}...")
        
        print()

def main():
    # 1. Récupérer les pages de la section Integration
    print("Récupération des pages de la section Integration (tous niveaux)...")
    integration_pages = get_integration_pages()
    
    if not integration_pages:
        # Si la méthode principale échoue, essayer la méthode alternative #1
        print("Tentative de récupération alternative #1 des pages d'intégration...")
        integration_pages = get_integration_pages_by_title()
        
        if not integration_pages:
            # Si la méthode alternative #1 échoue, essayer la méthode corrigée
            print("Tentative de récupération alternative #2 (méthode corrigée)...")
            integration_pages = get_integration_pages_fixed()
    
    if integration_pages:
        # 2. Afficher des statistiques sur les pages trouvées
        print("\n=== Statistiques des pages trouvées ===")
        print(f"Nombre total de pages: {len(integration_pages)}")
        
        # Compter les pages par niveau de profondeur
        page_levels = {}
        for page in integration_pages:
            url = page.metadata.get("source", "")
            # Compter le nombre de segments dans l'URL après "pages/"
            if "/pages/" in url:
                segments = url.split("/pages/")[1].strip("/").split("/")
                depth = len(segments) - 1  # -1 car le premier segment est l'ID de page
                page_levels[depth] = page_levels.get(depth, 0) + 1
        
        print("Répartition par niveau de profondeur:")
        for depth, count in sorted(page_levels.items()):
            print(f"  Niveau {depth}: {count} pages")
        
        # 3. Option pour afficher toutes les pages ou seulement certaines
        print("\nOptions disponibles:")
        print("1. Afficher toutes les pages")
        print("2. Filtrer les pages par mots-clés")
        print("3. Afficher uniquement les pages d'un certain niveau")
        choice = input("Choisissez une option (1-3): ")
        
        if choice == "1":
            print("\n=== Toutes les pages trouvées ===")
            display_page_info(integration_pages)
        elif choice == "2":
            print("Entrez les mots-clés séparés par des virgules :")
            keywords_input = input()
            keywords = [k.strip() for k in keywords_input.split(',')]
            
            filtered_pages = filter_integration_pages(integration_pages, keywords)
            
            print(f"\n=== Pages filtrées ({len(filtered_pages)}/{len(integration_pages)}) ===")
            display_page_info(filtered_pages)
        elif choice == "3":
            level = input("Entrez le niveau de profondeur à afficher (0, 1, 2, etc.): ")
            try:
                level_num = int(level)
                level_pages = []
                
                for page in integration_pages:
                    url = page.metadata.get("source", "")
                    if "/pages/" in url:
                        segments = url.split("/pages/")[1].strip("/").split("/")
                        depth = len(segments) - 1
                        if depth == level_num:
                            level_pages.append(page)
                
                print(f"\n=== Pages de niveau {level_num} ({len(level_pages)}/{len(integration_pages)}) ===")
                display_page_info(level_pages)
            except ValueError:
                print("Niveau invalide. Affichage de toutes les pages.")
                display_page_info(integration_pages)
        
        return integration_pages
    else:
        print("Aucune page d'intégration trouvée.")
        return []

if __name__ == "__main__":
    main()