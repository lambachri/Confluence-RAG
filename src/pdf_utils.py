import os
import logging
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

logger = logging.getLogger(__name__)

def load_pdf(file_path, extract_images=True):
    """Charge un fichier PDF et retourne des documents"""
    if not os.path.exists(file_path):
        logger.error(f"Le fichier PDF n'existe pas: {file_path}")
        return []
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Ajouter des métadonnées utiles
        file_name = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["title"] = file_name
            doc.metadata["type"] = "pdf"
        
        logger.info(f"PDF chargé avec succès: {file_path} - {len(documents)} pages")
        
        # Si extraction d'images est activée, essayer d'extraire le texte des images dans le PDF
        if extract_images:
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                unstructured_loader = UnstructuredPDFLoader(file_path, mode="elements")
                unstructured_docs = unstructured_loader.load()
                
                # Si on a plus de documents avec UnstructuredPDFLoader, c'est probablement 
                # parce qu'il a réussi à extraire le texte des images
                if len(unstructured_docs) > len(documents):
                    logger.info(f"Extraction d'images réussie: {len(unstructured_docs)} éléments extraits")
                    
                    # Mettre à jour les métadonnées
                    for doc in unstructured_docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["title"] = file_name
                        doc.metadata["type"] = "pdf_with_ocr"
                    
                    return unstructured_docs
            except Exception as e:
                logger.warning(f"Impossible d'extraire le texte des images: {e}")
        
        return documents
    except Exception as e:
        logger.error(f"Erreur lors du chargement du PDF {file_path}: {e}")
        return []

def load_image(file_path):
    """Charge une image et extrait son texte"""
    if not os.path.exists(file_path):
        logger.error(f"Le fichier image n'existe pas: {file_path}")
        return []
    
    try:
        loader = UnstructuredImageLoader(file_path, mode="elements")
        documents = loader.load()
        
        # Ajouter des métadonnées utiles
        file_name = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["title"] = file_name
            doc.metadata["type"] = "image"
        
        logger.info(f"Image chargée avec succès: {file_path} - {len(documents)} éléments extraits")
        return documents
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'image {file_path}: {e}")
        return []

def split_documents(documents, chunk_size=1000, chunk_overlap=300):
    """
    Découpe les documents en chunks pour l'indexation, avec une stratégie améliorée
    pour préserver le contexte et la structure logique du contenu.
    """
    # Utiliser un découpage plus intelligent qui respecte autant que possible les paragraphes
    splitter = RecursiveCharacterTextSplitter(
        # Augmenter le chevauchement pour plus de contexte entre chunks
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Respecter la structure du document (paragraphes et sections)
        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ":", " ", ""],
        # Essayer de garder ensemble les listes et sections numérotées
        keep_separator=True
    )
    
    # Découper les documents
    chunks = splitter.split_documents(documents)
    
    # Vérification de la cohérence des chunks produits
    coherent_chunks = ensure_chunks_coherence(chunks)
    
    return coherent_chunks

def ensure_chunks_coherence(chunks):
    """
    Vérifie et améliore la cohérence des chunks produits, en particulier
    pour les listes à puces et les procédures numérotées.
    """
    coherent_chunks = []
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        
        # Vérifier si le chunk commence au milieu d'une liste ou procédure
        starts_with_item = (
            re.match(r'^\s*\d+\..*', content) or  # Commence par un nombre suivi d'un point
            re.match(r'^\s*[\*\-\•].*', content)   # Commence par un caractère de liste à puces
        )
        
        # Vérifier si le chunk se termine au milieu d'une liste
        ends_with_incomplete_sentence = re.search(r'[^\.\!\?]\s*$', content)
        
        # Si problème détecté et qu'il y a un chunk précédent, essayer de combiner
        if i > 0 and (starts_with_item or ends_with_incomplete_sentence):
            prev_chunk = coherent_chunks[-1] if coherent_chunks else None
            
            if prev_chunk and prev_chunk.metadata.get('source') == chunk.metadata.get('source'):
                # Combiner avec le chunk précédent si possible
                merged_content = prev_chunk.page_content + " " + content
                
                # Vérifier si la taille du chunk combiné reste raisonnable
                if len(merged_content) <= 2000:  # Taille maximale raisonnable
                    # Mise à jour du chunk précédent
                    prev_chunk.page_content = merged_content
                    continue  # Sauter ce chunk car il a été fusionné
        
        # Ajouter le chunk tel quel s'il n'a pas été fusionné
        coherent_chunks.append(chunk)
    
    return coherent_chunks

def load_directory(directory_path, file_extensions=None):
    """Charge tous les fichiers d'un répertoire"""
    if file_extensions is None:
        file_extensions = [".pdf", ".png", ".jpg", ".jpeg"]
    
    if not os.path.exists(directory_path):
        logger.error(f"Le répertoire n'existe pas: {directory_path}")
        return []
    
    all_documents = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if not os.path.isfile(file_path):
            continue
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext == ".pdf":
                docs = load_pdf(file_path)
            elif file_ext in [".png", ".jpg", ".jpeg"]:
                docs = load_image(file_path)
            else:
                logger.warning(f"Type de fichier non supporté: {file_ext}")
                continue
                
            all_documents.extend(docs)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path}: {e}")
    
    logger.info(f"Total de documents chargés depuis {directory_path}: {len(all_documents)}")
    return all_documents

# Fonctions utilitaires supplémentaires

def get_pdf_metadata(file_path):
    """Extrait les métadonnées d'un PDF"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata
            num_pages = len(reader.pages)
            
            metadata = {
                "title": info.get('/Title', os.path.basename(file_path)),
                "author": info.get('/Author', 'Inconnu'),
                "subject": info.get('/Subject', ''),
                "creator": info.get('/Creator', ''),
                "producer": info.get('/Producer', ''),
                "num_pages": num_pages,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path)
            }
            
            return metadata
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des métadonnées du PDF {file_path}: {e}")
        return {
            "title": os.path.basename(file_path),
            "error": str(e),
            "file_path": file_path
        }

def batch_process_directory(directory_path, output_dir=None, file_extensions=None, recursive=False):
    """Traite tous les fichiers d'un répertoire et sauvegarde les résultats"""
    if output_dir is None:
        output_dir = os.path.join(directory_path, "processed")
    
    if file_extensions is None:
        file_extensions = [".pdf", ".png", ".jpg", ".jpeg"]
        
    os.makedirs(output_dir, exist_ok=True)
    
    all_documents = []
    processed_files = 0
    
    def process_file(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in file_extensions:
            try:
                if file_ext == ".pdf":
                    docs = load_pdf(file_path)
                elif file_ext in [".png", ".jpg", ".jpeg"]:
                    docs = load_image(file_path)
                else:
                    return []
                
                return docs
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {file_path}: {e}")
        return []
    
    if recursive:
        # Parcourir récursivement tous les sous-répertoires
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                docs = process_file(file_path)
                all_documents.extend(docs)
                processed_files += 1 if docs else 0
    else:
        # Traiter uniquement les fichiers du répertoire principal
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                docs = process_file(file_path)
                all_documents.extend(docs)
                processed_files += 1 if docs else 0
    
    # Découper les documents en chunks
    chunked_docs = split_documents(all_documents)
    
    # On pourrait sauvegarder les chunks dans un format sérialisé pour réutilisation
    # Par exemple, en JSON pour les visualiser facilement
    result_path = os.path.join(output_dir, "processed_documents.json")
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            import json
            # Sérialiser les documents (simplifiés)
            serialized_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in chunked_docs
            ]
            json.dump(serialized_docs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des résultats: {e}")
    
    logger.info(f"Traitement terminé: {processed_files} fichiers traités, {len(chunked_docs)} chunks générés")
    return chunked_docs
