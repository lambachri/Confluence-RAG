import streamlit as st
import os
import sys
import json
import time
import shutil
import tempfile
import pandas as pd
from pathlib import Path
import base64
from PIL import Image
import io

from auth import login_form, change_password_form, logout_user
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader, DirectoryLoader

sys.path.append('../')
from src.config import PERSIST_DIRECTORY
from load_db import DataLoader

# Importer les utilitaires pour les PDF et images
from pdf_utils import load_pdf, load_image, split_documents, load_directory

# Liste des types de fichiers supportés pour l'upload
SUPPORTED_TYPES = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "csv": "text/csv"  # Ajout du support CSV
}

def save_uploaded_file(uploaded_file, upload_dir):
    """Sauvegarde un fichier uploadé dans le répertoire spécifié"""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def process_pdf(file_path):
    """Traite un fichier PDF pour l'indexation"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Ajouter des métadonnées
    for doc in documents:
        doc.metadata["source"] = file_path
        doc.metadata["title"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "pdf"
    
    return documents

def process_image(file_path):
    """Traite un fichier image pour l'indexation"""
    loader = UnstructuredImageLoader(file_path, mode="elements")
    documents = loader.load()
    
    # Ajouter des métadonnées
    for doc in documents:
        doc.metadata["source"] = file_path
        doc.metadata["title"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "image"
    
    return documents

# Ajouter une fonction pour traiter les CSV
def process_csv(file_path):
    """Traite un fichier CSV pour l'indexation"""
    import pandas as pd
    from langchain_core.documents import Document
    
    try:
        # Charger le CSV
        df = pd.read_csv(file_path, encoding='utf-8')
        documents = []
        
        # Option 1: Transformer chaque ligne en document
        for idx, row in df.iterrows():
            # Convertir la ligne en texte formaté
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            
            # Créer un document avec métadonnées
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "title": f"{os.path.basename(file_path)} - Row {idx+1}",
                    "file_type": "csv",
                    "row_index": idx,
                    # Ajouter optionnellement les colonnes comme métadonnées
                    **{f"col_{col}": str(val) for col, val in row.items() if pd.notna(val)}
                }
            )
            documents.append(doc)
        
        return documents
    except Exception as e:
        print(f"Erreur lors du traitement du CSV: {e}")
        return []

def chunk_documents(documents):
    """Découpe les documents en chunks pour l'indexation"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def add_documents_to_vectorstore(documents, persist_directory=PERSIST_DIRECTORY):
    """Ajoute des documents à la base vectorielle existante"""
    # Initialiser l'embedding
    embeddings = OpenAIEmbeddings()
    
    # Charger la base existante ou en créer une nouvelle
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        # Ajouter les nouveaux documents
        db.add_documents(documents)
    else:
        # Créer une nouvelle base
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    return db

def add_confluence_pages(page_ids, force_reload=False):
    """Ajoute des pages Confluence à la base de connaissances"""
    if not page_ids:
        return "Aucun ID fourni."
    
    try:
        # Initialiser le DataLoader avec les IDs spécifiques
        loader = DataLoader(page_ids=page_ids, use_default_pages=False)
        embeddings = OpenAIEmbeddings()
        
        # Si force_reload est True, supprimer la base existante
        if force_reload and os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            st.warning("Base de données supprimée pour rechargement complet.")
        
        # Charger et indexer les pages
        db = loader.set_db(embeddings)
        
        # Vérifier le nombre de documents chargés
        try:
            doc_count = db._collection.count()
            return f"Chargement réussi. {doc_count} documents sont maintenant dans la base."
        except Exception as e:
            return f"Chargement terminé mais impossible de compter les documents: {e}"
    
    except Exception as e:
        return f"Erreur lors du chargement des pages: {e}"

def get_current_pages():
    """Récupère la liste des pages Confluence actuellement dans la base"""
    # Vérifier si le cache existe
    cache_file = "confluence_page_cache.json"
    if not os.path.exists(cache_file):
        return []
    
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        
        pages = []
        for page_id, page_data in cache.items():
            title = "Titre inconnu"
            if "metadata" in page_data and page_data["metadata"]:
                if isinstance(page_data["metadata"], list) and page_data["metadata"]:
                    title = page_data["metadata"][0].get("title", f"Page {page_id}")
                else:
                    title = page_data["metadata"].get("title", f"Page {page_id}")
            
            pages.append({
                "id": page_id,
                "title": title,
                "version": page_data.get("version", "Inconnue")
            })
        
        return pages
    
    except Exception as e:
        st.error(f"Erreur lors de la lecture du cache Confluence: {e}")
        return []

def get_uploaded_documents():
    """Récupère la liste des documents uploadés"""
    # Répertoire des documents uploadés
    upload_dir = "uploaded_files"
    if not os.path.exists(upload_dir):
        return []
    
    files = []
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            file_stats = os.stat(file_path)
            files.append({
                "filename": filename,
                "path": file_path,
                "size": file_stats.st_size,
                "date_added": time.ctime(file_stats.st_mtime)
            })
    
    return files

def admin_panel():
    """Interface d'administration principale"""
    st.title("Administration de la Base de Connaissances")
    
    # Authentification
    if not login_form():
        return
    
    # Menu de navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à",
        ["Ajout de Pages Confluence", "Upload de Documents", "Gestion des Documents", "Paramètres"]
    )
    
    # Bouton de déconnexion
    if st.sidebar.button("Déconnexion"):
        logout_user()
        st.rerun()
    
    # Pages d'administration
    if page == "Ajout de Pages Confluence":
        confluence_page_manager()
    elif page == "Upload de Documents":
        document_uploader()
    elif page == "Gestion des Documents":
        document_manager()
    elif page == "Paramètres":
        settings_page()

def confluence_page_manager():
    """Page d'ajout et de gestion des pages Confluence"""
    st.header("Gestion des Pages Confluence")
    
    # Afficher les pages existantes
    current_pages = get_current_pages()
    if current_pages:
        st.subheader("Pages actuellement indexées")
        df = pd.DataFrame(current_pages)
        st.dataframe(df)
    else:
        st.info("Aucune page Confluence n'est actuellement indexée.")
    
    # Formulaire d'ajout de nouvelles pages
    st.subheader("Ajouter de nouvelles pages")
    with st.form("add_confluence_pages"):
        page_ids_input = st.text_area(
            "IDs des pages Confluence (un par ligne ou séparés par des virgules)",
            help="Entrez les IDs numériques des pages Confluence à ajouter à la base de connaissances"
        )
        
        force_reload = st.checkbox(
            "Recharger complètement la base",
            help="Attention: cela supprimera la base existante et recréera tout à partir de zéro"
        )
        
        submitted = st.form_submit_button("Ajouter ces pages")
        
        if submitted:
            # Traiter les IDs de page
            if "," in page_ids_input:
                page_ids = [pid.strip() for pid in page_ids_input.split(",")]
            else:
                page_ids = [pid.strip() for pid in page_ids_input.splitlines()]
            
            # Filtrer les IDs vides
            page_ids = [pid for pid in page_ids if pid]
            
            if not page_ids:
                st.error("Veuillez entrer au moins un ID de page valide.")
            else:
                with st.spinner(f"Traitement de {len(page_ids)} pages Confluence..."):
                    result = add_confluence_pages(page_ids, force_reload)
                    st.success(result)
                    # Rafraîchir la page pour afficher les nouvelles données
                    st.rerun()

def document_uploader():
    """Page d'upload de documents (PDF, images)"""
    st.header("Upload de Documents")
    
    # Répertoire pour les fichiers uploadés
    upload_dir = "uploaded_files"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # Onglets pour séparer les différentes méthodes d'upload
    tab1, tab2 = st.tabs(["Upload de fichiers", "Indexation d'un dossier"])
    
    with tab1:
        # Formulaire d'upload
        uploaded_files = st.file_uploader(
            "Sélectionnez des fichiers à ajouter à la base de connaissances",
            accept_multiple_files=True,
            type=list(SUPPORTED_TYPES.keys())
        )
        
        if uploaded_files:
            st.subheader("Fichiers à traiter")
            file_info = []
            
            for file in uploaded_files:
                file_type = "PDF" if file.name.lower().endswith(".pdf") else "Image"
                file_info.append({
                    "nom": file.name,
                    "taille": f"{file.size / 1024:.1f} KB",
                    "type": file_type
                })
            
            # Afficher un tableau avec les informations des fichiers
            df = pd.DataFrame(file_info)
            st.table(df)
            
            # Options de prétraitement
            st.subheader("Options de traitement")
            
            chunk_size = st.slider(
                "Taille des chunks (caractères)",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="Nombre de caractères par segment de texte"
            )
            
            chunk_overlap = st.slider(
                "Chevauchement des chunks",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Nombre de caractères de chevauchement entre segments consécutifs"
            )
            
            extract_images = st.checkbox(
                "Extraire le texte des images dans les PDFs",
                value=True,
                help="Utilise l'OCR pour extraire le texte des images contenues dans les PDFs"
            )
            
            if st.button("Traiter ces fichiers"):
                with st.spinner("Traitement des fichiers..."):
                    # Sauvegarder les fichiers uploadés
                    saved_files = []
                    for file in uploaded_files:
                        saved_path = save_uploaded_file(file, upload_dir)
                        saved_files.append(saved_path)
                    
                    # Traiter chaque fichier selon son type
                    all_documents = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file_path in enumerate(saved_files):
                        progress = i / len(saved_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Traitement de {os.path.basename(file_path)}...")
                        
                        try:
                            if file_path.lower().endswith('.pdf'):
                                # Utiliser les fonctions de pdf_utils
                                docs = load_pdf(file_path)
                                status_text.text(f"PDF traité: {len(docs)} pages extraites")
                            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                # Utiliser les fonctions de pdf_utils
                                docs = load_image(file_path)
                                status_text.text(f"Image traitée: {len(docs)} éléments extraits")
                            elif file_path.lower().endswith('.csv'):
                                # Traiter les fichiers CSV
                                docs = process_csv(file_path)
                                status_text.text(f"CSV traité: {len(docs)} lignes extraites")
                            else:
                                st.warning(f"Type de fichier non supporté: {file_path}")
                                continue
                            
                            # Prévisualisation du contenu extrait
                            if docs:
                                content_preview = docs[0].page_content[:200] + "..." if len(docs[0].page_content) > 200 else docs[0].page_content
                                st.info(f"Extrait de {os.path.basename(file_path)}:\n\n{content_preview}")
                            
                            all_documents.extend(docs)
                        except Exception as e:
                            st.error(f"Erreur lors du traitement de {file_path}: {str(e)}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("Extraction des documents terminée")
                    
                    if all_documents:
                        # Découper les documents en chunks avec les paramètres personnalisés
                        status_text.text(f"Découpage de {len(all_documents)} documents...")
                        
                        # Utiliser le RecursiveCharacterTextSplitter avec les paramètres spécifiés
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=["\n\n", "\n", ".", ";", ",", " ", ""]
                        )
                        
                        chunked_docs = splitter.split_documents(all_documents)
                        
                        status_text.text(f"Indexation de {len(chunked_docs)} chunks...")
                        
                        # Ajouter à la base vectorielle
                        try:
                            db = add_documents_to_vectorstore(chunked_docs)
                            doc_count = db._collection.count()
                            st.success(f"Indexation réussie! La base contient maintenant {doc_count} documents.")
                            
                            # Afficher un récapitulatif des documents ajoutés
                            st.subheader("Documents ajoutés")
                            file_summary = []
                            for file_path in saved_files:
                                file_summary.append({
                                    "Nom": os.path.basename(file_path),
                                    "Type": "PDF" if file_path.lower().endswith('.pdf') else "Image",
                                    "Chemin": file_path
                                })
                            
                            st.table(pd.DataFrame(file_summary))
                        except Exception as e:
                            st.error(f"Erreur lors de l'indexation: {str(e)}")
                    else:
                        st.warning("Aucun document n'a été traité avec succès.")
    
    with tab2:
        st.subheader("Indexation d'un dossier")
        
        folder_path = st.text_input(
            "Chemin du dossier à indexer",
            help="Entrez le chemin complet vers le dossier contenant les fichiers à indexer"
        )
        
        file_types = st.multiselect(
            "Types de fichiers à indexer",
            options=["PDF", "Images (JPG, PNG)"],
            default=["PDF", "Images (JPG, PNG)"]
        )
        
        recursive = st.checkbox("Indexer les sous-dossiers", value=True)
        
        if st.button("Analyser le dossier"):
            if not folder_path or not os.path.exists(folder_path):
                st.error("Le dossier spécifié n'existe pas")
            else:
                try:
                    # Préparer les extensions en fonction des sélections
                    extensions = []
                    if "PDF" in file_types:
                        extensions.append(".pdf")
                    if "Images (JPG, PNG)" in file_types:
                        extensions.extend([".jpg", ".jpeg", ".png"])
                    
                    # Compter les fichiers pour afficher un aperçu
                    files_count = {ext: 0 for ext in extensions}
                    total_files = 0
                    
                    if recursive:
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                ext = os.path.splitext(file)[1].lower()
                                if ext in extensions:
                                    files_count[ext] += 1
                                    total_files += 1
                    else:
                        for file in os.listdir(folder_path):
                            if os.path.isfile(os.path.join(folder_path, file)):
                                ext = os.path.splitext(file)[1].lower()
                                if ext in extensions:
                                    files_count[ext] += 1
                                    total_files += 1
                    
                    # Afficher le résultat de l'analyse
                    st.info(f"Trouvé {total_files} fichiers à indexer:")
                    for ext, count in files_count.items():
                        if count > 0:
                            st.write(f"- {count} fichiers {ext}")
                    
                    # Option pour procéder à l'indexation
                    if total_files > 0 and st.button("Indexer ces fichiers"):
                        with st.spinner(f"Indexation de {total_files} fichiers..."):
                            # Utiliser la fonction load_directory de pdf_utils
                            from pdf_utils import load_directory
                            
                            if recursive:
                                # Implémenter une version récursive
                                all_documents = []
                                for root, _, files in os.walk(folder_path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        ext = os.path.splitext(file)[1].lower()
                                        if ext in extensions:
                                            try:
                                                if ext == ".pdf":
                                                    docs = load_pdf(file_path)
                                                else:
                                                    docs = load_image(file_path)
                                                all_documents.extend(docs)
                                            except Exception as e:
                                                st.warning(f"Erreur lors du traitement de {file}: {e}")
                            else:
                                # Version non récursive (utilisant load_directory)
                                all_documents = load_directory(folder_path, extensions)
                            
                            # Découper en chunks et indexer
                            if all_documents:
                                chunked_docs = split_documents(all_documents)
                                db = add_documents_to_vectorstore(chunked_docs)
                                try:
                                    doc_count = db._collection.count()
                                    st.success(f"Indexation réussie! {len(chunked_docs)} chunks ajoutés à la base.")
                                    st.info(f"La base contient maintenant {doc_count} documents.")
                                except Exception as e:
                                    st.error(f"Erreur lors du comptage des documents: {e}")
                            else:
                                st.warning("Aucun document n'a été extrait avec succès.")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse du dossier: {e}")

def document_manager():
    """Page de gestion des documents existants"""
    st.header("Gestion des Documents")
    
    # Vérifier si la base vectorielle existe
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        st.warning("Aucune base vectorielle n'existe actuellement.")
        return
    
    # Onglets pour les différentes sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Documents Confluence", 
        "Fichiers uploadés",
        "Visualiseur de PDF et Images", 
        "Base vectorielle"
    ])
    
    with tab1:
        st.subheader("Documents Confluence")
        
        current_pages = get_current_pages()
        if current_pages:
            df = pd.DataFrame(current_pages)
            st.dataframe(df)
            
            if st.button("Rafraîchir toutes les pages"):
                with st.spinner("Rafraîchissement des pages..."):
                    page_ids = [page["id"] for page in current_pages]
                    result = add_confluence_pages(page_ids, force_reload=False)
                    st.success(result)
        else:
            st.info("Aucune page Confluence n'est actuellement indexée.")
    
    with tab2:
        st.subheader("Fichiers uploadés")
        
        uploaded_docs = get_uploaded_documents()
        if uploaded_docs:
            df = pd.DataFrame(uploaded_docs)
            st.dataframe(df[["filename", "size", "date_added"]])
            
            # Option pour supprimer des fichiers
            selected_file = st.selectbox(
                "Sélectionnez un fichier à supprimer",
                options=[doc["filename"] for doc in uploaded_docs]
            )
            
            if st.button("Supprimer ce fichier"):
                file_to_delete = next((doc["path"] for doc in uploaded_docs if doc["filename"] == selected_file), None)
                if file_to_delete and os.path.exists(file_to_delete):
                    os.remove(file_to_delete)
                    st.success(f"Fichier {selected_file} supprimé.")
                    st.info("Note: le fichier a été supprimé, mais son contenu est toujours dans la base vectorielle.")
                    st.rerun()
        else:
            st.info("Aucun fichier n'a été uploadé.")
    
    with tab3:
        st.subheader("Visualiseur de PDF et Images")
        # Récupérer la liste des fichiers uploadés
        upload_dir = "uploaded_files"
        if not os.path.exists(upload_dir):
            st.info("Aucun fichier n'a été uploadé.")
        else:
            # Filtrer par type
            file_type = st.radio("Type de fichier à afficher", ["Tous", "PDF", "Images"])
            
            # Obtenir la liste des fichiers
            files = []
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    if file_type == "Tous" or \
                       (file_type == "PDF" and filename.lower().endswith(".pdf")) or \
                       (file_type == "Images" and any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"])):
                        files.append(file_path)
            
            if not files:
                st.info(f"Aucun fichier de type {file_type.lower()} trouvé.")
            else:
                # Sélecteur de fichier
                selected_file = st.selectbox(
                    "Sélectionnez un fichier à visualiser",
                    options=files,
                    format_func=os.path.basename
                )
                
                if selected_file:
                    # Afficher le fichier selon son type
                    file_extension = os.path.splitext(selected_file)[1].lower()
                    
                    if file_extension == ".pdf":
                        display_pdf(selected_file)
                    elif file_extension in [".jpg", ".jpeg", ".png"]:
                        display_image(selected_file)
    
    with tab4:
        st.subheader("Base vectorielle")
        
        # Statistiques de la base
        try:
            embeddings = OpenAIEmbeddings()
            db = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
            
            doc_count = db._collection.count()
            st.info(f"La base vectorielle contient actuellement {doc_count} documents.")
            
            # Option pour effacer complètement la base
            if st.button("⚠️ Réinitialiser la base vectorielle", help="Attention: cette action est irréversible!"):
                confirm = st.text_input("Tapez 'CONFIRMER' pour effacer définitivement la base vectorielle")
                if confirm == "CONFIRMER":
                    shutil.rmtree(PERSIST_DIRECTORY)
                    st.success("Base vectorielle supprimée avec succès.")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Erreur lors de l'accès à la base vectorielle: {e}")

def display_pdf(file_path):
    """Affiche un PDF dans l'interface Streamlit"""
    try:
        # Lire le fichier PDF
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # Intégrer le PDF dans un iframe
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Extraire et afficher un aperçu du texte
        try:
            from pdf_utils import load_pdf
            docs = load_pdf(file_path)
            if docs:
                with st.expander("Aperçu du texte extrait"):
                    for i, doc in enumerate(docs[:3]):  # Limiter à 3 pages pour l'aperçu
                        st.markdown(f"**Page {i+1}:**")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.markdown("---")
                    
                    if len(docs) > 3:
                        st.info(f"... et {len(docs) - 3} autres pages")
        except Exception as e:
            st.warning(f"Impossible d'extraire le texte du PDF: {e}")
    
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du PDF: {e}")

def display_image(file_path):
    """Affiche une image dans l'interface Streamlit"""
    try:
        image = Image.open(file_path)
        st.image(image, caption=os.path.basename(file_path), use_column_width=True)
        
        # Extraire et afficher le texte (OCR)
        try:
            from pdf_utils import load_image
            docs = load_image(file_path)
            if docs:
                with st.expander("Texte extrait par OCR"):
                    for i, doc in enumerate(docs):
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        except Exception as e:
            st.warning(f"OCR non disponible pour cette image: {e}")
            
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'image: {e}")

def settings_page():
    """Page de paramètres pour l'administration"""
    st.header("Paramètres")
    
    # Sécurité et authentification
    st.subheader("Changement de mot de passe")
    change_password_form()
    
    # Paramètres de l'application
    st.subheader("Configuration de l'application")
    
    # Paramètres du modèle
    st.subheader("Paramètres du modèle")
    with st.form("model_settings"):
        model_name = st.selectbox(
            "Modèle LLM",
            options=["gpt-3.5-turbo-16k", "gpt-4"],
            index=0
        )
        
        temperature = st.slider(
            "Température",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
        
        submitted = st.form_submit_button("Enregistrer les paramètres")
        
        if submitted:
            # Sauvegarder les paramètres dans un fichier de configuration
            config = {
                "model": {
                    "name": model_name,
                    "temperature": temperature
                }
            }
            
            try:
                with open("model_config.json", "w") as f:
                    json.dump(config, f, indent=4)
                st.success("Paramètres enregistrés avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement des paramètres: {e}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Administration RAG-Chatbot",
        page_icon="🔒",
        layout="wide"
    )
    admin_panel()
