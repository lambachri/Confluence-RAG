# Streamlit
# Use QARetrieval to find informations about the Octo Confluence
from venv import logger
import streamlit as st
from help_desk import HelpDesk
import os
import sys
import time
import importlib
from auth import login_form, logout_user
from logging import Logger
sys.path.append('../')
from src.config import PERSIST_DIRECTORY
from load_db import DataLoader

# Configuration de Torch pour éviter le problème avec asyncio et streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Utiliser la valeur de USE_RERANKER de l'environnement
os.environ["USE_RERANKER"] = os.environ.get("USE_RERANKER", "True")
# Indiquer qu'on est en mode Streamlit pour éviter d'importer torch
os.environ["STREAMLIT_MODE"] = "True"

# Liste des IDs des pages spécifiques à charger
SPECIFIC_PAGE_IDS = [
    "3376545896", "3710844929", "3985899883", "3729621164", "3712188435", 
    "3896606774", "4552491244", "4675077035", "3716349965", "3758030922", 
    "3758555439", "2384003194", "2640052240", "2722825033", "2639036983", 
    "2384691207", "2756640773","4444487681","4766369934"
]

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant OpCon",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS pour améliorer l'apparence et le contraste
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .stTextInput>div>div>input {
        padding: 0.8rem;
        font-size: 1.1rem;
    }
    .user-bubble {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 0.8rem;
        color: #0D47A1;  /* Texte bleu foncé pour meilleur contraste */
    }
    .assistant-bubble {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 0.8rem;
        color: #212121;  /* Texte presque noir pour meilleur contraste */
        border-left: 3px solid #1E88E5;
    }
    .sources-section {
        font-size: 0.85rem;
        color: #424242;  /* Gris foncé pour les sources */
        margin-top: 0.5rem;
        border-top: 1px solid #E0E0E0;
        padding-top: 0.5rem;
        background-color: #FAFAFA;
    }
    .typing-indicator {
        display: inline-block;
        font-size: 1.2rem;
        margin-left: 0.4rem;
        color: #1E88E5;
    }
    /* Amélioration de la visibilité des liens dans les sources */
    .sources-section a {
        color: #1565C0;
        text-decoration: underline;
    }
    /* Améliorer la visibilité de l'assistant pendant la saisie */
    .assistant-typing {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 0.8rem;
        color: #616161;
        border-left: 3px solid #90CAF9;
    }
    /* Style pour les éléments markdown dans les réponses */
    .assistant-bubble ul, .assistant-bubble ol {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
    }
    .assistant-bubble li {
        margin-bottom: 0.3rem;
    }
    .assistant-bubble p {
        margin-bottom: 0.8rem;
    }
    /* Espacement des listes à puces et numérotées */
    .assistant-bubble ul li, .assistant-bubble ol li {
        padding-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    # Check if DB exists before initializing
    db_exists = os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY)
    
    # Print debug information to sidebar only when expanded
    with st.sidebar.expander("Informations techniques"):
        st.write(f"Base de données existante: {db_exists}")
        if os.path.exists(PERSIST_DIRECTORY):
            st.write(f"Fichiers dans la base: {os.listdir(PERSIST_DIRECTORY)}")
    
    # Si la base n'existe pas, la créer avec les pages spécifiques
    if not db_exists:
        st.info("Première utilisation détectée. Création de la base de données avec les pages spécifiques...")
        with st.spinner("Chargement des données - veuillez patienter, cela peut prendre quelques minutes..."):
            try:
                # Initialiser le DataLoader avec les IDs
                from langchain_openai import OpenAIEmbeddings
                # Utilise automatiquement les pages par défaut (use_default_pages=True par défaut)
                loader = DataLoader()
                embeddings = OpenAIEmbeddings()
                db = loader.set_db(embeddings)
                st.success(f"Base de données créée avec succès!")
            except Exception as e:
                st.error(f"Erreur lors de la création de la base: {str(e)}")
    
    # Always use existing DB if available
    try:
        # Vérifier si le reranker est activé
        reranker_enabled = os.environ.get("USE_RERANKER", "True").lower() == "true"
        
        # Initialiser le modèle avec le jina_reranker
        # (HelpDesk utilisera automatiquement jina_reranker au lieu de reranker standard)
        from src.jina_reranker import create_reranker
        
        # Afficher l'état du reranker dans l'interface
        reranker_type = "Jina" if os.environ.get("JINA_API_KEY") else "Simple"
        st.sidebar.info(f"Reranker: {'✅ Activé (' + reranker_type + ')' if reranker_enabled else '❌ Désactivé'}")
        
        # Initialisation avec le paramètre use_reranker explicite
        model = HelpDesk(new_db=False, use_reranker=reranker_enabled)
        
        # Initialiser la mémoire conversationnelle
        model.init_conversation_memory()
        
        # Stocker le modèle dans la session pour pouvoir y accéder ailleurs
        if "model" not in st.session_state:
            st.session_state.model = model
        
        return model
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
        # Tenter d'initialiser avec une configuration minimale en cas d'échec
        try:
            # Configuration minimale sans fonctionnalités avancées
            model = HelpDesk(new_db=False)
            model.init_conversation_memory()
            
            if "model" not in st.session_state:
                st.session_state.model = model
            
            return model
        except Exception as e:
            st.error(f"Erreur fatale lors de l'initialisation du modèle: {str(e)}")
            return None

# Display app header
st.markdown("<h1 class='main-header'>Assistant Support OpCon</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Je suis votre expert technique pour le monitoring des schedulers. Comment puis-je vous aider aujourd'hui ?</p>", unsafe_allow_html=True)

# Sidebar with improved information and controls
with st.sidebar:
    st.title("Menu")
    
    # User guide
    with st.expander("Guide d'utilisation", expanded=False):
        st.write("""
        ### Comment utiliser l'assistant
        
        1. **Posez vos questions** sur OpCon, les schedulers, les jobs ou le monitoring
        2. **Soyez précis** dans vos demandes pour obtenir les meilleures réponses
        3. **Consultez les sources** en bas de chaque réponse pour plus d'informations
        
        L'assistant peut vous aider sur:
        - La configuration des jobs
        - Les erreurs courantes et leur résolution
        - Les procédures de monitoring
        - Les scripts et commandes techniques
        """)
    
    # Initialize conversation memory if needed
    if "model" in st.session_state and hasattr(st.session_state.model, "init_conversation_memory"):
        st.session_state.model.init_conversation_memory()
    
    # Add a button to clear conversation with improved styling
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour, je suis votre assistant technique OpCon. Comment puis-je vous aider aujourd'hui?"}]
        st.session_state.conversation_started = False
        st.rerun()
    
    # Add example questions for user to click
    st.subheader("Questions fréquentes")
    example_questions = [
        "Quelle est la procédure pour résoudre un job bloqué?",
        "Comment vérifier les logs d'un scheduler?",
        "Quelles sont les commandes de monitoring disponibles?",
        "Procédure de contrôle des articles dans M3 vers Cegid"  # Ajout de votre exemple problématique
    ]
    
    for question in example_questions:
        if st.button(f"📝 {question}", use_container_width=True, key=f"btn_{hash(question)}"):
            # Add question to session if not already in chat
            if "messages" in st.session_state:
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.current_question = question
                st.rerun()
    
    # Nouveau: Diagnostic technique
    with st.expander("Diagnostic technique", expanded=False):
        st.write("Vérifier le chargement des pages Confluence")
        if st.button("Exécuter le diagnostic des pages"):
            try:
                # Initialize the model for diagnostic
                model = get_model()
                from src.diagnostic import verify_pages
                
                with st.spinner("Vérification des pages Confluence..."):
                    placeholder = st.empty()
                    # Capture console output
                    import io
                    import sys
                    old_stdout = sys.stdout
                    new_stdout = io.StringIO()
                    sys.stdout = new_stdout
                    
                    # Run diagnostic
                    verify_pages()
                    
                    # Get output and restore stdout
                    output = new_stdout.getvalue()
                    sys.stdout = old_stdout
                    
                    # Display results
                    placeholder.code(output, language="text")
                    
                    # Check vectorstore
                    st.write("Vérification de la base vectorielle:")
                    try:
                        doc_count = model.db._collection.count()
                        st.success(f"✅ Base vectorielle OK: {doc_count} documents indexés")
                    except:
                        st.error("❌ Impossible de vérifier la base vectorielle")
            except Exception as e:
                st.error(f"Erreur lors du diagnostic: {str(e)}")
    
    # Section administration (après les autres sections)
    st.sidebar.markdown("---")
    admin_expander = st.sidebar.expander("🔒 Administration", expanded=False)
    
    with admin_expander:
        if "authenticated" in st.session_state and st.session_state.authenticated:
            st.success(f"Connecté en tant que {st.session_state.auth_username}")
            if st.button("Aller à l'interface d'administration"):
                # Importer dynamiquement le module admin
                admin_module = importlib.import_module("admin")
                st.session_state.show_admin = True
                st.rerun()
            
            if st.button("Se déconnecter"):
                logout_user()
                st.rerun()
        else:
            if st.button("Se connecter"):
                st.session_state.show_login = True
                st.rerun()

# Vérifier si l'utilisateur veut afficher l'admin ou le formulaire de connexion
if "show_admin" in st.session_state and st.session_state.show_admin and st.session_state.authenticated:
    # Afficher l'interface d'administration
    admin_module = importlib.import_module("admin")
    admin_module.admin_panel()
elif "show_login" in st.session_state and st.session_state.show_login:
    # Afficher le formulaire de connexion
    if login_form():
        st.success("Connexion réussie!")
        st.session_state.show_login = False
        st.session_state.show_admin = True
        st.rerun()
    
    # Bouton pour revenir à l'interface principale
    if st.button("Retour à l'interface principale"):
        st.session_state.show_login = False
        st.rerun()
else:
    # Initialize the model
    try:
        model = get_model()
        
        # Get collection count - show in expander in sidebar
        with st.sidebar.expander("Statistiques de la base", expanded=False):
            try:
                doc_count = model.db._collection.count()
                st.write(f"📊 Documents disponibles: {doc_count}")
                
                # Ajoute un bouton pour tester des requêtes directement
                if st.button("Tester la recherche de documents"):
                    with st.spinner("Recherche de documents..."):
                        test_queries = ["m3", "cegid", "contrôle des articles", "procédure"]
                        results = {}
                        
                        for query in test_queries:
                            docs = model.retriever.get_relevant_documents(query)
                            results[query] = [
                                f"{doc.metadata.get('title', 'Sans titre')} ({len(doc.page_content)} caractères)"
                                for doc in docs[:3]
                            ]
                        
                        for query, docs in results.items():
                            st.write(f"**Requête: '{query}'**")
                            if docs:
                                for i, doc in enumerate(docs):
                                    st.write(f"{i+1}. {doc}")
                            else:
                                st.write("Aucun document trouvé")
            except Exception as e:
                st.write(f"⚠️ Erreur de comptage: {str(e)}")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de l'initialisation du modèle: {str(e)}")
        model = None

    # Initialize chat history with a personalized welcome message
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour, je suis votre assistant technique OpCon. Comment puis-je vous aider aujourd'hui?"}]
        st.session_state.conversation_started = False
    
    # Track if this is beginning of conversation
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

    # Store the current question being processed to prevent duplicate processing
    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    # Fonction utilitaire pour formater les réponses de manière cohérente
    def format_assistant_response(content):
        """Assure un formatage markdown cohérent des réponses de l'assistant"""
        # Séparer le contenu des sources si présent
        if "Voici" in content and "sources" in content.lower():
            # Trouver la partie qui commence par "Voici"
            split_index = content.find("Voici")
            main_content = content[:split_index].strip()
            sources_text = content[split_index:].strip()
            
            # Assurer que le contenu principal utilise correctement le markdown
            main_content = main_content.replace("* ", "• ")  # Remplacer les astérisques par des puces visuelles
            main_content = main_content.replace("\n\n\n", "\n\n")  # Éviter les sauts de ligne excessifs
            
            # Extraire les liens des sources avec une expression régulière
            import re
            links = re.findall(r'\[(.*?)\]\((.*?)\)', sources_text)
            
            # Formater les sources avec un style sécurisé pour Streamlit
            formatted_sources = ""
            if links:
                source_title = "source" if len(links) == 1 else f"{len(links)} sources"
                formatted_sources = f'<div class="sources-section">Voici la {source_title} qui pourrait t\'être utile :</div>'
                
                # Créer une liste formatée sans HTML complexe, en utilisant des liens Markdown
                for title, url in links:
                    formatted_sources += f'<div class="source-item"><a href="{url}" target="_blank">📄 {title}</a></div>'
            else:
                # Si pas de liens trouvés, nettoyons le texte pour éviter les problèmes HTML
                formatted_sources = sources_text.replace("<", "&lt;").replace(">", "&gt;")
            
            # Encapsuler les réponses dans les balises HTML pour un style cohérent
            return f"<div class='assistant-bubble'><strong>Assistant:</strong><br>{main_content}</div>", f"<div class='sources-section'>{formatted_sources}</div>"
        else:
            # Formater le contenu sans sources
            content = content.replace("* ", "• ")  # Remplacer les astérisques par des puces visuelles
            content = content.replace("\n\n\n", "\n\n")  # Éviter les sauts de ligne excessifs
            return f"<div class='assistant-bubble'><strong>Assistant:</strong><br>{content}</div>", None

    # Display chat history with improved styling and contrast
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'><strong>Vous:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            # Utiliser la fonction de formatage pour les réponses de l'assistant
            content = msg["content"]
            main_content, sources = format_assistant_response(content)
            
            st.markdown(main_content, unsafe_allow_html=True)
            if sources:
                st.markdown(sources, unsafe_allow_html=True)

    # Process the current question if it exists and hasn't been processed
    if "current_question" in st.session_state and st.session_state.current_question:
        question = st.session_state.current_question
        
        # Show typing indicator with better contrast
        typing_placeholder = st.empty()
        typing_placeholder.markdown("<div class='assistant-typing'><strong>Assistant:</strong><br>En train de réfléchir<span class='typing-indicator'>...</span></div>", unsafe_allow_html=True)
        
        try:
            # Get answer with a slight delay to show typing effect
            time.sleep(0.5)  # Short delay to show typing
            
            # Get personalized greeting for first interaction
            if not st.session_state.conversation_started:
                greeting = model.get_personalized_greeting()
                # Add enhanced greeting to chat history
                st.session_state.messages.append({"role": "assistant", "content": greeting})
                st.session_state.conversation_started = True
                typing_placeholder.empty()
                st.rerun()
            
            # Get answer from model with context awareness enabled
            result, sources = model.retrieval_qa_inference(question, use_context=True)
            
            # Always apply context-based enhancement after the first exchange
            if len(st.session_state.messages) > 2:  # If we have previous exchanges
                result = model.enhance_response_with_context(result, st.session_state.messages)
            
            # Standardiser le formatage markdown de la réponse
            result = result.replace("- ", "* ")  # Uniformiser les puces
            result = result.replace("\n\n\n", "\n\n")  # Réduire les espaces excessifs
            
            # Add answer and sources
            typing_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": result + '  \n  \n' + sources})
            st.session_state.current_question = None  # Reset current question
            st.rerun()
            
        except Exception as e:
            error_msg = f"Désolé, j'ai rencontré un problème technique. Pourriez-vous reformuler votre question? (Erreur: {str(e)})"
            typing_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.current_question = None
            st.rerun()

    # Chat input with improved placeholder
    user_input = st.chat_input("Posez votre question technique sur OpCon ici...")

    if user_input:
        if model is None:
            st.error("L'assistant n'est pas disponible actuellement. Veuillez réessayer plus tard.")
        else:
            # Add user message to session
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.current_question = user_input
            st.rerun()
