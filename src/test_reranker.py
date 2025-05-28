"""
Script pour tester le reranker sans Streamlit

Ce script permet de tester le reranker en mode CLI, sans les problèmes
d'interférence avec Streamlit
"""

import os
import sys
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reranker_test')

# Ajuster le chemin d'importation pour être compatible avec Streamlit Cloud
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Activation du reranker pour ce test
os.environ["USE_RERANKER"] = "True"
os.environ["STREAMLIT_MODE"] = "False"

# Importer HelpDesk
from help_desk import HelpDesk

def test_reranker():
    """Test du reranker avec des requêtes types"""
    print("\n===== TEST DU RERANKER =====\n")
    
    # Initialiser le modèle avec reranker activé
    print("Initialisation du modèle...")
    model = HelpDesk(new_db=False, use_reranker=True)
    
    # Vérifier si le reranker est bien activé
    if model.use_reranker and model.reranker:
        print("✅ Reranker activé avec succès")
    else:
        print("❌ Échec de l'activation du reranker")
        return
    
    # Liste de requêtes à tester
    test_queries = [
        "Comment configurer un job dans OpCon?",
        "Quelle est la procédure pour résoudre un job bloqué?",
        "Comment vérifier les logs d'un scheduler?",
        "Procédure de contrôle des articles dans M3 vers Cegid"
    ]
    
    # Tester chaque requête
    for query in test_queries:
        print(f"\n🔍 Requête: '{query}'")
        
        # Récupérer les documents avec le reranker
        docs = model.retriever.get_relevant_documents(query)
        print(f"Documents récupérés: {len(docs)}")
        
        # Utiliser le reranker explicitement
        if model.reranker:
            print("Application du reranking...")
            reranked_docs = model.reranker.rerank(query, docs, top_k=5)
            
            print("\n📄 Top 5 documents après reranking:")
            for i, doc in enumerate(reranked_docs[:5]):
                title = doc.metadata.get('title', 'Sans titre')
                print(f"{i+1}. {title} (id: {doc.metadata.get('id', 'N/A')})")
        
        # Obtenir une réponse complète
        print("\nGénération de la réponse...")
        answer, sources = model.retrieval_qa_inference(query, use_context=False)
        
        print("\n💬 Réponse générée:")
        print(answer[:150] + "..." if len(answer) > 150 else answer)
        print("\n📚 Sources:")
        print(sources)
        print("\n" + "="*50)

if __name__ == "__main__":
    test_reranker()

"""
Test pour vérifier que les importations fonctionnent correctement
"""
import os
import sys
from pathlib import Path

# Configurer les chemins d'importation
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
    print(f"Ajout au path: {root_dir}")

# Tester les importations
def test_imports():
    try:
        print("Tentative d'importation de load_db...")
        import load_db
        print("✅ Importation de load_db réussie")
    except ImportError as e:
        print(f"❌ Erreur lors de l'importation de load_db: {e}")

    try:
        print("Tentative d'importation de config...")
        import config
        print(f"✅ Importation de config réussie: {config.PERSIST_DIRECTORY}")
    except ImportError as e:
        print(f"❌ Erreur lors de l'importation de config: {e}")

    try:
        print("Tentative d'importation de help_desk...")
        import help_desk
        print("✅ Importation de help_desk réussie")
    except ImportError as e:
        print(f"❌ Erreur lors de l'importation de help_desk: {e}")

if __name__ == "__main__":
    # Afficher l'environnement
    print(f"Python version: {sys.version}")
    print(f"Répertoire de travail: {os.getcwd()}")
    print(f"PYTHONPATH: {sys.path}")
    
    test_imports()
