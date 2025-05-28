"""
Script pour tester le Jina Reranker 

Ce script permet de vérifier le bon fonctionnement de l'API Jina
dans un environnement isolé
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
logger = logging.getLogger('jina_test')

# Ajuster le chemin d'importation pour être compatible avec Streamlit Cloud
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Activer le reranker pour ce test
os.environ["USE_RERANKER"] = "True"
os.environ["STREAMLIT_MODE"] = "True"  # Utiliser le mode Streamlit pour jina_reranker

# Importations nécessaires
from langchain_core.documents import Document

def test_jina_reranker():
    """Teste le fonctionnement du Jina Reranker"""
    # Vérifier si la clé API est disponible
    jina_api_key = os.environ.get("JINA_API_KEY", "")
    if not jina_api_key:
        print("❌ Erreur: Clé API Jina non configurée")
        print("Définissez la variable d'environnement JINA_API_KEY")
        return
    
    # Importation conditionnelle pour éviter les erreurs si le module n'est pas disponible
    try:
        from src.jina_reranker import create_reranker, JinaReranker
    except ImportError:
        print("❌ Erreur: Module jina_reranker non trouvé")
        return
    
    print("\n===== TEST DU JINA RERANKER =====\n")
    
    # Créer quelques documents de test
    test_docs = [
        Document(page_content="OpCon est un scheduler de jobs qui permet d'automatiser les tâches informatiques.",
                metadata={"title": "OpCon Introduction"}),
        Document(page_content="Les jobs dans OpCon peuvent être configurés pour s'exécuter selon un calendrier.",
                metadata={"title": "Configuration des Jobs"}),
        Document(page_content="Lorsqu'un job est bloqué, vous devez vérifier les logs pour diagnostiquer le problème.",
                metadata={"title": "Résolution des Problèmes"}),
        Document(page_content="La surveillance des jobs est essentielle pour assurer le bon fonctionnement des systèmes.",
                metadata={"title": "Monitoring"})
    ]
    
    # Créer une requête de test
    query = "Comment résoudre un job bloqué?"
    
    print(f"Requête de test: '{query}'")
    print(f"Documents disponibles: {len(test_docs)}")
    
    # Initialiser le reranker
    reranker = create_reranker()
    if not reranker:
        print("❌ Échec de la création du reranker")
        return
    
    print(f"✅ Reranker de type {type(reranker).__name__} créé avec succès")
    
    # Effectuer le reranking
    try:
        print("Exécution du reranking...")
        reranked_docs = reranker.rerank(query, test_docs, top_k=3)
        
        print("\nRésultats du reranking:")
        for i, doc in enumerate(reranked_docs):
            print(f"{i+1}. {doc.metadata.get('title')}: {doc.page_content[:50]}...")
        
        print("\n✅ Test réussi!")
    except Exception as e:
        print(f"\n❌ Erreur lors du reranking: {e}")

if __name__ == "__main__":
    print("=== TEST DU RERANKER JINA ===")
    test_jina_reranker()
