"""
Script pour tester le reranker Jina

Ce script permet de tester le reranker Jina avec différentes requêtes
pour vérifier son fonctionnement et sa pertinence.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('jina_test')

# Activation du reranker pour ce test
os.environ["USE_RERANKER"] = "True"

# Importer HelpDesk
from help_desk import HelpDesk

def test_jina_reranker():
    """Test du reranker Jina avec des requêtes types"""
    print("\n===== TEST DU RERANKER JINA =====\n")
    
    # Vérifier que la clé API Jina est configurée
    jina_api_key = os.environ.get("JINA_API_KEY")
    if not jina_api_key:
        print("❌ ERREUR: Clé API Jina non configurée dans le fichier .env")
        print("Veuillez ajouter votre clé API Jina dans le fichier .env:")
        print('JINA_API_KEY = "votre_clé_api"')
        return
    
    # Initialiser le modèle avec reranker activé
    print("Initialisation du modèle...")
    model = HelpDesk(new_db=False, use_reranker=True)
    
    # Vérifier si le reranker est bien activé
    if model.use_reranker and model.reranker:
        print(f"✅ Reranker activé avec succès: {type(model.reranker).__name__}")
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
        
        # Récupérer les documents avec le retriever standard
        docs = model.retriever.get_relevant_documents(query)
        print(f"Documents récupérés: {len(docs)}")
        
        # Extraction des titres avant reranking
        print("\n📄 Top 5 documents avant reranking:")
        for i, doc in enumerate(docs[:5]):
            title = doc.metadata.get('title', 'Sans titre')
            print(f"{i+1}. {title} (id: {doc.metadata.get('id', 'N/A')})")
        
        # Utiliser le reranker explicitement
        if model.reranker:
            print("\nApplication du reranking avec Jina...")
            reranked_docs = model.reranker.rerank(query, docs, top_k=5)
            
            print("\n📄 Top 5 documents après reranking:")
            for i, doc in enumerate(reranked_docs[:5]):
                title = doc.metadata.get('title', 'Sans titre')
                print(f"{i+1}. {title} (id: {doc.metadata.get('id', 'N/A')})")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_jina_reranker()
