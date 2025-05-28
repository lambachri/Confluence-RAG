"""
Script pour tester le reranker Jina ou Simple indépendamment

Ce script permet de tester le reranker sans Streamlit
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_reranker')

# Activation du reranker pour ce test
os.environ["USE_RERANKER"] = "True"

# Test du reranker Jina seul
def test_jina_reranker():
    try:
        from src.jina_reranker import create_reranker, Document
        
        print("\n===== TEST DU RERANKER JINA OU SIMPLE =====\n")
        
        # Vérifier la clé API Jina
        jina_api_key = os.environ.get("JINA_API_KEY")
        if not jina_api_key:
            print("⚠️ Clé API Jina non configurée, utilisation du SimpleReranker")
        else:
            print("✅ Clé API Jina configurée")
        
        # Créer le reranker
        reranker = create_reranker()
        if reranker:
            print(f"✅ Reranker créé: {type(reranker).__name__}")
        else:
            print("❌ Échec de la création du reranker")
            return
        
        # Créer quelques documents de test
        test_docs = [
            Document(page_content="OpCon est un scheduler de jobs qui permet d'automatiser les tâches informatiques.",
                     metadata={"title": "OpCon Introduction", "relevance_boost": 1.2}),
            Document(page_content="Les jobs dans OpCon peuvent être configurés pour s'exécuter selon un calendrier.",
                     metadata={"title": "Configuration des Jobs", "relevance_boost": 1.0}),
            Document(page_content="Lorsqu'un job est bloqué, vous devez vérifier les logs pour diagnostiquer le problème.",
                     metadata={"title": "Résolution des Problèmes", "relevance_boost": 1.1}),
            Document(page_content="La surveillance des jobs est essentielle pour assurer le bon fonctionnement des systèmes.",
                     metadata={"title": "Monitoring", "relevance_boost": 1.0})
        ]
        
        # Liste de requêtes à tester
        test_queries = [
            "Comment résoudre un job bloqué dans OpCon?",
            "Comment configurer un job dans OpCon?",
            "Comment surveiller les jobs?",
            "Quel est le rôle d'OpCon?"
        ]
        
        # Tester chaque requête
        for query in test_queries:
            print(f"\n🔍 Requête: '{query}'")
            
            # Appliquer le reranking
            reranked_docs = reranker.rerank(query, test_docs)
            
            print("\n📄 Documents réordonnés:")
            for i, doc in enumerate(reranked_docs):
                print(f"{i+1}. {doc.metadata.get('title')}: {doc.page_content[:50]}...")
            
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")

# Test avec HelpDesk pour voir l'intégration complète
def test_with_helpdesk():
    try:
        from help_desk import HelpDesk
        
        print("\n===== TEST AVEC HELPDESK =====\n")
        
        # Activer le reranker
        os.environ["USE_RERANKER"] = "True"
        
        # Initialiser HelpDesk
        print("Initialisation de HelpDesk avec reranker...")
        model = HelpDesk(new_db=False, use_reranker=True)
        
        # Vérifier si le reranker est activé
        if model.reranker:
            print(f"✅ Reranker activé: {type(model.reranker).__name__}")
        else:
            print("❌ Reranker non activé")
            return
        
        # Test d'une requête
        question = "Comment résoudre un job bloqué dans OpCon?"
        print(f"\n🔍 Requête: '{question}'")
        
        # Obtenir les documents pertinents
        docs = model.retriever.get_relevant_documents(question)
        print(f"Documents récupérés: {len(docs)}")
        
        # Appliquer le reranking
        if model.reranker:
            reranked_docs = model.reranker.rerank(question, docs, top_k=5)
            
            print("\n📄 Top 5 documents après reranking:")
            for i, doc in enumerate(reranked_docs[:5]):
                title = doc.metadata.get('title', 'Sans titre')
                print(f"{i+1}. {title} (id: {doc.metadata.get('id', 'N/A')})")
        
    except Exception as e:
        print(f"❌ Erreur lors du test avec HelpDesk: {e}")

if __name__ == "__main__":
    test_jina_reranker()
    
    # Décommenter pour tester avec HelpDesk complet
    # test_with_helpdesk()
