"""
Module de reranking basé uniquement sur Jina AI API, sans dépendances torch.
Ce module fournit des fonctionnalités de reranking sans utiliser sentence-transformers.
"""

import logging
import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Récupérer la clé API Jina
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")

class JinaReranker:
    """Utilise l'API Jina pour le reranking dans le cloud, sans dépendance à torch"""
    
    def __init__(self):
        self.api_key = JINA_API_KEY
        self.api_url = "https://api.jina.ai/v1/rerank"
        
        if not self.api_key:
            logger.warning("Clé API Jina non configurée, le reranker ne fonctionnera pas correctement")
        else:
            logger.info("JinaReranker initialisé avec succès")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Réordonne les documents en utilisant l'API Jina
        
        Args:
            query: La requête utilisateur
            documents: Liste des documents à réordonner
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste des documents réordonnés
        """
        if not self.api_key or not documents:
            logger.warning("Impossible d'utiliser JinaReranker: clé API manquante ou aucun document")
            return documents[:top_k] if documents else []
        
        try:
            # Préparer les données pour l'API Jina
            texts = [doc.page_content for doc in documents]
            
            # Limiter le nombre de documents si nécessaire
            max_docs = 32  # Limitation de l'API Jina
            if len(texts) > max_docs:
                logger.warning(f"Trop de documents ({len(texts)}) pour l'API Jina, limitation à {max_docs}")
                texts = texts[:max_docs]
                documents = documents[:max_docs]
            
            # Créer la requête pour l'API
            payload = {
                "model": "jina-reranker-v1",
                "query": query,
                "documents": texts,
                "top_k": top_k
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Envoyer la requête à l'API
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()  # Lever une exception si la requête a échoué
            
            result = response.json()
            
            # Traiter le résultat
            if "results" in result:
                # Créer un dictionnaire pour les indices et scores
                reranked_indices = []
                for item in result["results"]:
                    reranked_indices.append(item["index"])
                
                # Réordonner les documents selon les indices retournés par Jina
                reranked_docs = []
                for idx in reranked_indices[:top_k]:
                    if idx < len(documents):  # Vérification de sécurité
                        reranked_docs.append(documents[idx])
                
                logger.info(f"Reranking avec Jina réussi, {len(reranked_docs)} documents retournés")
                return reranked_docs
            else:
                logger.warning("Format de réponse inattendu de l'API Jina")
                return documents[:top_k]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la requête à l'API Jina: {e}")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"Erreur lors du reranking avec Jina: {e}")
            return documents[:top_k]

def create_reranker():
    """
    Crée une instance du reranker Jina si possible
    
    Returns:
        Un objet reranker ou None si impossible à initialiser
    """
    try:
        if not JINA_API_KEY:
            logger.warning("Clé API Jina non configurée, reranker désactivé")
            return None
        
        reranker = JinaReranker()
        logger.info("Reranker Jina créé avec succès")
        return reranker
    except Exception as e:
        logger.error(f"Erreur lors de la création du reranker Jina: {e}")
        return None

# Pour tester le module directement
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
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
    
    # Test du reranker
    reranker = create_reranker()
    if reranker:
        print(f"Type de reranker: {type(reranker).__name__}")
        result = reranker.rerank("Comment résoudre un job bloqué?", test_docs)
        for i, doc in enumerate(result):
            print(f"{i+1}. {doc.metadata.get('title')}: {doc.page_content[:50]}...")
