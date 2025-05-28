"""
Module de reranking pour améliorer la qualité des résultats de recherche.
Ce module implémente différentes stratégies de reranking pour affiner la pertinence
des documents récupérés à partir de la recherche vectorielle.
"""

import logging
import os
import requests
import json
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Vérifier si le reranker doit être utilisé
use_reranker = os.environ.get("USE_RERANKER", "False").lower() == "true"

# Récupérer la clé API Jina
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")

class BaseReranker:
    """Classe de base pour tous les rerankers"""
    def __init__(self):
        pass
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Méthode générique de reranking, à implémenter dans les sous-classes
        
        Args:
            query: La requête utilisateur
            documents: Liste des documents à réordonner
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste des documents réordonnés
        """
        raise NotImplementedError("Cette méthode doit être implémentée dans une sous-classe")

class JinaReranker(BaseReranker):
    """Utilise l'API Jina pour le reranking dans le cloud, sans dépendance locale à torch"""
    
    def __init__(self):
        super().__init__()
        self.api_key = JINA_API_KEY
        self.api_url = "https://api.jina.ai/v1/rerank"
        
        if not self.api_key:
            logger.warning("Clé API Jina non configurée, le reranker ne fonctionnera pas")
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
            return documents[:top_k]
        
        try:
            # Préparer les données pour l'API Jina
            texts = [doc.page_content for doc in documents]
            if len(texts) > 32:  # Limitation de l'API Jina
                logger.warning(f"Trop de documents ({len(texts)}) pour l'API Jina, limitation à 32")
                texts = texts[:32]
                documents = documents[:32]
            
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
            response = requests.post(self.api_url, headers=headers, json=payload)
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

class SimpleBM25Reranker(BaseReranker):
    """
    Implémentation légère d'un reranker basé sur BM25 sans dépendances externes.
    Parfait pour l'utilisation avec Streamlit et comme fallback.
    """
    def __init__(self):
        super().__init__()
        logger.info("SimpleBM25Reranker initialisé (version sans dépendances)")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Réordonne les documents en utilisant une variante simplifiée de BM25
        
        Args:
            query: La requête utilisateur
            documents: Liste des documents à réordonner
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste des documents réordonnés
        """
        if not documents:
            return []
        
        # Extraire les termes de la requête
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        # Calculer les scores pour chaque document
        scored_docs = []
        for doc in documents:
            score = 0.0
            content = doc.page_content.lower()
            
            # Composant TF (term frequency)
            for term in query_terms:
                # Nombre d'occurrences du terme dans le document
                term_count = content.count(term)
                if term_count > 0:
                    # Facteur de fréquence du terme (TF simplifié)
                    score += 1.0 + (0.5 * term_count)
            
            # Tenir compte du boost de pertinence des métadonnées
            boost = doc.metadata.get('relevance_boost', 1.0)
            score *= boost
            
            scored_docs.append((doc, score))
        
        # Trier les documents par score et retourner les top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"SimpleBM25Reranker a réordonné {len(documents)} documents")
        
        return [doc for doc, score in scored_docs[:top_k]]

def create_safe_reranker(reranker_type: str = "jina", **kwargs):
    """
    Crée un reranker de manière sécurisée en capturant les erreurs
    
    Args:
        reranker_type: Type de reranker à créer ('jina', 'simple', 'none')
        **kwargs: Arguments supplémentaires à passer au constructeur du reranker
        
    Returns:
        Un reranker ou None si une erreur se produit
    """
    if not use_reranker or reranker_type.lower() == "none":
        logger.info("Aucun reranker ne sera utilisé (désactivé par configuration)")
        return None
        
    try:
        if reranker_type.lower() == "jina":
            # Utiliser l'API Jina si la clé est disponible
            if JINA_API_KEY:
                return JinaReranker()
            else:
                logger.warning("Clé API Jina non configurée, utilisation du reranker simple comme fallback")
                return SimpleBM25Reranker()
        elif reranker_type.lower() == "simple":
            return SimpleBM25Reranker()
        else:
            logger.warning(f"Type de reranker inconnu: {reranker_type}, utilisation du reranker simple")
            return SimpleBM25Reranker()
    except Exception as e:
        logger.error(f"Erreur lors de la création du reranker: {e}")
        # Fallback sur le reranker simple en cas d'erreur
        try:
            return SimpleBM25Reranker()
        except:
            return None
