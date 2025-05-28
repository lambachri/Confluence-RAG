import sys
import load_db
import collections
import logging
import json
import os
import random
from typing import List # Added import for List
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangChainDocument # Ensure this is available for type hinting
from langchain_core.retrievers import BaseRetriever # Ensure this is available

# Configure le logger pour afficher plus d'informations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('help_desk')

# Import du reranker seulement si activé
USE_RERANKER = os.environ.get("USE_RERANKER", "False").lower() == "true"
STREAMLIT_MODE = os.environ.get("STREAMLIT_MODE", "False").lower() == "true"

if USE_RERANKER:
    try:
        # En mode streamlit, utiliser jina_reranker qui n'a pas de dépendances torch
        if STREAMLIT_MODE:
            from src.jina_reranker import create_reranker
            logger.info("Module jina_reranker importé avec succès")
        else:
            # Pour d'autres contextes (CLI, etc.), on peut utiliser le reranker standard
            from src.reranker import create_safe_reranker
            logger.info("Module de reranking standard importé avec succès")
    except Exception as e:
        logger.warning(f"Erreur lors de l'import du module de reranking: {e}")
        USE_RERANKER = False

class ConversationMemory:
    """Gère la mémoire conversationnelle pour maintenir le contexte entre les questions"""
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.conversation_history = []  # Liste des paires (question, réponse)
        self.document_cache = {}  # Cache des documents pertinents (doc_id -> score)
        self.relevant_keywords = set()  # Mots-clés importants identifiés dans la conversation
        self.current_topic = None  # Thème global de la conversation actuelle
        
    def add_exchange(self, question, answer, documents=None):
        """Ajoute un échange à l'historique et met à jour le cache de documents"""
        # Ajouter l'échange à l'historique
        self.conversation_history.append({"question": question, "answer": answer})
        
        # Limiter la taille de l'historique
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Mettre à jour le cache de documents
        if documents:
            for doc in documents:
                doc_id = self._get_document_id(doc)
                # Augmenter le score des documents utilisés récemment
                self.document_cache[doc_id] = self.document_cache.get(doc_id, 0) + 1
        
        # Mettre à jour les mots-clés pertinents
        self._extract_keywords_from_question(question)
        
        # Identifier ou mettre à jour le thème de la conversation
        self._update_conversation_topic()
    
    def get_conversation_context(self):
        """Renvoie le contexte de la conversation actuelle"""
        return {
            "history": self.conversation_history,
            "relevant_keywords": list(self.relevant_keywords),
            "topic": self.current_topic
        }
    
    def get_recent_documents(self, decay_factor=0.8, top_k=5):
        """Renvoie les documents les plus pertinents avec décroissance temporelle"""
        # Appliquer un facteur de décroissance aux scores des documents plus anciens
        decayed_scores = {}
        for doc_id, score in self.document_cache.items():
            decayed_scores[doc_id] = score * decay_factor
        
        # Trier les documents par score décroissant
        sorted_docs = sorted(decayed_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Renvoyer les top_k documents
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    def is_followup_question(self, question):
        """Détermine si une question est une question de suivi basée sur des indicateurs linguistiques"""
        # Indicateurs explicites de référence à une conversation précédente
        explicit_indicators = [
            "cela", "ça", "ce", "cette", "ces", "celui", "celle", "ceux", "celles",
            "il", "elle", "ils", "elles", "le", "la", "les", "leur", "leurs",
            "précédent", "mentionné", "dit", "parlé", "évoqué", "cité", "indiqué",
            "comme", "aussi", "également", "encore", "toujours", "même"
        ]
        
        # Vérifier si la question contient des indicateurs explicites
        if any(indicator in question.lower().split() for indicator in explicit_indicators):
            return True
        
        # Vérifier si la question est courte (souvent une question de suivi)
        if len(question.split()) < 4:
            return True
        
        # Vérifier si la question commence par un verbe (sans sujet) 
        # ou par une conjonction (et, mais, donc, etc.)
        first_word = question.lower().split()[0]
        starting_verbs = ["est", "sont", "peut", "peux", "dois", "doit", "faut", "a", "ont", "fait"]
        conjunctions = ["et", "mais", "donc", "or", "ni", "car", "puis", "ensuite", "alors"]
        
        if first_word in starting_verbs or first_word in conjunctions:
            return True
        
        # Si aucune condition n'est remplie, ce n'est probablement pas une question de suivi
        return False
    
    def reset(self):
        """Réinitialise complètement la mémoire conversationnelle"""
        self.conversation_history = []
        self.document_cache = {}
        self.relevant_keywords = set()
        self.current_topic = None
    
    def _get_document_id(self, doc):
        """Génère un identifiant unique pour un document"""
        # Utiliser le titre et l'URL comme identifiant
        return f"{doc.metadata.get('title', '')}|{doc.metadata.get('source', '')}"
    
    def _extract_keywords_from_question(self, question):
        """Extrait les mots-clés importants d'une question"""
        # Liste de mots vides (stopwords) à ignorer
        stopwords = ["le", "la", "les", "un", "une", "des", "du", "de", "et", "à", "en", "est", 
                    "comment", "pourquoi", "quand", "qui", "que", "quoi", "dont", "où", "quel", 
                    "quelle", "quels", "quelles", "pour", "par", "sur", "dans", "avec"]
        
        # Extraire les mots de 4 caractères ou plus qui ne sont pas des stopwords
        words = [w.lower() for w in question.split() if len(w) >= 4 and w.lower() not in stopwords]
        
        # Ajouter les mots à l'ensemble des mots-clés pertinents
        self.relevant_keywords.update(words)
        
        # Limiter le nombre de mots-clés pour éviter une explosion
        if len(self.relevant_keywords) > 20:
            # Garder uniquement les mots-clés les plus récents
            self.relevant_keywords = set(list(self.relevant_keywords)[-20:])
    
    def _update_conversation_topic(self):
        """Identifie ou met à jour le thème global de la conversation"""
        # Si pas encore d'historique, impossible de déterminer un thème
        if not self.conversation_history:
            return
        
        # Domaines techniques possibles
        tech_domains = {
            "opcon": ["opcon", "scheduler", "job", "planning", "batch", "tâche"],
            "monitoring": ["monitoring", "surveillance", "alerte", "notification", "statut"],
            "m3": ["m3", "infor", "erp", "movex"],
            "cegid": ["cegid", "y2", "retail", "pgi"],
            "integration": ["intégration", "api", "interface", "connecteur", "flux"],
            "azure": ["azure", "cloud", "blob", "storage", "vm"],
            "database": ["base de données", "sql", "table", "requête", "query"],
            "authentication": ["authentification", "connexion", "login", "access", "permission"]
        }
        
        # Compter les occurrences de mots-clés par domaine
        domain_scores = {domain: 0 for domain in tech_domains}
        
        # Analyser toutes les questions récentes
        for exchange in self.conversation_history:
            question = exchange["question"].lower()
            
            for domain, keywords in tech_domains.items():
                for keyword in keywords:
                    if keyword in question:
                        domain_scores[domain] += 1
        
        # Sélectionner le domaine avec le score le plus élevé
        if domain_scores:
            max_domain = max(domain_scores.items(), key=lambda x: x[1])
            # Ne mettre à jour que si le score est significatif
            if max_domain[1] > 0:
                self.current_topic = max_domain[0]

class HelpDesk():
    """Create the necessary objects to create a QARetrieval chain"""
    def __init__(self, new_db=False, use_reranker=None):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()
        
        # Déterminer si on utilise le reranker
        if use_reranker is None:
            # Utiliser la variable d'environnement si non spécifié
            use_reranker = os.environ.get("USE_RERANKER", "False").lower() == "true"
        self.use_reranker = use_reranker
        
        # Simply use the DataLoader to get the database
        data_loader = load_db.DataLoader()
        self.db = data_loader.set_db(self.embeddings, test_mode=False)

        # Use an improved retriever from the database - removed score_threshold
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20 if self.use_reranker else 12  # Augmenter pour le reranking, sinon valeur standard
            }
        )
        
        # Initialiser le reranker si activé
        self.reranker = None
        if self.use_reranker:
            try:
                # En mode streamlit, utiliser jina_reranker
                if STREAMLIT_MODE:
                    from src.jina_reranker import create_reranker
                    self.reranker = create_reranker()
                    if self.reranker:
                        logger.info(f"Reranker {type(self.reranker).__name__} initialisé avec succès")
                    else:
                        logger.warning("Reranker non disponible, utilisation du retriever standard")
                else:
                    # Pour d'autres contextes, utiliser le reranker standard
                    from src.reranker import create_safe_reranker
                    self.reranker = create_safe_reranker("cross-encoder")
                    if self.reranker:
                        logger.info("Reranker CrossEncoder initialisé avec succès")
                    else:
                        logger.warning("Reranker non disponible, utilisation du retriever standard")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du reranker: {e}")
                self.reranker = None
        
        self.retrieval_qa_chain = self.get_retrieval_qa()
        # Initialisation de la mémoire conversationnelle
        self.conversation_memory = None
        logger.info(f"HelpDesk initialisé avec succès - Retriever configuré avec k={20 if self.use_reranker else 12}, reranker: {self.reranker is not None}")

    def init_conversation_memory(self):
        """Initialise ou réinitialise la mémoire conversationnelle"""
        self.conversation_memory = ConversationMemory()
        logger.info("Mémoire conversationnelle initialisée")

    def get_template(self):
        template = """
        Tu es un expert technique du support L2 pour le monitoring des schedulers dans OpCon et les systèmes connexes (M3, Cegid, etc.). Tu es connu pour donner des réponses précises, directes et pratiques à tes collègues.
        
        Voici des informations techniques extraites de notre documentation interne :
        -----
        {context}
        -----

        Question de ton collègue: {question}
        
        Instructions:
        1. Réponds directement à la question avec les étapes concrètes à suivre en te basant sur les informations fournies dans le contexte ci-dessus.
        2. Si le contexte ne contient PAS d'informations pertinentes sur le sujet SPÉCIFIQUE demandé, mais contient des informations partielles ou adjacentes, fournis quand même ces informations en précisant les limites de ta réponse.
        3. Utilise uniquement les informations factuelles présentes dans le contexte. Si aucune information n'est trouvée, indique-le clairement.
        4. Utilise une structure en puces ou étapes numérotées pour les procédures.
        5. Ne mentionne jamais les sources documentation ou la provenance des informations.
        6. Reste factuel et ne crée jamais d'information qui n'est pas explicitement mentionnée dans les extraits.
        7. Concentre-toi sur les commandes précises, les paramètres et les étapes techniques.
        8. Utilise un ton professionnel mais amical, comme si tu parlais à un collègue.
        9. Si pertinent, mentionne les erreurs fréquentes à éviter.
        10. IMPORTANT: Ne réponds PAS "Je n'ai pas d'information" simplement parce que le nom exact "OpCon" n'est pas mentionné. Notre documentation couvre également des systèmes connexes comme M3, Cegid, et divers processus métier.
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        return prompt

    def get_embeddings(self) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings()
        return embeddings

    def get_llm(self):
        # Utiliser ChatOpenAI avec des paramètres optimisés pour des réponses plus précises
        llm = ChatOpenAI(
            temperature=0.3,  # Baissé pour plus de précision
            model_name="gpt-4.1-nano",  # Plus de contexte pour mieux comprendre les documents
            timeout=90,  # Augmenter le timeout pour des réponses plus élaborées
        )
        return llm

    def get_retrieval_qa(self):
        chain_type_kwargs = {"prompt": self.prompt}
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        return qa

    def _extract_time_information(self, question, documents):
        """
        Extrait directement les informations d'horaires des documents pour les questions simples sur les temps.
        """
        # Identifier ce qui est demandé
        question_lower = question.lower()
        
        # Chercher des informations sur les runs Galaxy
        if "galaxy" in question_lower and "run" in question_lower:
            # Chercher des mentions de Run Normal, Run BUY, Run Backup
            run_types = []
            if "normal" in question_lower:
                run_types.append("Run Normal")
            elif "buy" in question_lower:
                run_types.append("Run BUY")
            elif "backup" in question_lower:
                run_types.append("Run Backup")
            elif "referentiel" in question_lower:
                run_types.append("Referentiel")
            
            # Si on demande tous les runs ou aucun spécifique
            if "runs" in question_lower or not run_types:
                return self._extract_all_run_times(None, documents)
            
            # Si on demande un run spécifique
            for doc in documents:
                content = doc.page_content.lower()
                for run_type in run_types:
                    if run_type.lower() in content and "galaxy" in content:
                        # Chercher l'heure de démarrage dans le document
                        import re
                        # Chercher des patterns comme "start_time: 22:40" ou "démarre à 22:40"
                        time_patterns = [
                            r'start_time:\s*(\d{1,2})[:\.](\d{2})',
                            r'démarre à\s*(\d{1,2})[:\.](\d{2})',
                            r'commence à\s*(\d{1,2})[:\.](\d{2})',
                            r'tourne à\s*(\d{1,2})[:\.](\d{2})',
                            r'programmé[e]? pour\s*(\d{1,2})[:\.](\d{2})',
                            r'(\d{1,2})[:\.](\d{2})'  # Pattern générique en dernier recours
                        ]
                        
                        for pattern in time_patterns:
                            match = re.search(pattern, content)
                            if match:
                                time_str = f"{match.group(1)}:{match.group(2)}"
                                return f"Le {run_type} Galaxy est programmé pour tourner à {time_str}."
        
        # Si on n'a pas trouvé d'information directe
        return None
    
    def _extract_all_run_times(self, raw_response=None, documents=None):
        """
        Extrait les informations sur tous les runs Galaxy mentionnés dans les documents.
        """
        run_times = {}
        
        # Utiliser les documents source si disponibles
        if documents:
            # Chercher des informations structurées dans les documents
            import re
            for doc in documents:
                content = doc.page_content.lower()
                if "galaxy" in content and "run" in content:
                    # Chercher les différents types de runs
                    run_types = ["Run Normal", "Run BUY", "Run Backup", "Referentiel"]
                    for run_type in run_types:
                        if run_type.lower() in content:
                            # Chercher l'heure associée à ce run
                            # Pattern spécifique pour les données structurées comme dans un CSV
                            pattern = rf'{run_type.lower()}.+?start_time:\s*(\d{{1,2}})[:.:](\d{{2}})'
                            match = re.search(pattern, content, re.IGNORECASE)
                            
                            if match:
                                time_str = f"{match.group(1)}:{match.group(2)}"
                                run_times[run_type] = time_str
                            else:
                                # Pattern plus générique pour trouver l'heure près du nom du run
                                context_pattern = rf'{run_type}.{{0,50}}(\d{{1,2}})[:.:](\d{{2}})'
                                match = re.search(context_pattern, content, re.IGNORECASE)
                                if match:
                                    time_str = f"{match.group(1)}:{match.group(2)}"
                                    run_times[run_type] = time_str
        
        # Si nous avons trouvé des horaires pour les runs
        if run_times:
            response = "Voici les horaires des différents runs Galaxy :\n\n"
            for run_type, time in run_times.items():
                response += f"* {run_type} : {time}\n"
            return response
        
        # Si nous avons trouvé des documents mais pas réussi à extraire les horaires structurés
        # on peut analyser les sources de manière plus générique
        galaxy_docs = []
        if documents:
            for doc in documents:
                if "galaxy" in doc.page_content.lower():
                    galaxy_docs.append(doc)
        
        if galaxy_docs:
            # Chercher tous les horaires mentionnés
            times = []
            for doc in galaxy_docs:
                # Extraire toutes les heures mentionnées
                import re
                matches = re.findall(r'(\d{1,2})[:\.](\d{2})', doc.page_content)
                for hour, minute in matches:
                    times.append(f"{hour}:{minute}")
            
            if times:
                if len(times) <= 3:  # Si peu d'horaires, probablement pertinents
                    return f"D'après les informations disponibles, les runs Galaxy tournent aux heures suivantes: {', '.join(times)}."
        
        # Si on n'a toujours pas trouvé d'information utile
        return None
    
    def _simplify_time_context(self, context):
        """
        Simplifie le contexte autour d'une mention d'heure pour créer une réponse concise.
        """
        # Nettoyer le contexte pour créer une réponse plus directe
        import re
        
        # Chercher des modèles spécifiques comme "GALAXY_RUN_NORMAL ... start_time: 22:40"
        run_info = re.search(r'(GALAXY_RUN_\w+|Run \w+).{1,30}start_time:\s*(\d{1,2})[:\.](\d{2})', context, re.IGNORECASE)
        if run_info:
            run_name = run_info.group(1).replace('GALAXY_RUN_', 'Run ').title()
            time_str = f"{run_info.group(2)}:{run_info.group(3)}"
            return f"Le {run_name} Galaxy tourne à {time_str}."
        
        # Chercher "Run X tourne/démarre/commence à HH:MM"
        run_time = re.search(r'(Run \w+).{1,20}(?:tourne|démarre|commence|exécute).{1,10}(\d{1,2})[:\.](\d{2})', context, re.IGNORECASE)
        if run_time:
            run_name = run_time.group(1)
            time_str = f"{run_time.group(2)}:{run_time.group(3)}"
            return f"Le {run_name} Galaxy tourne à {time_str}."
        
        return None
    
    def retrieval_qa_inference(self, question, verbose=True, use_context=True):
        """Traite une question et renvoie une réponse basée sur la recherche de documents pertinents"""
        logger.info(f"Question reçue: {question}")
        
        # Sauvegarder la question courante pour référence
        self._current_question = question
        
        # Vérifier si nous avons une mémoire conversationnelle
        if use_context and self.conversation_memory is None:
            self.init_conversation_memory()
            logger.info("Mémoire conversationnelle initialisée automatiquement")
        
        # First check if we have any documents in the database
        try:
            doc_count = self.db._collection.count()
            logger.info(f"Nombre de documents dans la base: {doc_count}")
            
            if doc_count == 0:
                logger.warning("La base de données vectorielle est vide! Aucun document à rechercher.")
                return "Je ne peux pas répondre car la base de connaissances est vide.", "Aucune source disponible"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la base de données: {e}")
        
        try:
            # Déterminer si c'est une question de suivi
            is_followup = False
            if use_context and self.conversation_memory and self.conversation_memory.conversation_history:
                is_followup = self.conversation_memory.is_followup_question(question)
                logger.info(f"Question identifiée comme question de suivi: {is_followup}")
            
            # Récupérer et logger les documents pertinents pour la question
            # Construire une requête plus complète mais uniquement si c'est une question de suivi
            if is_followup:
                enhanced_question = self._enhance_query(question, is_followup)
                if enhanced_question != question:
                    logger.info(f"Question enrichie (question de suivi): {enhanced_question}")
                retrieved_docs = self.retriever.get_relevant_documents(enhanced_question)
            else:
                # Pour les questions initiales, utiliser la question telle quelle pour éviter la dilution
                retrieved_docs = self.retriever.get_relevant_documents(question)
                logger.info("Question initiale: utilisation de la question originale sans enrichissement")
            
            logger.info(f"Nombre de documents récupérés: {len(retrieved_docs)}")
            
            # Si aucun document pertinent n'est trouvé, essayer une recherche plus large
            if not retrieved_docs:
                logger.warning("Aucun document trouvé avec la recherche initiale. Tentative de recherche élargie...")
                # Simplifier la question pour élargir la recherche
                simplified_question = self._simplify_query(question)
                logger.info(f"Question simplifiée: {simplified_question}")
                retrieved_docs = self.retriever.get_relevant_documents(simplified_question)
                logger.info(f"Nombre de documents récupérés après simplification: {len(retrieved_docs)}")
            
            # Filtrer les documents par pertinence sémantique
            if len(retrieved_docs) > 3:
                filtered_docs = self._filter_most_relevant_docs(retrieved_docs, question, is_followup)
                logger.info(f"Documents filtrés par pertinence: {len(filtered_docs)}/{len(retrieved_docs)}")
            else:
                filtered_docs = retrieved_docs
            
            # Utiliser le reranker pour réordonner les documents récupérés et filtrer davantage les moins pertinents
            if self.reranker is not None and filtered_docs:
                try:
                    # Effectuer le reranking
                    logger.info(f"Reranking de {len(filtered_docs)} documents")
                    reranked_docs = self.reranker.rerank(question, filtered_docs, top_k=8)
                    
                    if reranked_docs:
                        # Filtrer les documents dont le score de pertinence est faible après reranking
                        # On devrait vérifier les scores ici, mais si l'API reranker ne les fournit pas,
                        # on prend les premiers documents qui sont supposés être les plus pertinents
                        filtered_docs = reranked_docs
                        logger.info(f"Reranking terminé. Utilisation des {len(filtered_docs)} documents réordonnés.")
                    else:
                        # Si le reranking échoue, conserver les documents déjà filtrés
                        logger.warning("Échec du reranking, utilisation des documents filtrés précédemment")
                except Exception as e:
                    logger.error(f"Erreur lors du reranking: {e}")
            else:
                # Si pas de reranker, utiliser seulement les documents filtrés (max 8)
                filtered_docs = filtered_docs[:8]
                logger.info(f"Pas de reranking: utilisation des {len(filtered_docs)} documents filtrés")
            
            # Analyse supplémentaire de pertinence thématique
            if len(filtered_docs) > 1:
                # Vérifier la cohérence thématique entre les documents
                coherent_docs = self._ensure_thematic_coherence(filtered_docs, question)
                if len(coherent_docs) < len(filtered_docs):
                    logger.info(f"Filtrage thématique: {len(coherent_docs)}/{len(filtered_docs)} documents conservés")
                    filtered_docs = coherent_docs
            
            # Créer un log détaillé des documents récupérés
            log_docs = []
            for i, doc in enumerate(filtered_docs):
                # Extraire les métadonnées pertinentes
                metadata = doc.metadata
                title = metadata.get('title', 'Sans titre')
                source = metadata.get('source', 'Source inconnue')
                
                # Créer un résumé du contenu pour le log
                content_preview = doc.page_content[:150].replace('\n', ' ').strip() + "..."
                
                log_entry = {
                    'index': i+1,
                    'title': title,
                    'source': source,
                    'content_preview': content_preview,
                    'metadata': {k: v for k, v in metadata.items() if k not in ['title', 'source']}
                }
                log_docs.append(log_entry)
                
                # Logger chaque document individuellement pour facilité de lecture
                logger.info(f"Document {i+1}: {title} - {source}")
                logger.info(f"Aperçu: {content_preview}")
            
            # Sauvegarder les documents complets dans un fichier JSON pour analyse
            with open('retrieved_docs_log.json', 'w', encoding='utf-8') as f:
                json.dump(log_docs, f, ensure_ascii=False, indent=2)
            logger.info("Détails complets des documents enregistrés dans retrieved_docs_log.json")
            
            # Standard query processing
            logger.info("Traitement de la question avec RetrievalQA...")
            
            # Créer un retriever temporaire avec seulement les documents filtrés et réordonnés
            # from langchain.schema import Document as LangChainDocument # Already imported at file level
            
            # La classe StaticRetriever n'existe plus dans langchain.retrievers
            # À la place, on peut utiliser la classe de base Retriever avec une méthode _get_relevant_documents personnalisée
            # from langchain_core.retrievers import BaseRetriever # Already imported at file level
            
            class SimpleStaticRetriever(BaseRetriever):
                """Simple retriever that just returns the documents it was given."""
                documents_list: List[LangChainDocument] # Declare documents_list as a Pydantic field

                def _get_relevant_documents(self, query: str) -> List[LangChainDocument]: # query is required by BaseRetriever
                    return self.documents_list
            
            # Utiliser notre implémentation personnalisée
            # Instantiate by passing the field name `documents_list`
            temp_retriever = SimpleStaticRetriever(documents_list=filtered_docs)
            
            # Ajouter des instructions supplémentaires dans le prompt pour le modèle
            if is_followup and self.conversation_memory:
                # Récupérer le contexte conversationnel
                context = self.conversation_memory.get_conversation_context()
                # Adapter le prompt pour inclure le contexte précédent
                enhanced_prompt = self._create_enhanced_prompt_with_context(context, self.prompt)
                chain_type_kwargs = {"prompt": enhanced_prompt}
            else:
                # Utiliser le prompt standard
                chain_type_kwargs = {"prompt": self.prompt}
            
            # Créer une chaîne temporaire avec le retriever statique
            temp_qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=temp_retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )
            
            # Exécuter la chaîne
            query = {"query": question}
            answer = temp_qa.invoke(query)
            
            # Extraire et logger les sources utilisées
            sources = self.list_top_k_sources(answer, k=2)
            logger.info(f"Sources utilisées: {sources}")
            
            # Post-traitement de la réponse pour la rendre plus "chirurgicale"
            raw_result = answer["result"]
            logger.info(f"Réponse brute du LLM: {raw_result[:200]}...")
            
            response = self.format_response(raw_result)
            logger.info(f"Réponse formatée: {response[:200]}...")
            
            # Logger les changements entre la réponse brute et formatée
            if raw_result != response:
                logger.info("La réponse a été modifiée pendant le post-traitement")
            
            # Dernier ajustement de présentation pour assurer la cohérence
            response = response.replace('\n\n\n', '\n\n')
            
            # Mettre à jour la mémoire conversationnelle avec cette interaction
            if use_context and self.conversation_memory:
                self.conversation_memory.add_exchange(
                    question=question,
                    answer=response,
                    documents=answer.get("source_documents", [])
                )
                logger.info("Mémoire conversationnelle mise à jour avec la nouvelle interaction")
            
            return response, sources
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question: {e}", exc_info=True)
            return f"Une erreur s'est produite lors de la recherche: {str(e)}", "Aucune source disponible"

    def _enhance_query(self, question, is_followup=False):
        """Améliore la requête avec des termes supplémentaires pour augmenter les chances de trouver des documents pertinents"""
        # Initialiser enhanced comme une copie de la question (qui est une str)
        enhanced = question
        
        # Dictionnaire de mots-clés et leurs synonymes étendu
        keyword_map = {
            "M3": ["M3", "Infor", "ERP", "Movex"],
            "Cegid": ["Cegid", "Y2", "Retail", "Orli", "PGI"],
            "article": ["article", "produit", "référence", "item", "SKU"],
            "procédure": ["procédure", "processus", "méthode", "démarche", "étapes"],
            "contrôle": ["contrôle", "vérification", "monitoring", "supervision", "validation"],
            "opcon": ["opcon", "scheduler", "ordonnanceur", "planificateur", "job"],
            "job": ["job", "tâche", "traitement", "batch", "travail"],
            "azure": ["azure", "cloud", "microsoft", "blob", "storage"],
            "erreur": ["erreur", "problème", "incident", "bug", "défaillance"],
            "logs": ["logs", "journaux", "traces", "historique", "événements"],
            "scheduler": ["scheduler", "planificateur", "ordonnanceur", "opcon", "programmation"]
        }
        
        # Vérifier si des mots-clés connus sont présents et enrichir si besoin
        query_terms = []
        for keyword, synonyms in keyword_map.items():
            if any(term.lower() in question.lower() for term in [keyword]):
                # Ajouter certains synonyms à la recherche
                for syn in synonyms[:2]:  # Limiter pour ne pas trop diluer
                    if syn.lower() not in question.lower():
                        query_terms.append(syn)
        
        # Pour les questions de suivi, enrichir avec les mots-clés pertinents de la conversation
        if is_followup and self.conversation_memory:
            # Récupérer les mots-clés significatifs de l'historique de conversation
            context = self.conversation_memory.get_conversation_context()
            conversation_keywords = context["relevant_keywords"]
            
            # Ajouter les mots-clés pertinents au contexte (en limitant le nombre)
            for keyword in conversation_keywords[:3]:  # Limiter à 3 mots-clés du contexte
                if keyword not in query_terms and keyword.lower() not in question.lower():
                    query_terms.append(keyword)
            
            # Si un thème de conversation est identifié, l'utiliser pour guider la recherche
            topic = context.get("topic")
            if topic and topic in keyword_map:
                # Ajouter quelques synonymes du thème principal
                topic_keywords = keyword_map[topic]
                for kw in topic_keywords[:2]:
                    if kw.lower() not in question.lower() and kw not in query_terms:
                        query_terms.append(kw)
        
        # Ajouter les termes supplémentaires si pertinent
        if query_terms:
            enhanced = f"{question} {' '.join(query_terms)}"
        return enhanced

    def _simplify_query(self, question):
        """Simplifie la requête pour une recherche plus large"""
        # Extraire les mots-clés principaux
        words = question.lower().split()
        # Filtrer les mots courts et les mots vides
        stop_words = ["de", "la", "les", "des", "du", "pour", "avec", "dans", "comment", "est-ce", "que"]
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Prendre les 3-5 mots-clés les plus pertinents
        return " ".join(keywords[:5])

    def _filter_most_relevant_docs(self, retrieved_docs, question, is_followup=False):
        """
        Filtre les documents les plus pertinents en fonction de la question
        et des métadonnées de boost de pertinence, en tenant compte du contexte.
        """
        logger.info(f"Filtrage de pertinence pour {len(retrieved_docs)} documents")
        # Si moins de 5 documents, ne pas filtrer
        if len(retrieved_docs) <= 5:
            return retrieved_docs
        
        # Améliorons notre approche pour les questions de suivi
        if is_followup and self.conversation_memory:
            # Récupérer le contexte de la conversation précédente
            context = self.conversation_memory.get_conversation_context()
            prev_exchanges = context.get("history", [])
            
            if prev_exchanges:
                # Vérifier si des documents ont déjà été utilisés pour répondre à des questions similaires
                recent_doc_ids = self.conversation_memory.get_recent_documents()
                
                # Ajouter ces documents en priorité (avec un boost important)
                relevant_docs = []
                other_docs = []
                
                for doc in retrieved_docs:
                    doc_id = self.conversation_memory._get_document_id(doc)
                    if doc_id in recent_doc_ids:
                        # Document déjà utilisé dans les échanges précédents → prioritaire
                        relevant_docs.append(doc)
                    else:
                        other_docs.append(doc)
                
                # Limiter le nombre de documents pour éviter la dilution
                # Mais garder un mix de documents connus et nouveaux pour une meilleure exploration
                max_relevant = min(4, len(relevant_docs))  # Maximum 4 documents déjà connus
                max_other = min(4, len(other_docs))        # Maximum 4 nouveaux documents
                
                # Toujours conserver au moins 1 nouveau document pour éviter de tourner en boucle
                if max_relevant >= 4 and max_other == 0 and len(other_docs) > 0:
                    max_other = 1
                
                filtered_docs = relevant_docs[:max_relevant] + other_docs[:max_other]
                
                # Ne pas filtrer plus s'il reste moins de 3 documents
                if len(filtered_docs) >= 3:
                    return filtered_docs
    
        # Reste du code existant inchangé
        # Extraire les mots clés importants de la question
        question_lower = question.lower()
        keywords = []
        # Mots-clés prioritaires spécifiques aux domaines, en tenant compte du contexte.
        domain_keywords = {
            "opcon": 3.0,  # Priorité maximale
            "scheduler": 2.5,
            "job": 2.0,
            "m3": 2.0,
            "cegid": 2.0,
            "contrôle": 1.5,    
            "procédure": 1.5,
            "article": 1.5,
            "monitoring": 1.5,
            "vérification": 1.2,
            "erreur": 1.2,
            "configuration": 1.2,
            "azure": 2.0,  # Priorité maximale
            "cloud": 1.5,
            "blob": 1.5,
            "storage": 1.5,
            "lancement": 1.5,
            "script": 1.5,
            "commande": 1.5
        }
        
        # Vérifier la présence de mots-clés techniques dans la question
        for kw, weight in domain_keywords.items():
            if kw in question_lower:
                keywords.append((kw, weight))
        
        # Ajouter des mots non techniques de la question (mots de 4+ caractères)
        words = [w.lower() for w in question_lower.split() if len(w) >= 4 and w not in domain_keywords]
        for word in words:
            keywords.append((word, 1.0))  # Poids standard pour les autres mots
            
        logger.info(f"Mots-clés extraits pour le filtrage: {[kw for kw, _ in keywords]}")
            
        # Calculer les scores de pertinence pour chaque document
        scored_docs = []
        
        for doc in retrieved_docs:
            score = 0.0
            content_lower = doc.page_content.lower()
            doc_id = None
            # Générer un identifiant pour le document
            if self.conversation_memory:
                doc_id = self.conversation_memory._get_document_id(doc)
            
            # Boosting basé sur les métadonnées (si présent)
            relevance_boost = doc.metadata.get('relevance_boost', 1.0)
            
            # Calculer le score basé sur la présence des mots-clés
            for keyword, weight in keywords:
                if keyword in content_lower:
                    # Nombre d'occurrences * poids du mot-clé
                    occurrences = content_lower.count(keyword)
                    score += occurrences * weight
            
            # Pour les questions de suivi, augmenter la priorité des documents utilisés précédemment
            if is_followup and self.conversation_memory and doc_id:
                # Récupérer les IDs des documents récemment utilisés
                recent_docs = self.conversation_memory.get_recent_documents()
                if doc_id in recent_docs:
                    # Bonus de continuité pour les documents précédemment utilisés
                    continuity_bonus = 3.0 - 0.5 * recent_docs.index(doc_id)  # Plus de poids aux plus récents
                    score += continuity_bonus
                    logger.info(f"Bonus de continuité appliqué ({continuity_bonus}) au document: {doc.metadata.get('title', 'Sans titre')}")
            
            # Appliquer le boost de pertinence des métadonnées
            score *= relevance_boost
            scored_docs.append((doc, score))
        
        # Trier les documents par score décroissant
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Après filtrage: {len(scored_docs)}/{len(retrieved_docs)} documents retenus")
        # Retenir les 8 documents les plus pertinents
        filtered_docs = [doc for doc, _ in scored_docs[:8]]
        return filtered_docs

    def format_response(self, raw_response):
        """Formate la réponse pour la rendre plus directe et naturelle"""
        logger.info("Début du formatage de la réponse")
        
        # Phrases indiquant un manque d'information
        no_info_phrases = ["je n'ai pas d'information", "je n'ai pas cette information", "je n'ai pas suffisamment d'information", 
                         "pas dans ma base de connaissances", "je ne peux pas répondre", "je ne dispose pas", "je ne trouve pas"]
        
        # Vérifier si la réponse est trop générique ou sans contenu
        if any(phrase in raw_response.lower() for phrase in no_info_phrases):
            logger.info("Réponse de type 'je ne sais pas' détectée")
            # Vérifier si la réponse contient quand même des informations partielles
            information_indicators = ["cependant", "toutefois", "néanmoins", "par contre", "mais", "bien que"]
            if any(indicator in raw_response.lower() for indicator in information_indicators):
                # La réponse contient des informations partielles - extraire ces informations
                for indicator in information_indicators:
                    if indicator in raw_response.lower():
                        parts = raw_response.split(indicator, 1)
                        if len(parts) > 1 and len(parts[1]) > 50:
                            improved_response = f"Je n'ai pas d'information complète sur ce sujet précis, {indicator} {parts[1]}"
                            logger.info("Extraction d'informations partielles réussie")
                            return improved_response
            
            return "Je n'ai pas d'information spécifique sur ce sujet dans ma base de connaissances. Pourriez-vous reformuler votre question ou préciser ce que vous recherchez?"
        
        # Vérifier si la réponse est trop courte (souvent une réponse générique)
        if len(raw_response) < 50:
            logger.warning(f"Réponse trop courte ({len(raw_response)} caractères)")
            return "Je n'ai pas trouvé d'information spécifique sur ce sujet. Pourriez-vous reformuler votre question ou être plus précis?"
        
        # Détecter les questions sur des horaires ou plannings
        current_question = getattr(self, '_current_question', '')
        time_related_question = False
        time_patterns = ["heure", "horaire", "quand", "planning", "schedule", "tourne", "démarre", "commence", "start"]
        
        if current_question and any(pattern in current_question.lower() for pattern in time_patterns):
            time_related_question = True
            logger.info("Question liée à des horaires détectée")
            
            # Extraire des informations de temps directement des documents source si disponibles
            if hasattr(self, '_retrieved_docs') and self._retrieved_docs:
                direct_answer = self._extract_time_information(current_question, self._retrieved_docs)
                if direct_answer:
                    logger.info(f"Réponse directe extraite des documents: {direct_answer}")
                    return direct_answer
        
        # Prétraitement pour améliorer la structure de la réponse
        raw_response = self._preprocess_text_structure(raw_response)
        
        # Pour les questions factuelles simples, essayer d'extraire directement la réponse
        if time_related_question:
            # Extraire directement les heures et valeurs numériques
            import re
            
            # Chercher des motifs comme "22:40", "22h40", "22 h 40"
            time_patterns = [
                r'\b(\d{1,2})[:\.](\d{2})\b',           # 22:40 or 22.40
                r'\b(\d{1,2})h(\d{2})\b',               # 22h40
                r'\b(\d{1,2})\s*h\s*(\d{2})\b',         # 22 h 40
                r'\b(\d{1,2})\s*heures?\s*(\d{2})?\b'   # 22 heures 40 or 22 heures
            ]
            
            # Extraire toutes les mentions d'heures
            found_times = []
            for pattern in time_patterns:
                matches = re.finditer(pattern, raw_response)
                for match in matches:
                    # Capturer le contexte autour de l'heure trouvée
                    start_pos = max(0, match.start() - 50)
                    end_pos = min(len(raw_response), match.end() + 50)
                    context = raw_response[start_pos:end_pos]
                    
                    # Extraire l'heure au format standard
                    if match.lastindex == 1:  # S'il n'y a qu'un groupe capturé (cas de "22 heures")
                        time_str = f"{match.group(1)}:00"
                    else:
                        time_str = f"{match.group(1)}:{match.group(2)}"
                    
                    found_times.append((time_str, context))
            
            # Si nous avons trouvé des heures et que la question concerne des heures spécifiques
            if found_times:
                # Identifier les types de runs Galaxy mentionnés
                run_types = []
                if "run normal" in current_question.lower() or "normal" in current_question.lower():
                    run_types.append("Run Normal")
                elif "run buy" in current_question.lower() or "buy" in current_question.lower():
                    run_types.append("Run BUY")
                elif "run backup" in current_question.lower() or "backup" in current_question.lower():
                    run_types.append("Run Backup")
                elif "referentiel" in current_question.lower():
                    run_types.append("Referentiel")
                
                # Si on demande tous les runs Galaxy
                if "runs galaxy" in current_question.lower() or "tous les runs" in current_question.lower():
                    # Créer une réponse directe avec tous les horaires trouvés
                    runs_info = self._extract_all_run_times(raw_response, self._retrieved_docs if hasattr(self, '_retrieved_docs') else None)
                    if runs_info:
                        return runs_info
                
                # Si on demande un run spécifique
                elif run_types:
                    for time_str, context in found_times:
                        # Vérifier si le contexte de l'heure correspond au run demandé
                        if any(run_type.lower() in context.lower() for run_type in run_types):
                            return f"Le {run_types[0]} Galaxy est programmé pour tourner à {time_str}."
                
                # Si aucun run spécifique n'est identifié mais qu'on a trouvé des heures
                # Prendre la première heure trouvée avec son contexte
                time_str, context = found_times[0]
                
                # Nettoyer et simplifier le contexte
                simplified_context = self._simplify_time_context(context)
                if simplified_context:
                    return simplified_context
                
                # Si le contexte n'est pas utile, donner juste l'heure
                return f"D'après les informations disponibles, le run Galaxy tourne à {time_str}."
        
        # Nous acceptons maintenant toutes les réponses substantielles, qu'elles contiennent 
        # ou non des mots-clés techniques spécifiques à OpCon
        
        lines = raw_response.split("\n")
        formatted_lines = []
        in_list = False
        list_type = None  # 'bullet' ou 'numbered'
        
        # Première passe: détecter et marquer les éléments de structure
        for i, line in enumerate(lines):
            # Sauvegarder la ligne originale avant modifications
            original_line = line
            
            # Détecter si nous sommes dans une liste à puces ou numérotée
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list or list_type != 'bullet':
                    # Nouvelle liste à puces ou changement de type
                    if i > 0 and formatted_lines and formatted_lines[-1].strip() and not formatted_lines[-1].endswith(':'):
                        formatted_lines.append('')  # Ajoute une ligne vide avant la liste
                    in_list = True
                    list_type = 'bullet'
            elif line.strip() and line.strip()[0].isdigit() and '. ' in line.strip()[:5]:
                if not in_list or list_type != 'numbered':
                    # Nouvelle liste numérotée ou changement de type
                    if i > 0 and formatted_lines and formatted_lines[-1].strip() and not formatted_lines[-1].endswith(':'):
                        formatted_lines.append('')  # Ajoute une ligne vide avant la liste
                    in_list = True
                    list_type = 'numbered'
            elif line.strip():  # Ligne avec du contenu mais pas une liste
                # Si on sort d'une liste, ajouter une ligne vide après la liste
                if in_list and i > 0:
                    formatted_lines.append('')
                in_list = False
                list_type = None
            
            # Supprimer les formules d'introduction
            intro_phrases = ["Selon", "D'après", "Voici", "Je vais", "Je vous présente", "Dans la documentation"]
            if any(line.strip().startswith(phrase) for phrase in intro_phrases):
                # Extraire seulement la partie informative après l'introduction
                for phrase in intro_phrases:
                    if line.strip().startswith(phrase):
                        content_part = line.split(phrase, 1)[1]
                        if content_part and len(content_part.strip()) > 10:  # Vérifier qu'il reste du contenu significatif
                            line = content_part.strip()
                        break
            
            # Suppression des mentions aux sources
            if "source" in line.lower() or "document" in line.lower() or "confluence" in line.lower():
                logger.info(f"Ligne supprimée car mentionne une source: '{line}'")
                continue
            
            # Optimisation des listes à puces pour les instructions d'action
            if line.strip() and not (line.strip().startswith("-") or line.strip().startswith("*")) and not line.strip()[0].isdigit():
                action_verbs = ["cliquer", "sélectionner", "ouvrir", "utiliser", "exécuter", "entrer", "vérifier", "configurer", "installer"]
                if any(verb in line.lower() for verb in action_verbs) and not ":" in line:
                    line = "* " + line
                    logger.info(f"Ajout d'une puce à une instruction: '{line}'")
            
            # Améliorer la mise en forme des titres et sections
            if line.strip() and len(line.strip()) < 80 and not in_list:
                # Si la ligne ressemble à un titre (courte, sans ponctuation finale)
                if not any(c in line.strip()[-1] for c in ['.', ',', ';', ':', '?', '!']):
                    # C'est probablement un titre de section
                    line = f"\n**{line.strip()}**"  # Mettre en gras et ajouter un saut de ligne avant
            
            # Vérifier si la ligne a été modifiée
            if line != original_line:
                logger.info(f"Ligne modifiée: '{original_line}' -> '{line}'")
            
            # Ajouter la ligne traitée
            formatted_lines.append(line)
        
        # Rejoindre et nettoyer
        response = "\n".join(formatted_lines).strip()
        
        # Assurer des sauts de lignes cohérents
        response = response.replace('\n\n\n', '\n\n')  # Remplace les triples sauts par des doubles
        
        # Standardiser le format des puces
        response = response.replace('- ', '* ')
        
        # Ajouter des espaces après les numéros dans les listes numérotées si nécessaire
        response = self._fix_numbered_lists(response)
        
        # Assurer la séparation des paragraphes
        response = self._ensure_paragraph_separation(response)
        
        return response
    
    def _preprocess_text_structure(self, text):
        """Prétraitement pour améliorer la structure du texte"""
        # Identifier les étapes numérotées qui pourraient être sur une seule ligne
        if "étape" in text.lower() or "procédure" in text.lower() or "suivez" in text.lower():
            # Chercher des motifs comme "Étape 1: Faire ceci. Étape 2: Faire cela."
            step_patterns = [
                (r'(Étape \d+[\s:]+)', r'\n\1'),
                (r'(\d+\.\s+[A-Z])', r'\1'),
                (r'(\.\s+)(\d+\.\s+)', r'\1\n\2'),
                (r'(\n\s*\d+\.)', r'\n\1')  # Assurer que les lignes numérotées commencent par un saut de ligne
            ]
            
            for pattern, replacement in step_patterns:
                import re
                text = re.sub(pattern, replacement, text)
        
        # Restructurer les listes à puces qui sont fusionnées
        bullet_patterns = [
            (r'(\*\s+[^*\n]+)(\*\s+)', r'\1\n\2'),
            (r'(-\s+[^-\n]+)(-\s+)', r'\1\n\2'),
            (r'(\n\s*[\*-])', r'\n\1')  # Assurer que les puces commencent par un saut de ligne
        ]
        
        for pattern, replacement in bullet_patterns:
            import re
            text = re.sub(pattern, replacement, text)
        
        # Ajouter des sauts de ligne après les titres potentiels (phrases courtes terminant par :)
        text = text.replace(':\n', ':\n\n')
        
        return text
    
    def _fix_numbered_lists(self, text):
        """Corrige le formatage des listes numérotées"""
        import re
        # Assurer qu'il y a un espace après le numéro et le point dans les listes numérotées
        return re.sub(r'(\n\d+\.)(\S)', r'\1 \2', text)
    
    def _ensure_paragraph_separation(self, text):
        """Assure une bonne séparation entre les paragraphes"""
        import re
        
        # Séparer les paragraphes qui ne sont pas des listes
        paragraphs = re.split(r'\n\s*\n', text)
        formatted_paragraphs = []
        
        for p in paragraphs:
            # Si le paragraphe n'est pas une liste ou un titre
            if not (p.strip().startswith('*') or p.strip().startswith('-') or 
                   p.strip().startswith('1.') or p.strip().startswith('**')):
                # Séparer les phrases longues par des sauts de ligne
                sentences = re.split(r'(?<=[.!?])\s+', p)
                if len(sentences) > 1 and len(p) > 150:
                    # Si c'est un paragraphe long avec plusieurs phrases
                    p = '\n'.join(sentences)
            
            formatted_paragraphs.append(p)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def list_top_k_sources(self, answer, k=2):
        """Liste les sources les plus pertinentes pour la réponse"""
        sources = []
        
        # Sauvegarder les documents sources pour référence future
        self._retrieved_docs = answer.get("source_documents", [])
        
        # Filtrer les sources pour n'inclure que celles qui sont utiles
        for res in answer["source_documents"]:
            # Vérifier si c'est une source Confluence ou un fichier utile
            source_url = res.metadata.get("source", "")
            title = res.metadata.get("title", "")
            
            # Ne pas inclure les fichiers CSV ou autres fichiers de données brutes comme sources
            if ".csv" in source_url.lower() or ".json" in source_url.lower() or "uploaded_files" in source_url.lower():
                continue
                
            # Inclure uniquement les sources Confluence valides
            if "atlassian.net" in source_url or "confluence" in source_url.lower():
                sources.append(f'[{title}]({source_url})')
        
        # Initialiser distinct_sources comme une liste vide par défaut
        distinct_sources = []
        distinct_sources_str = ""
        if sources:
            k = min(k, len(sources))
            # Vérifier qu'il y a au moins une source avant d'utiliser Counter
            counter = collections.Counter(sources)
            most_common = counter.most_common()
            if most_common:  # Vérifier que most_common n'est pas vide
                distinct_sources = [item[0] for item in most_common[:k]]
                distinct_sources_str = "  \n- ".join(distinct_sources)
        
        # Ne pas afficher de source si aucune n'est pertinente
        if not distinct_sources:
            return ""
            
        if len(distinct_sources) == 1:
            return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"
        elif len(distinct_sources) > 1:
            return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"
        else:
            return ""

    # Nouvelles méthodes pour améliorer l'expérience conversationnelle
    def get_personalized_greeting(self):
        """Génère un message d'accueil personnalisé en début de conversation"""
        greetings = [
            "Bonjour ! Je suis votre assistant technique spécialisé en OpCon. Je peux vous aider sur les questions liées aux schedulers, à la configuration des jobs ou au monitoring. Comment puis-je vous assister aujourd'hui ?",
            "Salut ! Je suis là pour répondre à vos questions techniques sur OpCon. N'hésitez pas à me demander des informations précises sur les schedulers ou les procédures de monitoring.",
            "Bienvenue ! En tant qu'expert OpCon, je suis prêt à vous aider avec les problématiques de jobs, schedulers ou monitoring. Que souhaitez-vous savoir ?"
        ]
        return random.choice(greetings)
    
    def enhance_response_with_context(self, response, message_history):
        """Améliore la réponse en tenant compte du contexte de la conversation"""
        # Si pas de mémoire conversationnelle initialisée, on ne peut pas améliorer la réponse
        if not self.conversation_memory:
            logger.warning("Impossible d'améliorer la réponse: mémoire conversationnelle non initialisée")
            return response
        
        # Extraire les questions et réponses précédentes pour contexte
        previous_exchanges = []
        for i in range(len(message_history) - 1):
            if message_history[i]["role"] == "user" and message_history[i+1]["role"] == "assistant":
                previous_exchanges.append({
                    "question": message_history[i]["content"],
                    "answer": message_history[i+1]["content"].split("Voici")[0] if "Voici" in message_history[i+1]["content"] else message_history[i+1]["content"]
                })
        
        # Si pas d'historique significatif, retourner la réponse telle quelle
        if len(previous_exchanges) < 1:
            return response
        
        # Récupérer la question actuelle
        current_question = message_history[-1]["content"]
        # Vérifier si c'est une question de suivi
        is_followup = self.conversation_memory.is_followup_question(current_question)
        # Si ce n'est pas une question de suivi, pas besoin d'améliorer la réponse
        if not is_followup and len(previous_exchanges) < 2:
            return response
        
        # Préparer un prompt pour améliorer la réponse avec le contexte
        system_prompt = """
        Tu dois améliorer la réponse en tenant compte du contexte de la conversation précédente.
        Assure-toi que la nouvelle réponse est cohérente avec les informations précédemment fournies.
        Si la question actuelle fait référence à un élément mentionné dans une réponse précédente, 
        reprends cet élément explicitement dans ta réponse pour plus de clarté.
        Évite de contredire des informations déjà données, sauf si la nouvelle question requiert une correction.
        Ne mentionne pas explicitement que tu utilises le contexte - intègre simplement les informations pertinentes.
        Garde le même ton et style que la réponse initiale.
        """
        
        # Formater le contexte de conversation
        conversation_context = "Historique de la conversation:\n"
        for i, exchange in enumerate(previous_exchanges[-3:]):  # Limiter aux 3 derniers échanges
            conversation_context += f"Question {i+1}: {exchange['question']}\n"
            conversation_context += f"Réponse {i+1}: {exchange['answer']}\n"
        # Dernière question et réponse générée
        conversation_context += f"\nQuestion actuelle: {current_question}\n"
        conversation_context += f"Réponse initiale: {response}\n"
        conversation_context += "\nRéponse améliorée avec contexte:"
        
        # Utiliser le LLM pour améliorer la réponse
        try:
            from langchain.schema import SystemMessage, HumanMessage
            improved_response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=conversation_context)
            ]).content
            # Vérifier que la réponse améliorée est valide
            if improved_response and len(improved_response) > 50:
                logger.info("Réponse améliorée avec le contexte conversationnel")
                return improved_response
            else:
                logger.warning("La réponse améliorée était trop courte ou invalide, utilisation de la réponse originale")
        except Exception as e:
            logger.error(f"Erreur lors de l'amélioration avec contexte: {e}")
        
        return response

    def _ensure_thematic_coherence(self, documents, question):
        """
        Vérifie la cohérence thématique des documents pour éviter les mélanges de sources non pertinentes.
        """
        if len(documents) <= 1:
            return documents
            
        # Extraire les mots-clés importants de la question
        question_keywords = self._extract_key_terms(question)
        logger.info(f"Mots-clés extraits de la question: {question_keywords}")
        
        # Calculer un score de pertinence thématique pour chaque document
        scored_docs = []
        for doc in documents:
            # Calculer un score basé sur la présence des mots-clés de la question
            content = doc.page_content.lower()
            score = sum(1 for kw in question_keywords if kw in content)
            
            # Ajouter un bonus pour les documents ayant un titre pertinent
            title = doc.metadata.get('title', '').lower()
            title_score = sum(2 for kw in question_keywords if kw in title)
            
            total_score = score + title_score
            scored_docs.append((doc, total_score))
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Si le score du premier document est significativement plus élevé,
        # ne garder que les documents ayant un score minimum par rapport au meilleur
        if scored_docs and scored_docs[0][1] > 0:
            top_score = scored_docs[0][1]
            threshold = max(1, top_score * 0.3)  # Au moins 30% du score du meilleur
            
            # Filtrer les documents ayant un score suffisant
            coherent_docs = [doc for doc, score in scored_docs if score >= threshold]
            
            # Limiter à 5 documents maximum pour éviter le bruit
            return coherent_docs[:5]
        
        # Si pas de scoring significatif, garder tous les documents (max 5)
        return [doc for doc, _ in scored_docs[:5]]

    def _extract_key_terms(self, text):
        """
        Extrait les termes clés d'un texte en filtrant les mots vides et les mots courts.
        """
        # Liste de mots vides en français
        stopwords = ["le", "la", "les", "un", "une", "des", "du", "de", "et", "à", "en", "est", 
                    "qui", "que", "quoi", "comment", "pourquoi", "où", "quand", "quel", "quelle", 
                    "quels", "quelles", "ce", "cette", "ces", "il", "elle", "ils", "elles", "nous", 
                    "vous", "leur", "leurs", "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
                    "pour", "par", "sur", "dans", "avec", "sans", "sous", "entre", "vers", "chez", "plus",
                    "moins", "très", "trop", "peu", "beaucoup", "aussi", "ainsi", "alors", "donc", "mais",
                    "car", "parce", "comme", "puis", "après", "avant", "pendant", "depuis", "jusqu"]
        
        # Normaliser le texte
        text = text.lower()
        
        # Extraire les mots de 4 caractères ou plus qui ne sont pas des stopwords
        words = [w for w in text.split() if len(w) >= 4 and w not in stopwords]
        
        return words

    def _create_enhanced_prompt_with_context(self, context, base_prompt):
        """
        Crée un prompt enrichi qui inclut le contexte conversationnel pour les questions de suivi.
        """
        # Extraire les informations pertinentes du contexte
        topic = context.get("topic", "")
        keywords = context.get("relevant_keywords", [])[:5]  # Limiter à 5 mots-clés
        
        # Récupérer la dernière question-réponse pour le contexte immédiat
        history = context.get("history", [])
        last_exchange = history[-1] if history else None
        
        # Créer un contexte textuel
        context_text = ""
        if topic:
            context_text += f"Le sujet principal de la conversation est: {topic}.\n"
        
        if last_exchange:
            context_text += f"Question précédente: {last_exchange['question']}\n"
            context_text += f"Réponse précédente: {last_exchange['answer']}\n\n"
        
        # Créer un nouveau template qui intègre directement le contexte conversationnel 
        # sans ajouter de nouvelle variable d'entrée
        enhanced_template = self.template.replace(
            "{context}", 
            f"Contexte de la conversation:\n{context_text}\n\nInformations techniques:\n{{context}}"
        )
        
        # Créer un nouveau prompt avec le même template modifié et les mêmes variables d'entrée
        from langchain.prompts import PromptTemplate
        enhanced_prompt = PromptTemplate(
            template=enhanced_template,
            input_variables=["context", "question"]  # Utiliser uniquement les variables existantes
        )
        
        return enhanced_prompt

    def _extract_context_around_match(self, text, match, window_size=50):
        """Extrait le contexte autour d'une correspondance dans le texte."""
        index = text.find(match)
        if index == -1:
            return None
        
        # Déterminer le début et la fin de la fenêtre
        start = max(0, index - window_size)
        end = min(len(text), index + len(match) + window_size)
        
        # Extraire le contexte
        context = text[start:end].strip()
        
        # Si le contexte commence au milieu d'une phrase, chercher le début de phrase
        if start > 0:
            sentence_start = context.find(". ")
            if sentence_start != -1:
                context = context[sentence_start + 2:]
        
        # Si le contexte se termine au milieu d'une phrase, chercher la fin de phrase
        sentence_end = context.rfind(". ")
        if sentence_end != -1 and sentence_end < len(context) - 2:
            context = context[:sentence_end + 1]
        
