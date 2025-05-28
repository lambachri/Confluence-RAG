"""
Script d'analyse des documents récupérés lors des recherches.
Ce script permet de visualiser quels documents sont récupérés pour une question donnée.
"""

import json
import os
import sys
import pandas as pd
from datetime import datetime
from help_desk import HelpDesk

def analyze_query(question, verbose=True):
    """
    Analyse le processus de récupération pour une question donnée
    """
    print(f"\n=== ANALYSE DE LA QUESTION: '{question}' ===\n")
    
    # Initialiser le modèle
    print("Initialisation du modèle...")
    model = HelpDesk(new_db=False)
    
    # Récupérer les documents pour la question (sans formater la réponse)
    print("Récupération des documents pertinents...")
    retrieved_docs = model.retriever.get_relevant_documents(question)
    
    print(f"Nombre de documents récupérés: {len(retrieved_docs)}")
    
    # Créer une liste de résultats pour analyse
    results = []
    for i, doc in enumerate(retrieved_docs):
        # Extraire les métadonnées importantes
        metadata = doc.metadata
        title = metadata.get('title', 'Sans titre')
        source = metadata.get('source', 'Source inconnue')
        
        # Créer un aperçu du contenu
        content_preview = doc.page_content[:300].replace('\n', ' ').strip() + "..."
        
        result = {
            'index': i+1,
            'title': title,
            'source': source,
            'content_preview': content_preview,
            'content_length': len(doc.page_content),
            'metadata': {k: v for k, v in metadata.items() if k not in ['title', 'source']}
        }
        results.append(result)
        
        # Afficher les résultats si demandé
        if verbose:
            print(f"\n--- Document {i+1}: {title} ---")
            print(f"Source: {source}")
            print(f"Longueur: {len(doc.page_content)} caractères")
            print(f"Aperçu: {content_preview}")
            if 'relevance_boost' in metadata:
                print(f"Boost de pertinence: {metadata['relevance_boost']}")
    
    # Sauvegarder les résultats dans un fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"retrieval_analysis_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'question': question,
            'results': results,
            'timestamp': timestamp
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nAnalyse sauvegardée dans: {filename}")
    
    # Tester la réponse complète
    print("\n=== RÉPONSE GÉNÉRÉE ===")
    response, sources = model.retrieval_qa_inference(question)
    print(response)
    print("\n=== SOURCES ===")
    print(sources)
    
    return results

def analyze_multiple_questions(questions):
    """
    Analyse plusieurs questions et génère un rapport comparatif
    """
    all_results = {}
    
    for question in questions:
        print(f"\n\n{'='*80}\nANALYSE DE: {question}\n{'='*80}")
        results = analyze_query(question, verbose=False)
        all_results[question] = results
    
    # Générer un tableau comparatif des sources les plus fréquentes
    sources_counter = {}
    for question, results in all_results.items():
        for result in results:
            source = result['source']
            title = result['title']
            key = f"{title} ({source})"
            if key not in sources_counter:
                sources_counter[key] = 0
            sources_counter[key] += 1
    
    # Afficher les sources les plus fréquentes
    print("\n\n=== SOURCES LES PLUS FRÉQUENTES ===")
    for source, count in sorted(sources_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{count} occurrences: {source}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Utiliser la question fournie en argument
        question = " ".join(sys.argv[1:])
        analyze_query(question)
    else:
        # Questions de test par défaut
        test_questions = [
            "Comment configurer un job dans OpCon?",
            "Quelle est la procédure pour résoudre un job bloqué?",
            "Comment puis-je vérifier les logs d'un scheduler?",
            "Quelles sont les commandes de monitoring disponibles dans OpCon?"
        ]
        
        print("Analyse de plusieurs questions test...")
        choice = input("Voulez-vous analyser une seule question (1) ou toutes les questions test (2)? ")
        
        if choice == "1":
            print("\nChoisissez une question à analyser:")
            for i, q in enumerate(test_questions):
                print(f"{i+1}. {q}")
            q_idx = int(input("\nNuméro de la question: ")) - 1
            analyze_query(test_questions[q_idx])
        else:
            analyze_multiple_questions(test_questions)
