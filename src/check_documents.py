import os
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
# Update Chroma import
from langchain_chroma import Chroma
import pandas as pd
from collections import Counter

sys.path.append('../')
from src.config import PERSIST_DIRECTORY

def check_documents():
    """Vérifie les documents chargés dans la base ChromaDB"""
    print(f"Vérification des documents dans {PERSIST_DIRECTORY}")
    
    # Vérifie si le répertoire existe
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"ERREUR: Le répertoire {PERSIST_DIRECTORY} n'existe pas.")
        return
    
    try:
        # Initialise l'embedding
        embeddings = OpenAIEmbeddings()
        
        # Charge la base existante
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        
        # Obtient des statistiques sur les documents
        collection = db._collection
        count = collection.count()
        
        print(f"\n=== Statistiques de la base de documents ===")
        print(f"Nombre total de chunks: {count}")
        
        # Récupère les métadonnées pour analyser les documents
        if count > 0:
            # Obtient tous les IDs
            all_ids = collection.get()["ids"]
            
            # Récupère les métadonnées (limité aux 1000 premiers pour éviter des problèmes de mémoire)
            sample_size = min(1000, count)
            sample_ids = all_ids[:sample_size]
            results = collection.get(ids=sample_ids, include=["metadatas", "documents"])
            metadatas = results["metadatas"]
            documents = results.get("documents", [])
            
            # Compte les documents uniques
            unique_sources = {}
            unique_titles = {}
            content_by_title = {}
            
            for i, meta in enumerate(metadatas):
                source = meta.get("source", "Source inconnue")
                title = meta.get("title", f"Document sans titre {i}")
                
                if title not in unique_titles:
                    unique_titles[title] = 0
                    content_by_title[title] = documents[i][:150] if i < len(documents) else ""
                
                unique_titles[title] += 1
                
                if source not in unique_sources:
                    unique_sources[source] = 0
                unique_sources[source] += 1
            
            print(f"Nombre de pages uniques: {len(unique_titles)}")
            print(f"Nombre de sources uniques: {len(unique_sources)}")
            
            # Créer un DataFrame pour un affichage plus propre
            df = pd.DataFrame({
                'Titre': list(unique_titles.keys()),
                'Nombre de chunks': list(unique_titles.values()),
                'Aperçu du contenu': [content_by_title[title] for title in unique_titles.keys()]
            })
            
            # Trier par nombre de chunks (décroissant)
            df = df.sort_values(by='Nombre de chunks', ascending=False)
            
            # Affiche un résumé
            print("\n=== Top 10 documents par nombre de chunks ===")
            print(df.head(10).to_string(index=False))
            
            # Interface interactive simple
            while True:
                print("\n=== Menu d'analyse des documents ===")
                print("1. Voir tous les titres (triés par nombre de chunks)")
                print("2. Rechercher un titre par mot-clé")
                print("3. Voir les détails d'un document spécifique")
                print("4. Exporter la liste complète en CSV")
                print("5. Quitter")
                
                choice = input("Votre choix (1-5): ")
                
                if choice == '1':
                    print("\n=== Liste complète des documents ===")
                    pd.set_option('display.max_rows', None)
                    print(df[['Titre', 'Nombre de chunks']].to_string(index=False))
                    pd.reset_option('display.max_rows')
                
                elif choice == '2':
                    keyword = input("Entrez un mot-clé à rechercher dans les titres: ").lower()
                    filtered_df = df[df['Titre'].str.lower().str.contains(keyword)]
                    if len(filtered_df) > 0:
                        print(f"\n=== Documents correspondant à '{keyword}' ({len(filtered_df)} résultats) ===")
                        print(filtered_df[['Titre', 'Nombre de chunks']].to_string(index=False))
                    else:
                        print(f"Aucun document trouvé avec le mot-clé '{keyword}'")
                
                elif choice == '3':
                    try:
                        print("\nVoici les 5 premiers documents:")
                        for i, title in enumerate(df['Titre'].head(5)):
                            print(f"{i+1}. {title}")
                        
                        print("\nPour voir un autre document, entrez son numéro de titre")
                        doc_index = input("Numéro du document à examiner (ou 'r' pour rechercher par titre): ")
                        
                        if doc_index.lower() == 'r':
                            search_term = input("Partie du titre à rechercher: ").lower()
                            matching_titles = [title for title in df['Titre'] if search_term in title.lower()]
                            
                            if matching_titles:
                                print("\nTitres correspondants:")
                                for i, title in enumerate(matching_titles[:10]):
                                    print(f"{i+1}. {title}")
                                
                                if len(matching_titles) > 10:
                                    print(f"... et {len(matching_titles) - 10} autres titres")
                                
                                selection = input("\nSélectionnez un numéro: ")
                                try:
                                    selected_title = matching_titles[int(selection)-1]
                                    show_document_details(db, selected_title)
                                except (ValueError, IndexError):
                                    print("Sélection invalide.")
                            else:
                                print("Aucun titre correspondant trouvé.")
                        else:
                            try:
                                index = int(doc_index) - 1
                                if 0 <= index < len(df):
                                    selected_title = df['Titre'].iloc[index]
                                    show_document_details(db, selected_title)
                                else:
                                    print("Numéro hors limites.")
                            except ValueError:
                                print("Veuillez entrer un nombre valide.")
                    
                    except Exception as e:
                        print(f"Erreur lors de l'affichage des détails: {e}")
                
                elif choice == '4':
                    filename = input("Nom du fichier CSV (défaut: documents_chromadb.csv): ") or "documents_chromadb.csv"
                    df.to_csv(filename, index=False, encoding='utf-8')
                    print(f"Liste exportée dans {filename}")
                
                elif choice == '5':
                    break
                
                else:
                    print("Choix non valide. Veuillez réessayer.")
        
        else:
            print("AVERTISSEMENT: Aucun document n'a été trouvé dans la base.")
            
    except Exception as e:
        print(f"ERREUR lors de la vérification des documents: {e}")

def show_document_details(db, title):
    """Affiche les détails d'un document spécifique"""
    try:
        # Rechercher les documents par titre
        results = db._collection.get(
            where={"title": title},
            include=["metadatas", "documents"]
        )
        
        if results["ids"]:
            print(f"\n=== Détails du document: {title} ===")
            print(f"Nombre de chunks: {len(results['ids'])}")
            
            if "metadatas" in results and results["metadatas"]:
                print("\nMétadonnées:")
                for key, value in results["metadatas"][0].items():
                    if key != "title":  # On a déjà affiché le titre
                        print(f"- {key}: {value}")
            
            if "documents" in results and results["documents"]:
                print("\nAperçu du contenu (premier chunk):")
                content = results["documents"][0]
                # Afficher un extrait du contenu (limité à 500 caractères)
                print(content[:500] + ("..." if len(content) > 500 else ""))
                
                # Proposer de voir plus
                if input("\nVoir tous les chunks? (o/n): ").lower() == 'o':
                    for i, doc in enumerate(results["documents"]):
                        print(f"\n--- Chunk {i+1}/{len(results['documents'])} ---")
                        print(doc)
        else:
            print(f"Aucun document trouvé avec le titre: {title}")
            
    except Exception as e:
        print(f"Erreur lors de la récupération des détails: {e}")

def reset_chroma_db():
    """Supprime et réinitialise la base ChromaDB"""
    import shutil
    if os.path.exists(PERSIST_DIRECTORY):
        confirm = input(f"Êtes-vous sûr de vouloir supprimer la base ChromaDB dans {PERSIST_DIRECTORY}? (o/n): ")
        if confirm.lower() == 'o':
            shutil.rmtree(PERSIST_DIRECTORY)
            print(f"La base ChromaDB a été supprimée: {PERSIST_DIRECTORY}")
        else:
            print("Opération annulée.")
    else:
        print(f"Aucune base à supprimer: {PERSIST_DIRECTORY}")

def clear_confluence_cache():
    """Vide le cache des pages Confluence"""
    cache_file = "confluence_page_cache.json"
    if os.path.exists(cache_file):
        confirm = input(f"Êtes-vous sûr de vouloir vider le cache Confluence dans {cache_file}? (o/n): ")
        if confirm.lower() == 'o':
            with open(cache_file, 'w') as f:
                f.write('{}')
            print(f"Le cache Confluence a été vidé: {cache_file}")
        else:
            print("Opération annulée.")
    else:
        print(f"Fichier de cache non trouvé: {cache_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            reset_chroma_db()
        elif sys.argv[1] == "--clear-cache":
            clear_confluence_cache()
        else:
            print(f"Option non reconnue: {sys.argv[1]}")
            print("Options disponibles: --reset, --clear-cache")
    else:
        check_documents()