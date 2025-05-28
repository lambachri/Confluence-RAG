import os
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from collections import Counter
import pandas as pd
import logging

# Configure proper logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('../')
from src.config import PERSIST_DIRECTORY

def show_db_content():
    """Affiche rapidement les titres des documents dans ChromaDB"""
    print(f"Lecture de la base de données dans {PERSIST_DIRECTORY}")
    
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
        
        print(f"Base de données contenant {count} chunks")
        
        if count == 0:
            print("La base de données est vide.")
            return
            
        # Récupère les métadonnées pour les titres
        results = collection.get(include=["metadatas"])
        metadatas = results["metadatas"]
        
        # Compte les occurrences de chaque titre
        titles_counter = Counter()
        
        for meta in metadatas:
            title = meta.get("title", "Sans titre")
            titles_counter[title] += 1
        
        # Affiche les titres triés par fréquence
        print(f"\nNombre de documents uniques: {len(titles_counter)}")
        print("\n=== Titres des documents (par nombre de chunks) ===")
        
        # Création d'un dataframe pour un affichage propre
        df = pd.DataFrame({
            'Titre': list(titles_counter.keys()),
            'Nombre de chunks': list(titles_counter.values())
        }).sort_values(by='Nombre de chunks', ascending=False)
        
        # Affichage tableau
        pd.set_option('display.max_rows', None)
        print(df.to_string(index=False))
        
        # Option d'exportation rapide
        export = input("\nExporter cette liste en CSV? (o/n): ")
        if export.lower() == 'o':
            filename = "chromadb_documents.csv"
            df.to_csv(filename, index=False)
            print(f"Liste exportée dans {filename}")
        
    except Exception as e:
        print(f"ERREUR lors de la lecture de la base de données: {e}")

if __name__ == "__main__":
    show_db_content()
