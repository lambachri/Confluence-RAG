import sys
import os
import argparse
import logging
import json
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('load_specific_pages')

sys.path.append('../')
from src.config import PERSIST_DIRECTORY
from load_db import DataLoader

def load_pages_from_json(filename):
    """Charge les IDs de pages depuis un fichier JSON"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Le format peut être soit une liste simple, soit un dictionnaire avec une clé "pages"
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "pages" in data:
            return data["pages"]
        else:
            logger.error(f"Format de fichier JSON non reconnu: {filename}")
            return []
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier JSON {filename}: {e}")
        return []

def load_pages_from_csv(filename):
    """Charge les IDs de pages depuis un fichier CSV"""
    try:
        df = pd.read_csv(filename)
        
        # Chercher une colonne qui pourrait contenir les IDs
        id_columns = [col for col in df.columns if "id" in col.lower()]
        
        if id_columns:
            # Utiliser la première colonne qui contient "id" dans son nom
            return df[id_columns[0]].astype(str).tolist()
        else:
            # Utiliser la première colonne par défaut
            return df.iloc[:, 0].astype(str).tolist()
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier CSV {filename}: {e}")
        return []

def parse_comma_separated_ids(ids_string):
    """Parse une chaîne d'IDs séparés par des virgules"""
    if not ids_string:
        return []
    
    # Nettoyer la chaîne et diviser par les virgules
    ids = [id.strip() for id in ids_string.split(',')]
    # Filtrer les valeurs vides
    return [id for id in ids if id]

def load_pages_from_file(filename):
    """Charge les IDs de pages depuis un fichier (détecte automatiquement le format)"""
    if filename.endswith('.json'):
        return load_pages_from_json(filename)
    elif filename.endswith('.csv'):
        return load_pages_from_csv(filename)
    else:
        # Supposer un fichier texte simple avec un ID par ligne
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {filename}: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Charge des pages Confluence spécifiques dans ChromaDB")
    
    # Options pour spécifier les pages
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Fichier contenant les IDs des pages (JSON, CSV ou texte)')
    group.add_argument('--ids', type=str, nargs='+', help='Liste d\'IDs de pages séparés par des espaces')
    group.add_argument('--comma-ids', type=str, help='Liste d\'IDs de pages séparés par des virgules (ex: 1225478,78855)')
    
    # Option pour réinitialiser la base
    parser.add_argument('--reset', action='store_true', help='Supprimer la base existante avant de la recréer')
    
    args = parser.parse_args()
    
    # Récupérer les IDs des pages
    page_ids = []
    if args.file:
        page_ids = load_pages_from_file(args.file)
    elif args.ids:
        page_ids = args.ids
    elif args.comma_ids:
        page_ids = parse_comma_separated_ids(args.comma_ids)
    
    if not page_ids:
        logger.error("Aucun ID de page valide trouvé.")
        return
    
    logger.info(f"Chargement de {len(page_ids)} pages avec les IDs suivants:")
    for i, pid in enumerate(page_ids):
        logger.info(f"  {i+1}. {pid}")
    
    # Supprimer la base existante si demandé
    if args.reset and os.path.exists(PERSIST_DIRECTORY):
        import shutil
        logger.info(f"Suppression de la base existante: {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)
    
    # Initialiser les embeddings
    embeddings = OpenAIEmbeddings()
    
    # Charger les pages spécifiques
    data_loader = DataLoader(page_ids=page_ids)
    db = data_loader.set_db(embeddings)
    
    # Afficher des statistiques sur la base créée
    try:
        doc_count = db._collection.count()
        logger.info(f"Base de données créée avec succès contenant {doc_count} chunks.")
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la base de données: {e}")

if __name__ == "__main__":
    main()
