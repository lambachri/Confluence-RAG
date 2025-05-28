"""
Configuration pour le RAG Chatbot
Ce fichier est une copie de src/config.py placée à la racine pour faciliter les importations
"""
import os
from pathlib import Path
import sys
import logging

# Configurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('config')

# Chemin vers les fichiers .env possibles
paths = [
    Path(os.path.dirname(os.path.abspath(__file__))) / '.env',
    Path(os.path.dirname(os.path.abspath(__file__))) / 'src/.env',
    Path(os.path.dirname(os.path.abspath(__file__)))
]

# Tenter de charger les variables d'environnement depuis .env
env_loaded = False
for dotenv_path in paths:
    try:
        if Path(dotenv_path).is_file():
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=dotenv_path)
            logger.info(f"Variables d'environnement chargées depuis {dotenv_path}")
            env_loaded = True
            break
    except Exception as e:
        logger.warning(f"Erreur lors du chargement de {dotenv_path}: {e}")

if not env_loaded:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        env_loaded = True
        logger.info("Variables d'environnement chargées depuis le chemin par défaut")
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des variables d'environnement: {e}")
        logger.warning("Utilisation des valeurs par défaut ou des variables d'environnement du système")

# Variables d'authentification Confluence
CONFLUENCE_USERNAME = os.environ.get("EMAIL_ADRESS", "")
CONFLUENCE_API_KEY = os.environ.get("CONFLUENCE_PRIVATE_API_KEY", "")
CONFLUENCE_SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", "")
CONFLUENCE_SPACE_NAME = os.environ.get("CONFLUENCE_SPACE_NAME", "")

# Directory where vector database will be stored
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "vectorstore")

# Afficher les valeurs pour debugging (sans les clés sensibles)
logger.info(f"Configuration chargée pour {CONFLUENCE_SPACE_KEY} sur {CONFLUENCE_SPACE_NAME}")
logger.info(f"Base vectorielle dans: {PERSIST_DIRECTORY}")
