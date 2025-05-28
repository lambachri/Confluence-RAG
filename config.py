"""
Configuration pour le RAG Chatbot
Ce fichier est une copie de src/config.py placée à la racine pour faciliter les importations
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Charger les variables d'environnement depuis .env
dotenv_path = Path(os.path.dirname(os.path.abspath(__file__))) / '.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Essayer de charger depuis un autre chemin
    alt_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'src/.env'
    if os.path.exists(alt_path):
        load_dotenv(dotenv_path=alt_path)
    else:
        load_dotenv()  # Essai standard

# Variables d'authentification Confluence
CONFLUENCE_USERNAME = os.environ.get("EMAIL_ADRESS", "")
CONFLUENCE_API_KEY = os.environ.get("CONFLUENCE_PRIVATE_API_KEY", "")
CONFLUENCE_SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", "")
CONFLUENCE_SPACE_NAME = os.environ.get("CONFLUENCE_SPACE_NAME", "")

# Directory where vector database will be stored
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "vectorstore")
