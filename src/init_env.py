"""
Configure l'environnement avant l'importation des modules potentiellement problématiques
Ce fichier doit être importé au tout début des scripts principaux
"""
import os
import sys
import logging
from pathlib import Path

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('init_env')

# Désactiver la télémétrie de chromadb pour éviter les problèmes avec opentelemetry
os.environ["CHROMADB_TELEMETRY"] = "false"
os.environ["CHROMADB_CLIENT_NAME"] = "langchain-streamlit"

# Configuration pour éviter les avertissements et erreurs courants
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ajuster le chemin d'importation pour être compatible avec Streamlit Cloud
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
    logger.info(f"Ajout au path: {root_dir}")

# Configurer pour utiliser des caches en mémoire si possible
os.environ["CHROMADB_USE_MEMORY"] = "true"

# Réduire la verbosité de certaines bibliothèques
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('opentelemetry').setLevel(logging.ERROR)

logger.info("Environnement configuré avec succès")

def verify_imports():
    """Vérifie que les importations fonctionnent correctement"""
    try:
        # Tenter d'importer config
        try:
            from config import PERSIST_DIRECTORY
            logger.info(f"Config importé avec succès: {PERSIST_DIRECTORY}")
        except ImportError:
            try:
                from src.config import PERSIST_DIRECTORY
                logger.info(f"Config importé via src: {PERSIST_DIRECTORY}")
            except ImportError:
                logger.warning("Impossible d'importer config")
        
        # Vérifier si les dépendances clés sont installées
        try:
            import chromadb
            logger.info(f"chromadb version: {chromadb.__version__ if hasattr(chromadb, '__version__') else 'inconnu'}")
        except ImportError:
            logger.warning("chromadb n'est pas installé")
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des importations: {e}")
        return False
