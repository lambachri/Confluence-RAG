"""
Test des connexions avec Confluence
"""

import os
import sys
from pathlib import Path
import requests
import json
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_confluence')

# Configurer les chemins d'importation
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Importer la configuration avec la nouvelle méthode flexible
try:
    from config import (
        CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
        CONFLUENCE_USERNAME, CONFLUENCE_API_KEY
    )
except ImportError:
    try:
        from src.config import (
            CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
            CONFLUENCE_USERNAME, CONFLUENCE_API_KEY
        )
    except ImportError:
        from .config import (
            CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY,
            CONFLUENCE_USERNAME, CONFLUENCE_API_KEY
        )

def test_confluence_connection():
    """Test la connexion à l'API Confluence"""
    print(f"Test de connexion à {CONFLUENCE_SPACE_NAME}")
    print(f"Espace: {CONFLUENCE_SPACE_KEY}")
    print(f"Utilisateur: {CONFLUENCE_USERNAME}")
    
    # Vérifier que les informations d'authentification sont définies
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_KEY:
        print("❌ Informations d'authentification manquantes")
        return False
    
    # Tester la connexion à Confluence
    url = f"{CONFLUENCE_SPACE_NAME}/rest/api/space/{CONFLUENCE_SPACE_KEY}"
    auth = (CONFLUENCE_USERNAME, CONFLUENCE_API_KEY)
    
    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Connexion réussie à l'espace {data.get('name', CONFLUENCE_SPACE_KEY)}")
            return True
        else:
            print(f"❌ Échec de la connexion: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la connexion: {e}")
        return False

def test_specific_pages():
    """Test la récupération de pages spécifiques"""
    # Liste des IDs à tester
    test_ids = [
        "3376545896", "3710844929", "3985899883"
    ]
    
    success_count = 0
    for page_id in test_ids:
        print(f"\nTest de récupération de la page {page_id}...")
        url = f"{CONFLUENCE_SPACE_NAME}/rest/api/content/{page_id}?expand=version,title"
        auth = (CONFLUENCE_USERNAME, CONFLUENCE_API_KEY)
        
        try:
            response = requests.get(url, auth=auth)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Page trouvée: {data.get('title')} (version {data.get('version', {}).get('number', 'inconnue')})")
                success_count += 1
            else:
                print(f"❌ Page non trouvée: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Erreur lors de la récupération: {e}")
    
    print(f"\nRésultat: {success_count}/{len(test_ids)} pages récupérées avec succès")
    return success_count == len(test_ids)

if __name__ == "__main__":
    print("=== TEST DES CONNEXIONS CONFLUENCE ===")
    if test_confluence_connection():
        test_specific_pages()
    print("Tests terminés")