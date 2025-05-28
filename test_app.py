"""
Script de test pour vérifier les importations au niveau racine
"""
import os
import sys
from pathlib import Path

# Configurer l'environnement pour le test
os.environ["USE_RERANKER"] = "False"
os.environ["STREAMLIT_MODE"] = "True"

print("=== TEST D'IMPORTATION AU NIVEAU RACINE ===")
print(f"Répertoire courant: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Ajouter le dossier src au path si nécessaire
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))
    print(f"Ajout de {src_dir} au path")

# Tester l'importation de la configuration
print("\nTest d'importation de config:")
try:
    import config
    print(f"✅ Importation réussie. PERSIST_DIRECTORY = {config.PERSIST_DIRECTORY}")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")

# Tester l'importation de load_db
print("\nTest d'importation de load_db:")
try:
    import load_db
    print("✅ Importation réussie")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")

# Tester l'importation de help_desk
print("\nTest d'importation de help_desk:")
try:
    import help_desk
    print("✅ Importation réussie")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")

# Tester l'importation de streamlit (sans l'exécuter)
print("\nTest d'importation de streamlit.py:")
try:
    import streamlit as st
    print("✅ Module streamlit importé")
    # Comme streamlit.py est notre application, nous ne l'importons pas directement
    print("Note: Le module streamlit.py de l'application n'est pas importé directement")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")

print("\nTests terminés")
