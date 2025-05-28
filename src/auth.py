import streamlit as st
import hashlib
import os
import json
from datetime import datetime, timedelta

# Fichier pour stocker les informations d'authentification
AUTH_FILE = "auth_config.json"

# Utilisateur et mot de passe par défaut
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "password"

def load_auth_config():
    """Charge la configuration d'authentification ou crée le fichier par défaut"""
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier d'authentification: {e}")
    
    # Créer le fichier avec les valeurs par défaut
    default_config = {
        "users": {
            DEFAULT_USERNAME: hash_password(DEFAULT_PASSWORD)
        },
        "session_duration_hours": 24
    }
    
    try:
        with open(AUTH_FILE, "w") as f:
            json.dump(default_config, f, indent=4)
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier d'authentification: {e}")
    
    return default_config

def hash_password(password):
    """Hache le mot de passe pour le stockage sécurisé"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(username, password):
    """Vérifie si le nom d'utilisateur et le mot de passe sont valides"""
    auth_config = load_auth_config()
    if username in auth_config["users"]:
        stored_hash = auth_config["users"][username]
        return hash_password(password) == stored_hash
    return False

def login_form():
    """Affiche le formulaire de connexion et gère l'authentification"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.auth_username = None
        st.session_state.auth_time = None
    
    if st.session_state.authenticated:
        # Vérifier si la session a expiré
        auth_config = load_auth_config()
        session_duration = auth_config.get("session_duration_hours", 24)
        
        if st.session_state.auth_time:
            expiry_time = st.session_state.auth_time + timedelta(hours=session_duration)
            if datetime.now() > expiry_time:
                st.session_state.authenticated = False
                st.warning("Votre session a expiré. Veuillez vous reconnecter.")
        
        return st.session_state.authenticated
    
    with st.form("login_form"):
        st.subheader("Connexion Administration")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        
        if submit:
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.auth_username = username
                st.session_state.auth_time = datetime.now()
                # Pas besoin de st.rerun() ici car le formulaire gère déjà la soumission
                return True
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")
    
    return False

def change_password_form():
    """Formulaire de changement de mot de passe pour l'utilisateur connecté"""
    if not st.session_state.authenticated:
        return
    
    username = st.session_state.auth_username
    
    with st.form("change_password_form"):
        st.subheader(f"Changer le mot de passe pour {username}")
        current_password = st.text_input("Mot de passe actuel", type="password")
        new_password = st.text_input("Nouveau mot de passe", type="password")
        confirm_password = st.text_input("Confirmer le nouveau mot de passe", type="password")
        submit = st.form_submit_button("Changer le mot de passe")
        
        if submit:
            if not verify_password(username, current_password):
                st.error("Mot de passe actuel incorrect")
                return False
            
            if new_password != confirm_password:
                st.error("Les nouveaux mots de passe ne correspondent pas")
                return False
            
            if len(new_password) < 6:
                st.error("Le nouveau mot de passe doit contenir au moins 6 caractères")
                return False
            
            # Mettre à jour le mot de passe
            auth_config = load_auth_config()
            auth_config["users"][username] = hash_password(new_password)
            
            try:
                with open(AUTH_FILE, "w") as f:
                    json.dump(auth_config, f, indent=4)
                st.success("Mot de passe changé avec succès")
                return True
            except Exception as e:
                st.error(f"Erreur lors de la mise à jour du mot de passe: {e}")
                return False
    
    return False

def logout_user():
    """Déconnecte l'utilisateur"""
    if "authenticated" in st.session_state:
        st.session_state.authenticated = False
        st.session_state.auth_username = None
        st.session_state.auth_time = None
        return True
    return False
