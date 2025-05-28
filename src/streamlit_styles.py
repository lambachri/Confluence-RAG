"""
Ce fichier contient les styles CSS personnalisés pour améliorer l'apparence 
de l'interface Streamlit du chatbot, avec une attention particulière au contraste.
"""

def load_css():
    """Renvoie le CSS personnalisé pour l'interface du chatbot avec un contraste amélioré."""
    return """
    <style>
        /* Styles généraux */
        .main-header {
            font-size: 2.5rem;
            color: #1565C0;  /* Bleu plus foncé pour meilleur contraste */
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #212121;  /* Presque noir pour meilleur contraste */
            margin-bottom: 2rem;
            font-style: italic;
        }
        
        /* Personnalisation des bulles de chat avec contraste amélioré */
        .user-bubble {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 15px 15px 15px 5px;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            position: relative;
            color: #0D47A1;  /* Bleu très foncé pour contraste élevé */
            border: 1px solid #BBDEFB;
        }
        .assistant-bubble {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 15px 15px 5px 15px;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            position: relative;
            color: #212121;  /* Gris très foncé, presque noir */
            border-left: 3px solid #1976D2;
        }
        
        /* Animation de saisie avec meilleur contraste */
        .typing-indicator {
            display: inline-block;
            margin-left: 0.4rem;
            color: #1976D2;  /* Bleu plus foncé */
            font-weight: bold;
        }
        .typing-indicator::after {
            content: "";
            animation: typing 1.5s infinite;
            display: inline-block;
        }
        @keyframes typing {
            0% { content: ""; }
            25% { content: "."; }
            50% { content: ".."; }
            75% { content: "..."; }
        }
        
        /* Section sources avec meilleur contraste */
        .sources-section {
            font-size: 0.9rem;
            color: #424242;  /* Gris foncé pour meilleur contraste */
            margin-top: 0.8rem;
            border-top: 1px solid #BDBDBD;  /* Bordure plus visible */
            padding-top: 0.8rem;
            background-color: #FAFAFA;
            border-radius: 8px;
            padding: 10px;
        }
        .sources-section a {
            color: #0D47A1;  /* Liens en bleu foncé */
            text-decoration: underline;
            font-weight: 500;
        }
        
        /* Personnalisation des boutons avec meilleur contraste */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            background-color: #1976D2;  /* Fond bleu plus foncé */
            color: white;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            background-color: #1565C0;  /* Plus foncé au survol */
        }
        
        /* Amélioration du champ de saisie */
        .stTextInput>div>div>input {
            padding: 0.8rem;
            font-size: 1.1rem;
            border-radius: 12px;
            border: 2px solid #BDBDBD;  /* Bordure plus visible */
        }
        
        /* Sidebar styling avec meilleur contraste */
        .sidebar-title {
            color: #0D47A1;  /* Bleu plus foncé */
            font-size: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        /* Badges pour les exemples avec meilleur contraste */
        .example-badge {
            display: inline-block;
            padding: 4px 8px;
            background-color: #1976D2;  /* Fond bleu plus foncé */
            color: white;  /* Texte blanc pour contraste */
            border-radius: 4px;
            margin-right: 4px;
            margin-bottom: 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        /* Mise en évidence des informations techniques avec meilleur contraste */
        .tech-highlight {
            background-color: #FFF59D;  /* Jaune légèrement plus foncé */
            padding: 0.3em 0.5em;
            border-radius: 4px;
            font-family: monospace;
            color: #212121;  /* Texte foncé */
            border: 1px solid #FBC02D;  /* Bordure jaune plus foncée */
        }
        
        /* Style spécial pour les noms des commandes et paramètres techniques */
        .command-name {
            font-family: monospace;
            background-color: #E8EAF6;
            padding: 0.1em 0.4em;
            border-radius: 3px;
            color: #303F9F;
            font-weight: 600;
            border: 1px solid #C5CAE9;
        }
        
        /* Amélioration des titres dans les bulles */
        .assistant-bubble strong, .user-bubble strong {
            font-size: 1.1rem;
            display: block;
            margin-bottom: 0.5rem;
        }
    </style>
    """
