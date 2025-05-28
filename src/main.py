# Demo
if __name__ == '__main__':
    import os
    import sys
    import logging
    from pathlib import Path
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('main')
    
    # Ajuster le chemin d'importation pour être compatible avec tous les environnements
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    root_dir = current_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    
    # Importation flexible de HelpDesk
    try:
        from help_desk import HelpDesk
    except ImportError:
        try:
            from src.help_desk import HelpDesk
        except ImportError:
            from .help_desk import HelpDesk
    
    # Importation flexible de config
    try:
        from config import PERSIST_DIRECTORY
    except ImportError:
        try:
            from src.config import PERSIST_DIRECTORY
        except ImportError:
            from .config import PERSIST_DIRECTORY
    
    # Importation flexible de load_db
    try:
        from load_db import DataLoader
    except ImportError:
        try:
            from src.load_db import DataLoader
        except ImportError:
            from .load_db import DataLoader
    
    # Check if DB exists
    db_exists = os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY)
    print(f"Database exists: {db_exists}")
    
    # Si la base de données n'existe pas ou si on veut la recréer, 
    # l'exemple ci-dessous montre comment charger directement des pages spécifiques
    if not db_exists or "--reset" in sys.argv:
        print("Initialisation de la base avec des pages spécifiques")
        
        # Liste des IDs spécifiques à charger
        page_ids = [
            "3376545896", "3710844929", "3985899883", "3729621164", "3712188435", 
            "3896606774", "4552491244", "4675077035", "3716349965", "3758030922", 
            "3758555439", "2384003194", "2640052240", "2722825033", "2639036983", 
            "2384691207", "2756640773","4444487681","4766369934","4055826672"
        ]
        
        # Initialiser le DataLoader avec les IDs
        loader = DataLoader(page_ids=page_ids)
        
        # Supprimer l'ancienne base si demandé
        if "--reset" in sys.argv and os.path.exists(PERSIST_DIRECTORY):
            import shutil
            print(f"Suppression de la base existante: {PERSIST_DIRECTORY}")
            shutil.rmtree(PERSIST_DIRECTORY)
        
        # Créer la base avec les embeddings
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        db = loader.set_db(embeddings)
        
        try:
            doc_count = db._collection.count()
            print(f"Base créée avec succès. Nombre de documents: {doc_count}")
        except Exception as e:
            print(f"Erreur lors de la vérification de la base: {e}")
    
    # Use existing DB
    try:
        # Définir si on utilise le reranker ou non via variable d'environnement
        use_reranker_env = os.environ.get("USE_RERANKER", "True").lower() == "true"
        
        # Utiliser la valeur de USE_RERANKER plutôt que de forcer à False
        os.environ["USE_RERANKER"] = os.environ.get("USE_RERANKER", "True")
        
        # Initialiser HelpDesk avec le reranker
        model = HelpDesk(new_db=False, use_reranker=use_reranker_env)
        
        # Check document count
        try:
            doc_count = model.db._collection.count()
            print(f"Documents in database: {doc_count}")
        except Exception as e:
            print(f"Error counting documents: {e}")
        
        # Test a question
        prompt = 'Comment faire ma photo de profil Octo ?'
        print(f"\nQuestion: {prompt}")
        result, sources = model.retrieval_qa_inference(prompt)
        print("\nRéponse:")
        print(result)
        print("\nSources:")
        print(sources)
        
        # Try another question
        prompt2 = 'Comment configurer un job dans OpCon?'
        print(f"\nQuestion: {prompt2}")
        result2, sources2 = model.retrieval_qa_inference(prompt2)
        print("\nRéponse:")
        print(result2)
        print("\nSources:")
        print(sources2)
    except Exception as e:
        logger.error(f"Error initializing model: {e}", exc_info=True)
        print(f"⚠️ Error initializing model: {e}")
