import os
import sys
import json
import requests
from pprint import pprint
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import re
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('diagnostic')

sys.path.append('../')
from src.config import (
    CONFLUENCE_SPACE_NAME, CONFLUENCE_USERNAME, CONFLUENCE_API_KEY,
    PERSIST_DIRECTORY
)

# List of specific page IDs to check
SPECIFIC_PAGE_IDS = [
    "3376545896", "3710844929", "3985899883", "3729621164", "3712188435", 
    "3896606774", "4552491244", "4675077035", "3716349965", "3758030922", 
    "3758555439", "2384003194", "2640052240", "2722825033", "2639036983", 
    "2384691207", "2756640773"
]

def check_page_exists(page_id):
    """Verify if a page exists and get its title from Confluence API"""
    url = f"{CONFLUENCE_SPACE_NAME}/rest/api/content/{page_id}?expand=version,title"
    auth = (CONFLUENCE_USERNAME, CONFLUENCE_API_KEY)
    
    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            data = response.json()
            return {
                "id": page_id,
                "exists": True,
                "title": data.get("title", "Unknown title"),
                "version": data.get("version", {}).get("number", "Unknown")
            }
        else:
            return {
                "id": page_id,
                "exists": False,
                "status_code": response.status_code,
                "error": response.text[:100] + "..." if len(response.text) > 100 else response.text
            }
    except Exception as e:
        return {
            "id": page_id,
            "exists": False,
            "error": str(e)
        }

def verify_pages():
    """Verify all specified pages and print a report"""
    print(f"Verifying {len(SPECIFIC_PAGE_IDS)} page IDs...")
    
    results = []
    for page_id in SPECIFIC_PAGE_IDS:
        result = check_page_exists(page_id)
        results.append(result)
        
        if result["exists"]:
            print(f"✓ Page {page_id}: '{result['title']}' (version {result['version']})")
        else:
            print(f"✗ Page {page_id}: NOT FOUND - {result.get('error', 'Unknown error')}")
    
    # Summary
    existing_pages = [r for r in results if r["exists"]]
    missing_pages = [r for r in results if not r["exists"]]
    
    print(f"\nSummary: {len(existing_pages)}/{len(SPECIFIC_PAGE_IDS)} pages exist")
    
    if missing_pages:
        print(f"\nMissing pages ({len(missing_pages)}):")
        for page in missing_pages:
            print(f"- ID {page['id']}")
    
    # Save results to file
    with open("page_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to page_verification_results.json")

def check_vectorstore():
    """Check if vectorstore exists and contains data"""
    print(f"\nChecking vectorstore at {PERSIST_DIRECTORY}...")
    
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"✗ Vectorstore directory doesn't exist!")
        return False
    
    files = os.listdir(PERSIST_DIRECTORY)
    print(f"Files in directory: {files}")
    
    if "chroma.sqlite3" in files:
        print("✓ Found chroma.sqlite3 database")
        try:
            # Optional: Count entries if possible (would need sqlite3 module)
            import sqlite3
            conn = sqlite3.connect(f"{PERSIST_DIRECTORY}/chroma.sqlite3")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            print(f"✓ Database contains {count} embeddings")
            conn.close()
            return True
        except Exception as e:
            print(f"! Could not count embeddings: {e}")
            return True
    else:
        print("✗ No chroma.sqlite3 database found!")
        return False

def sample_content_from_vectorstore():
    """Get sample content from vectorstore to verify document content"""
    # This would require access to the loaded embeddings or the Chroma DB
    # Simple implementation - needs Chroma and embeddings from your actual code
    try:
        # Désactiver l'utilisation de sentence_transformers qui peut causer des erreurs
        os.environ["USE_RERANKER"] = "False"
        
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        
        print("\nSampling content from vectorstore...")
        
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        # Query for content related to our test cases
        for query in ["M3", "Cegid", "procédure", "contrôle", "article"]:
            print(f"\nSearching for documents related to '{query}':")
            docs = db.similarity_search(query, k=3)
            
            for i, doc in enumerate(docs):
                print(f"--- Result {i+1} ---")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Title: {doc.metadata.get('title', 'Unknown')}")
                print(f"Content preview: {doc.page_content[:150]}...")
        
        return True
    except Exception as e:
        print(f"Error sampling content: {e}")
        return False

def verify_pages():
    """Vérifie les pages et leur découpage dans la base vectorielle"""
    print("Vérification des pages et de leur découpage...")
    
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"Le répertoire {PERSIST_DIRECTORY} n'existe pas.")
        return
    
    try:
        # Initialiser l'embedding
        embeddings = OpenAIEmbeddings()
        
        # Charger la base existante
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        
        # Obtenir des statistiques sur les documents
        collection = db._collection
        count = collection.count()
        
        print(f"Nombre total de chunks: {count}")
        
        if count == 0:
            print("Base vide, aucune vérification à effectuer.")
            return
        
        # Récupérer tous les IDs
        all_ids = collection.get()["ids"]
        
        # Prendre un échantillon pour analyse
        sample_size = min(100, count)
        sample_ids = all_ids[:sample_size]
        
        results = collection.get(ids=sample_ids, include=["metadatas", "documents"])
        
        doc_starts = []  # Début des documents
        doc_ends = []    # Fin des documents
        proc_steps = []  # Morceaux de procédures
        
        for i, doc in enumerate(results["documents"]):
            # Vérifier si c'est le début d'un document
            if doc.strip().startswith("1.") or doc.strip().startswith("I.") or doc.strip().startswith("Introduction"):
                doc_starts.append({
                    "id": sample_ids[i],
                    "title": results["metadatas"][i].get("title", "Unknown"),
                    "start": doc[:100]
                })
            
            # Vérifier si c'est la fin d'un document
            if doc.strip().endswith(".") and len(doc) < 500:
                doc_ends.append({
                    "id": sample_ids[i],
                    "title": results["metadatas"][i].get("title", "Unknown"),
                    "end": doc[-100:]
                })
            
            # Vérifier les procédures potentiellement fragmentées
            step_matches = re.findall(r'(\d+)\.\s+(.*?)(?=\n\d+\.|\Z)', doc)
            if step_matches:
                step_numbers = [int(m[0]) for m in step_matches]
                
                # Vérifier s'il manque des étapes
                if len(step_numbers) > 1:
                    expected_steps = list(range(min(step_numbers), max(step_numbers) + 1))
                    missing_steps = [s for s in expected_steps if s not in step_numbers]
                    
                    if missing_steps:
                        proc_steps.append({
                            "id": sample_ids[i],
                            "title": results["metadatas"][i].get("title", "Unknown"),
                            "steps_found": step_numbers,
                            "missing_steps": missing_steps,
                            "fragment": doc[:200] + "..." if len(doc) > 200 else doc
                        })
        
        print("\n--- Analyse de la fragmentation des documents ---")
        print(f"Échantillon analysé: {sample_size} chunks sur {count} total")
        print(f"Documents commençant par une introduction ou numérotation: {len(doc_starts)}")
        print(f"Documents avec une fin propre: {len(doc_ends)}")
        print(f"Procédures potentiellement fragmentées: {len(proc_steps)}")
        
        if proc_steps:
            print("\nExemples de procédures fragmentées:")
            for i, proc in enumerate(proc_steps[:3]):
                print(f"\n{i+1}. Document: {proc['title']}")
                print(f"   Étapes trouvées: {proc['steps_found']}")
                print(f"   Étapes manquantes: {proc['missing_steps']}")
                print(f"   Aperçu: {proc['fragment']}")
            
            if len(proc_steps) > 3:
                print(f"\n... et {len(proc_steps) - 3} autres procédures fragmentées")
        
        # Rapport à enregistrer pour analyse future
        diagnostic_report = {
            "timestamp": datetime.now().isoformat(),
            "total_chunks": count,
            "sample_size": sample_size,
            "doc_starts": len(doc_starts),
            "doc_ends": len(doc_ends),
            "fragmented_procedures": len(proc_steps),
            "procedure_samples": proc_steps[:10] if len(proc_steps) > 0 else []
        }
        
        with open("chunks_diagnostic_report.json", "w") as f:
            json.dump(diagnostic_report, f, indent=2)
        
        print(f"\nRapport diagnostic complet enregistré dans 'chunks_diagnostic_report.json'")
        
    except Exception as e:
        print(f"Erreur lors de la vérification des pages: {e}")

if __name__ == "__main__":
    # S'assurer que le reranker est désactivé pour éviter les erreurs
    os.environ["USE_RERANKER"] = "False"
    
    print("=== Page Verification Diagnostic ===")
    verify_pages()
    
    print("\n=== Vectorstore Diagnostic ===")
    if check_vectorstore():
        sample_content_from_vectorstore()
    
    print("\nDiagnostic complete!")
