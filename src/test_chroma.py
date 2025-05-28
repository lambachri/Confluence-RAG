import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Test directory for ChromaDB
TEST_PERSIST_DIRECTORY = './test_db/chroma/'

def test_chroma_setup():
    print("Testing ChromaDB setup...")
    
    # Create some test documents
    documents = [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test", "title": "AI Test"}
        ),
        Document(
            page_content="Python is a programming language used for many applications.",
            metadata={"source": "test", "title": "Python Test"}
        ),
        Document(
            page_content="ChromaDB is a vector database for storing embeddings.",
            metadata={"source": "test", "title": "ChromaDB Test"}
        )
    ]
    
    # Clean up any existing test database
    if os.path.exists(TEST_PERSIST_DIRECTORY):
        print(f"Removing existing test directory: {TEST_PERSIST_DIRECTORY}")
        try:
            shutil.rmtree(TEST_PERSIST_DIRECTORY)
        except Exception as e:
            print(f"Error removing directory: {e}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(TEST_PERSIST_DIRECTORY), exist_ok=True)
    
    try:
        # Create embeddings
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Save to Chroma
        print("Saving documents to ChromaDB...")
        db = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory=TEST_PERSIST_DIRECTORY
        )
        db.persist()
        
        # Test retrieval
        print("Testing document retrieval...")
        results = db.similarity_search("What is artificial intelligence?")
        
        print("\nRetrieval test results:")
        for doc in results:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 50)
        
        print("\nChromaDB is working correctly!")
        return True
        
    except Exception as e:
        print(f"Error during ChromaDB testing: {e}")
        return False

if __name__ == "__main__":
    test_chroma_setup()