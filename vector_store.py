import os
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = "chroma_db"

# Global cache for the vector store instance
_vector_store_cache = None

def get_vector_store():
    """Initialize or load the ChromaDB vector store with caching."""
    global _vector_store_cache
    
    if _vector_store_cache is not None:
        return _vector_store_cache
        
    api_key = os.getenv("Gemini_API_KEY")
    # Using the dedicated embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=api_key
            )
    
    # Ensure the directory exists
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)
        
    _vector_store_cache = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="aviation_news"
    )
    return _vector_store_cache

def add_documents_to_store(documents):
    """Add documents to the vector store one by one for maximum stability.
    
    Args:
        documents (list): A list of Document objects to be added to the vector store.
        
    Returns:
        Chroma: The updated vector store after adding documents.
    """
    vector_store = get_vector_store()
    
    for doc in documents:
        try:
            vector_store.add_documents([doc])
        except Exception:
            continue
            
    return vector_store

# def query_vector_store(query, k=4):
#     """Query the vector store for similar documents.
    
#     Args:
#         query (str): The query string to search for.
#         k (int, optional): The number of similar documents to return. Defaults to 4.
        
#     Returns:
#         list: A list of Document objects that are most similar to the query.
#     """
#     vector_store = get_vector_store()
#     return vector_store.similarity_search(query, k=k)

def get_recent_headlines(days=7):
    """Retrieve all document titles/headlines from the last N days using numeric timestamps.
    
    Args:
        days (int, optional): The number of days back to retrieve headlines. Defaults to 7.
        
    Returns:
        list: A sorted list of unique document titles/headlines from the last N days.
    """
    from datetime import datetime, timedelta
    import time
    
    vector_store = get_vector_store()
    
    # Calculate threshold as a Unix timestamp (integer)
    # Current time minus (days * seconds in a day)
    threshold_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    
    # Chroma filter syntax using integer comparison
    # This is more robust than string-based ISO comparison
    results = vector_store.get(
        where={"date_timestamp": {"$gte": threshold_timestamp}}
    )
    
    # Extract unique headlines from the documents
    metadatas = results.get('metadatas', [])
    headlines = sorted(list(set([m.get('title') for m in metadatas if m.get('title')])))
    
    return headlines
