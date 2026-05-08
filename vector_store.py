import os
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = "chroma_db"

def get_vector_store():
    """Initialize or load the ChromaDB vector store."""
    api_key = os.getenv("Gemini_API_KEY")
    # Using the dedicated embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=api_key
         )
    
    # Ensure the directory exists
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)
        
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="aviation_news"
    )
    return vector_store

def add_documents_to_store(documents):
    """Add documents to the vector store one by one for maximum stability."""
    vector_store = get_vector_store()
    
    for doc in documents:
        try:
            vector_store.add_documents([doc])
        except Exception:
            continue
            
    return vector_store

def query_vector_store(query, k=4):
    """Query the vector store for similar documents."""
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)
