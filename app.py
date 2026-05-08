# Imports
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import local modules
from news_processor import fetch_news, process_news_to_documents
from vector_store import add_documents_to_store, get_vector_store

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AeroInsight - Aviation News Chat", layout="wide")

st.title("🛩️ AeroInsight: Aviation News Assistant")
st.markdown("Fetch the latest aviation news and ask questions about it!")

rss_urls = [
            "https://simpleflying.com/feed/"
        ]

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    rss_url = st.text_input("RSS Feed URL", value="https://simpleflying.com/feed/")
    
    if st.button("Fetch & Index News"):
        with st.spinner("Fetching news..."):
            try:
                news_items = fetch_news(rss_url)
                if not news_items:
                    st.error("No news items found. Please check the URL.")
                else:
                    st.info(f"Fetched {len(news_items)} news items. Processing...")
                    try:
                        docs = process_news_to_documents(news_items)
                        st.info(f"Split into {len(docs)} document chunks. Indexing...")
                        add_documents_to_store(docs)
                        st.success(f"Indexed {len(docs)} document chunks to ChromaDB!")
                    except Exception as inner_e:
                        st.error(f"Processing/Indexing Error: {inner_e}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.info("Ensure your HUGGINGFACEHUB_API_TOKEN is set in the environment.")
    # st.info("Ensure your Gemini_API_KEY is set in the environment.")

# Function to display a message with sources if available
def display_message(role, content, sources=None):
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander("Related Sources"):
                for i, src in enumerate(sources):
                    # Handle both Document objects (with .metadata) and simple dicts
                    title = src.metadata.get('title', 'Unknown') if hasattr(src, 'metadata') else src.get('title', 'Unknown')
                    link = src.metadata.get('link', 'N/A') if hasattr(src, 'metadata') else src.get('link', 'N/A')
                    st.write(f"**Source {i+1}: {title}**")
                    st.write(f"Link: {link}")
                    st.write(f"---")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    display_message(message["role"], message["content"], message.get("sources"))

# Chat input
if prompt := st.chat_input("Ask a question about the aviation news:"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)

    # Initialize sources for this specific interaction to avoid leakage
    current_sources = []
    source_docs = []

    # Generate response
    try:

        # --- Google Gemini Model (Commented Out) ---
        # llm_engine = ChatGoogleGenerativeAI(
        #     model="gemini-1.5-flash", 
        #     api_key=os.getenv("Gemini_API_KEY")
        # )          
        #   
        # --- Hugging Face Model (Free) ---
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        provider="featherless-ai"

        # Streaming response example
        from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        
        # Create a callback handler
        callbacks = [StreamingStdOutCallbackHandler()]
        
        llm_engine = HuggingFaceEndpoint(
            repo_id=repo_id,
            provider=provider,
            max_new_tokens=256,
            temperature=0.1,
            callbacks=callbacks,
            streaming=True,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )


        # ChatHuggingFace handles the apply_chat_template() logic internally
        # It transforms a standard ChatPromptTemplate into the model-specific format
        llm_chat = ChatHuggingFace(llm=llm_engine)
        
        # 1. Define the system prompt for the document chain
        system_prompt = (
            "You are a specialized Aviation Assistant."
            "\n\nGUIDELINES:"
            "\n- For general greetings (e.g., 'Hi', 'Hello') or general aviation concepts, be helpful and professional using your general knowledge."
            "\n- For news-specific queries, prioritize the provided context. If the specific news is not in the context, inform the user that it's not in the latest indexed feeds."
            "\n- DO NOT use email-style signatures (e.g., 'Best regards', 'Sincerely')."
            "\n- DO NOT use 'Best regards' and placeholders in the end of your answers."
            "\n- DO NOT use conversational filler like 'I hope this helps' or 'As an AI model'."
            "\n- Use markdown lists and bold text for clarity."
            "\n- Keep responses concise and technical. Again, itemize the points. Avoid long sentences."
            "\n\nContext: {context}"
        )
        
        # 2. Use a general ChatPromptTemplate format. 
        # ChatHuggingFace will automatically wrap this in the model's required tags.
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        # 3. Create the document chain
        combine_docs_chain = create_stuff_documents_chain(llm_chat, prompt_template)
        
        # 4. Create the retrieval chain (replaces RetrievalQA)
        # Using similarity_score_threshold to only return relevant documents
        # This prevents showing "Sources" for general chat like "Hi" or "How are you?"
        vector_store = get_vector_store()
        retriever=vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.7}
            )
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )
        
        with st.spinner("Thinking..."):
            # Invoke the chain
            # Note: 'input' is the key expected by the retrieval chain for the user question
            response = retrieval_chain.invoke({"input": prompt})
            
            # Extract answer and source documents
            answer = response["answer"]
            source_docs = response["context"]
            
            # Convert Document objects to simple dicts for storage
            # This ensures we only keep relevant metadata and clear the full objects
            current_sources = []
            if source_docs:
                current_sources = [
                    {"title": doc.metadata.get("title", "Unknown"), "link": doc.metadata.get("link", "N/A")}
                    for doc in source_docs
                ]
            
            # Display response and sources
            display_message("assistant", answer, current_sources)
            
            # Add assistant message to history with its specific sources
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": current_sources
            })
                
    except Exception as e:
        error_msg = f"Error generating response: {e}"
        st.error(error_msg)
        if "api_key" in str(e).lower():
            st.warning("Please make sure your Gemini_API_KEY is correctly set.")