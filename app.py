# Imports
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import for GUI streaming
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Import local modules
from news_processor import fetch_news, process_news_to_documents
from vector_store import add_documents_to_store, get_vector_store, get_recent_headlines

# Load environment variables
load_dotenv()

# Functions: RAG initialization, Message displaying
@st.cache_resource
def get_llm_chat():
    """Initialize and cache the full Chat model."""    
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    repo_id = "meta-llama/Llama-3.1-70B-Instruct"
    # provider = "featherless-ai"
    provider = "scaleway"
    
    llm_engine = HuggingFaceEndpoint(
        repo_id=repo_id,
        provider=provider,
        task="text-generation", # Fixes the "conversational task not supported" error
        max_new_tokens=1024,
        temperature=0.0,
        streaming=True,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    # Return the Chat wrapper directly
    return ChatHuggingFace(llm=llm_engine)

def get_retrieval_chain(llm_chat, prompt_template, 
                        search_type="similarity", 
                        search_kwargs={"k": 5, "score_threshold": 0.4}):
    """Create retrieval chain to invoke()."""
    vector_store = get_vector_store()
    
    # Define how each document is formatted within the {context} variable
    # This makes the title and link very clear to the LLM for citation purposes
    from langchain_core.prompts import PromptTemplate
    document_prompt = PromptTemplate.from_template(
        "Source: {title}\nLink: {link}\nContent: {page_content}"
    )
    
    combine_docs_chain = create_stuff_documents_chain(
        llm_chat, 
        prompt_template,
        document_prompt=document_prompt
    )
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

# Function to display a message with sources if available
def display_message(role, content, sources=None):
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander("Relevant Articles"):
                for i, src in enumerate(sources):
                    # Handle both Document objects (with .metadata) and simple dicts
                    title = src.metadata.get('title', 'Unknown') if hasattr(src, 'metadata') else src.get('title', 'Unknown')
                    link = src.metadata.get('link', 'N/A') if hasattr(src, 'metadata') else src.get('link', 'N/A')
                    st.write(f"**Source {i+1}: {title}**")
                    st.write(f"Link: {link}")
                    st.write(f"---")

### Parameters ###
rss_urls = [
            "https://simpleflying.com/feed/",
            "https://www.defensenews.com/arc/outboundfeeds/rss/category/air/",
            "https://www.twz.com/feed",
            "https://www.airbus.com/en/rss-all-feeds/15571?tid=15571&fid=29711", # Airbus Commercial
            "https://www.airbus.com/en/rss-all-feeds/15576?tid=15576&fid=29721", # Airbus Defense
            "https://boeing.mediaroom.com/news-releases-statements?pagetemplate=rss&category=786", # Boeing Commercial Airplanes
            "https://boeing.mediaroom.com/news-releases-statements?pagetemplate=rss&category=791", # Boeing Commercial Services
            "https://boeing.mediaroom.com/news-releases-statements?pagetemplate=rss&category=795" # Boeing Defense
        ]

######################
### Main App Logic ###
######################

# Initialize LLM
llm_chat = get_llm_chat()

# GUI Page configuration
st.set_page_config(page_title="AeroInsight - Aviation News Chat", layout="wide")

st.title("🛩️ AeroInsight: Aviation News Assistant")
st.markdown("Fetch the latest aviation news and ask questions about it!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    user_rss_urls = st.text_input("RSS Feed URLs (comma-separated)", value=",".join(rss_urls))
    user_rss_urls = [url.strip() for url in user_rss_urls.split(',')]
    # user_rss_urls = st.multiselect("RSS Feed URLs", options=rss_urls)
    
    if st.button("Fetch & Index News"):
        with st.spinner("Fetching news..."):
            try:
                # news_items = fetch_news(rss_urls)
                news_items = fetch_news(user_rss_urls)
                if not news_items:
                    st.error("No news items found. Please check the URLs.")
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
    st.header("Insights")
    if st.button("Generate Weekly Business Insight"):
        with st.spinner("Analyzing recent aviation news..."):
            try:
                # 1. Define the insight specialized prompt
                insight_system_prompt = (
                    "You are an Aviation Business Analyst. Categorize the provided news into:"
                    "\n1. **Network Expansion** (routes, codeshares)"
                    "\n2. **Fleet Strategy** (orders, deliveries)"
                    "\n3. **Financial Market** (earnings, mergers, strikes)"
                    "\n\nRules:"
                    "\n- Use bullet points."
                    "\n- For every item, Insert TWO new lines, then include the [Source Title](Direct Link) format inline."
                    "The link must be on its own line below the text."
                    "\n- Be extremely concise."
                    "\n- Avoid email template formatting, e.g. 'Best regards,'"
                    "\n\nContext: {context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", insight_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                # 2. Get a business insight chain
                # We use a higher k to get more context for the insight
                insight_chain = get_retrieval_chain(llm_chat, prompt_template, 
                                search_type="similarity_score_threshold", 
                                search_kwargs={"k": 10, "score_threshold": 0.3})
                
                # 3. Invoke the chain with a broad query to capture business news
                response = insight_chain.invoke({
                    "input": "What are the latest developments in airline networks, fleets, and financial results?",
                    "chat_history": [] # Insight generation usually doesn't need previous chat context
                })
                
                insight_text = response["answer"]
                source_docs = response["context"]
                
                # 4. Process current sources for storage and deduplicate by link
                current_sources = []
                seen_links = set()
                if source_docs:
                    for doc in source_docs:
                        link = doc.metadata.get("link", "N/A")
                        if link not in seen_links:
                            current_sources.append({
                                "title": doc.metadata.get("title", "Unknown"), 
                                "link": link
                            })
                            seen_links.add(link)
                
                full_content = f"### 📊 Weekly Business Insight\n\n{insight_text}"
                
                # Use the unified display and history logic
                display_message("assistant", full_content, current_sources)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_content,
                    "sources": current_sources
                })
                st.rerun()
            except Exception as e:
                st.error(f"Insight Error: {e}")

    st.divider()
    st.info("Ensure your API_TOKEN is set in the environment.")

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
        # 1. Define the system prompt for the document chain
        system_prompt = (
            "You are an Aviation News Assistant. Your job is to extract facts from the 'Context' and answer user questions using chat history."
            "\n\n--- MANDATORY FORMATTING RULES ---"
            "\n1. BULLET POINTS ONLY: Every statement must start with a '-' bullet point."
            "\n2. LINK ON NEW LINE: Insert TWO new lines before every link. The link must be on its own line below the text."
            "\n3. LINK FORMAT: Use [Source Title](Direct Link). NEVER post a raw URL."
            "\n4. NO INTRO: Start immediately with the '-' bullet point."
            "\n\n--- ANTI-HALLUCINATION RULES (CRITICAL) ---"
            "\n- LINKS ONLY FROM CONTEXT: You are FORBIDDEN from creating or guessing links. You MUST ONLY use the links provided in the 'Context' block below."
            "\n- NO LINKS FOR HISTORY: If you are answering based on chat history, DO NOT include any link."
            "\n- NO NEWS FROM TRAINING: If a news item is not in the 'Context', it does not exist. Do not use your internal knowledge for news."
            "\n\n--- CONTENT RULES ---"
            "\n- For news: Extract facts from 'Context' and append the specific link from that context on a new line."
            "\n- For history/general: Answer briefly using history in '-' bullet point format, with NO link."
            "\n- If no info is found: Say '- I don't have news on that topic in my database.' and STOP."
            "\n\nContext: {context}"
        )
        
        # 2. Use a general ChatPromptTemplate format. 
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # 3. Use the encapsulated function to get the chain
        retrieval_chain = get_retrieval_chain(llm_chat, prompt_template, 
                                search_type="similarity_score_threshold", 
                                search_kwargs={"k": 5, "score_threshold": 0.4})
        
        with st.spinner("Thinking..."):
            # Convert session state to LangChain messages for history
            chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            # 1. Initial Attempt with Threshold
            response = retrieval_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            answer = response["answer"]
            source_docs = response["context"]
            
            # 2. Vague Prompt Fallback
            # If no docs meet the 0.4 threshold, but user is asking for general news
            vague_keywords = ["news", "latest", "updates", "what's happening", "bring up", "show me"]
            if not source_docs and any(kw in prompt.lower() for kw in vague_keywords):
                # Perform a broad search for the most recent content (no threshold)
                fallback_chain = get_retrieval_chain(llm_chat, prompt_template,
                                search_type="similarity",
                                search_kwargs={"k":5})
                
                # Modify the prompt to include historical context
                vague_prompt = f"Give me the latest news about {prompt}"
                response = fallback_chain.invoke({
                    "input": vague_prompt,
                    "chat_history": chat_history
                })
                answer = response["answer"]
                source_docs = response["context"]

            # 3. Process current sources for storage and deduplicate by link
            current_sources = []
            seen_links = set()
            if source_docs:
                for doc in source_docs:
                    link = doc.metadata.get("link", "N/A")
                    if link not in seen_links:
                        current_sources.append({
                            "title": doc.metadata.get("title", "Unknown"), 
                            "link": link
                        })
                        seen_links.add(link)

            # Display response and sources using the unified function
            display_message("assistant", answer, current_sources)
            
            # Add assistant message to history
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