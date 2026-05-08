import feedparser
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def fetch_news(feed_url):
    """Fetch and parse news from an RSS feed using dictionary access for maximum robustness."""
    feed = feedparser.parse(feed_url)
    news_items = []
    
    for entry in feed.entries:
        try:
            # Using .get() on the entry dictionary is safer than getattr() for feedparser
            title = entry.get('title', 'No Title')
            link = entry.get('link', '')
            
            content = ""
            # Check 'content' key safely
            if 'content' in entry and isinstance(entry['content'], list) and len(entry['content']) > 0:
                content = entry['content'][0].get('value', '')
            
            # Fallback to summary
            if not content:
                content = entry.get('summary', '')
                
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                clean_text = soup.get_text(separator=' ')
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            else:
                clean_text = "No content available."
            
            news_items.append({
                'title': title,
                'link': link,
                'content': clean_text,
                'published': entry.get('published', '')
            })
        except Exception:
            continue # Skip problematic entries rather than crashing the whole fetch
        
    return news_items

def process_news_to_documents(news_items):
    """Convert news items into LangChain Document objects and split into chunks with strict filtering."""
    documents = []
    for item in news_items:
        # Final safety check on content
        content = item.get('content', '').strip()
        if not content:
            continue
            
        text = f"Title: {item['title']}\n\nContent: {content}"
        metadata = {
            'title': item['title'],
            'link': item['link'],
            'published': item['published']
        }
        documents.append(Document(page_content=text, metadata=metadata))
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    all_chunks = text_splitter.split_documents(documents)
    
    # CRITICAL: Filter chunks AGAIN after splitting. 
    # Sometimes splitting results in a chunk that only contains "Title: ..." with no content.
    filtered_chunks = []
    for chunk in all_chunks:
        # Check if the chunk has actual content beyond the Title header
        if "\n\nContent: " in chunk.page_content:
            body = chunk.page_content.split("\n\nContent: ")[1].strip()
            if body:
                filtered_chunks.append(chunk)
        elif len(chunk.page_content.strip()) > 20: # Fallback for chunks without the header
            filtered_chunks.append(chunk)
            
    return filtered_chunks
