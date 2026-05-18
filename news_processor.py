# news_processor.py
# Process RSS feeds and extract news items.

import feedparser
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from datetime import datetime
import time

def parse_date(date_str):
    """Parse various RSS date formats into a standard ISO format (YYYY-MM-DD) and Unix timestamp.
    Args:
        date_str (str): The date string to parse.        
    Returns:
        tuple: A tuple containing the ISO date string (YYYY-MM-DD) and the Unix timestamp.
    """
    if not date_str:
        return "", 0
    try:
        # Let feedparser do the heavy lifting if possible
        # but we'll try manual parsing for robustness
        formats = [
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d"), int(dt.timestamp())
            except ValueError:
                continue
        
        return "", 0
    except Exception:
        return "", 0

def fetch_news(feed_urls):
    """Fetch and parse news from multiple RSS feeds using dictionary access for maximum robustness.
        
    Returns:
        list: A list of dictionaries with keys: 
        title, link, content, published, date_iso, date_timestamp.
    """
    all_news_items = []
    for feed_url in feed_urls:
        news_items = fetch_feed_news(feed_url)
        all_news_items.extend(news_items)
    return all_news_items

def fetch_feed_news(feed_url):
    """Fetch and parse news from an RSS feed using dictionary access for maximum robustness.
        
    Returns:
        list: A list of dictionaries with keys: 
        title, link, content, published, date_iso, date_timestamp.
    """
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
            
            # Parse publication date
            raw_date = entry.get('published', '')
            iso_date, timestamp = parse_date(raw_date)
            
            news_items.append({
                'title': title,
                'link': link,
                'content': clean_text,
                'published': raw_date,
                'date_iso': iso_date,
                'date_timestamp': timestamp
            })
        except Exception:
            continue # Skip problematic entries rather than crashing the whole fetch
        
    return news_items

def process_news_to_documents(news_items):
    """Convert news items into LangChain Document objects and split into chunks with strict filtering.
    
    Args:
        news_items (list): A list of dictionaries, each containing news item data.
        
    Returns:
        list: A list of Document objects, each split into chunks with metadata.
    """
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
            'published': item['published'],
            'date_iso': item.get('date_iso', ''),
            'date_timestamp': item.get('date_timestamp', 0)
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
