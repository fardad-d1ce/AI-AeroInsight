![Static Badge](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=%233776AB) ![Static Badge](https://img.shields.io/badge/LangChain-AI%20Agent -%237FC8FF?logo=LangChain&logoColor=%237FC8FF) ![Static Badge](https://img.shields.io/badge/Streamlit-GUI-%23FF4B4B?logo=Streamlit&logoColor=%23FF4B4B)


# 🛩️ AeroInsight: Aviation News Smart Assistant

AeroInsight is a RAG application built with Streamlit and LangChain. It fetches the latest aviation news from RSS feeds, indexes them into a local vector database (ChromaDB), and provides a chat interface to query the news using a free LLM from Hugging Face.


![Alaska Airlines Longest Flights](https://static0.simpleflyingimages.com/wordpress/wordpress/wp-content/uploads/2026/04/25_lh_alaska-airlines-1st-europe-flight-launch-delta-fighting-back_enr_h_site-copy.jpg)
[Simpleflying](https://simpleflying.com/alaska-airlines-longest-flights/): *Alaska Airlines’ inaugural Europe flight marks a bold expansion as competition heats up with Delta.*


## 🌟 Features

- **RSS News Fetching**: Automatically parses aviation news from feeds like Simple Flying.
- **Local ChromaDB**: Uses ChromaDB to store and retrieve news context locally.
- **Customizable AI Architecture**:  
  - **Google Gemini**: Powers high-quality document embeddings (`text-embedding-004`).
  - **Hugging Face**: Powers the LLM inference (`Mistral-7B-Instruct-v0.2`) for a free and flexible experience.
- **Modern LangChain (LCEL)**: Built using the latest LangChain Expression Language patterns.
- **Persistent Chat History**: Maintains chat context and source links throughout the session.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Framework**: LangChain
- **LLM**: Mistral-7B-Instruct-v0.2 (via Hugging Face Endpoint)
- **Embeddings**: Google Generative AI (`text-embedding-004`)
- **Database**: ChromaDB
- **Parsing**: Feedparser & BeautifulSoup

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API Keys:
  - For embeddings, e.g. Google AI Studio API Key
  - For LLM inference, e.g. Hugging Face Hub API Token

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd AeroInsight
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

1. **Fetch News**: Enter an RSS URL in the sidebar (default is [Simple Flying](https://simpleflying.com/category/aviation-news/)) and click "Fetch & Index News".
2. **Chat**: Once indexed, ask any question about the aviation news in the chat box.
3. **View Sources**: Click the "View Sources" expander under any AI response to see the original news links.

## 📋 Future Enhancements

#### 📈 Data Insights
- [ ] Generate smarter business insights.
- [ ] Add a dashboard to visualize the insights.

#### 📎User Interface
- [ ] Customizable RSS feeds.
- [ ] Improve chat history.
- [ ] Integrate with a cloud service for scalable storage and processing.
- [ ] Add a web interface for easier access from any device.
