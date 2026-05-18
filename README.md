![Static Badge](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=%233776AB) ![Static Badge](https://img.shields.io/badge/LangChain-AI%20Agent%20-%237FC8FF?logo=LangChain&logoColor=%237FC8FF) ![Static Badge](https://img.shields.io/badge/Streamlit-GUI-%23FF4B4B?logo=Streamlit&logoColor=%23FF4B4B)


# 🛩️ AeroInsight: Aviation News Smart Assistant

AeroInsight is a RAG application built with Streamlit and LangChain. It fetches the latest aviation news from RSS feeds, indexes them into a local vector database (ChromaDB), and provides a chat interface to query the news using a free LLM from Hugging Face.

Example:
<p align="center" style="width:50%; margin:auto;">
   <img src="App Appearance.jpg" alt="AeroInsight Chat Example"/>

Welcome to Aviation News Smart Assistant!
<p align="center" style="width:50%; margin:auto;">
  <img src="https://static0.simpleflyingimages.com/wordpress/wordpress/wp-content/uploads/2026/04/25_lh_alaska-airlines-1st-europe-flight-launch-delta-fighting-back_enr_h_site-copy.jpg" alt="Alaska Airlines Longest Flights"/>
  <br/>
  <a href="https://simpleflying.com/alaska-airlines-longest-flights/">Simpleflying</a>: <em>Alaska Airlines’ inaugural Europe flight marks a bold expansion as competition heats up with Delta.</em>
</p>




## 🌟 Features

- **RSS News Fetching**: Automatically parses aviation news from feeds like Simple Flying.
- **Local ChromaDB**: Uses ChromaDB to store and retrieve news context locally.
- **Customizable AI Architecture**:  
  - **Google Gemini**: Powers high-quality document embeddings (`text-embedding-004`).
  - **Hugging Face**: Powers the LLM inference (`Mmeta-llama/Llama-3.1-70B-Instruct`) for a flexible experience.
- **Modern LangChain (LCEL)**: Built using the latest LangChain Expression Language patterns.
- **Persistent Chat History**: Maintains chat context and source *inline* links throughout the session.

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
   git clone https://github.com/fardad-d1ce/AeroInsight.git
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

1. **Fetch News**: Enter multiple RSS URLs in the sidebar and click "Fetch & Index News".
   
   |Default *rss* feeds|||
   |-|-|-|
   |Simple Flying|Airbus Commercial|Airbus Defense|
   |Boeing Commercial Airplanes| Boeing Commercial Services| Boeing Defense Services|
   |DefenseNews|TWZ||
2. **Business Insights**: Click on "Generate Weekly Business Insight", et voilà!
3. **Chat**: Ask any question about the aviation news in the chat box.
4. **Relevant Articles**: Click on "Relevant Articles" expander under any AI response.

## 📋 Enhancements Plan

#### 📈 Data Insights
- [ ] Generate smarter business insights.
- [ ] Add a dashboard to visualize the insights.

#### 📎User Interface
- [ ] ***Scoring layer pipeline*** to filter out low-quality responses in real-time:
   <p align="left" style="width:60%; margin-top:auto;">
      <img src="https://contributor.insightmediagroup.io/wp-content/uploads/2026/04/The-architecture-scaled.png" alt="Scoring Architecture"/>
      <br/>
      <em>Architecture:
      <a href="https://towardsdatascience.com/rag-hallucinates-i-built-a-self-healing-layer-that-fixes-it-in-real-time/">Detect, Score, Heal, Route</a>.</em>
   </p>
- [x] Customizable RSS feeds.
- [x] Improve chat history.
- [ ] Integrate with a cloud service for scalable storage and processing.
- [ ] Add a web interface for easier access from any device.
