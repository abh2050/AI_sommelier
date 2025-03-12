# Wine Sommelier AI Assistant

An AI-powered sommelier assistant that answers questions about wines, winery information, and food pairings using RAG (Retrieval-Augmented Generation).

## Features

- 🍷 Answer questions about wines, winery details, and food pairings
- 📚 Uses information from your winery's knowledge base
- 🔍 Advanced semantic search to find the most relevant information
- 💬 Conversational interface with streaming responses
- ⚙️ Configurable context size and model options

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- OpenAI API key
- Pinecone API key (free tier works fine)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/abh2050/AI_sommelier.git
cd wine-sommelier-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file by copying `.env.example`:
```bash
cp .env.example .env
```

4. Add your API keys to the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Usage

1. First, index your documents to Pinecone:
```bash
python indexer.py
```

2. Run the Streamlit app:
```bash
streamlit run app1.py
```

3. Open your browser at the URL provided (typically http://localhost:8501)

### Re-indexing

If you add new documents or want to start fresh:
```bash
python indexer.py --recreate
```

## Configuration

- Adjust context size and model in the app sidebar
- Edit chunking parameters in the `.env` file
- Modify prompt templates in `app1.py` for different responses

## Project Structure

- `app1.py`: Main Streamlit application
- `indexer.py`: Document processing and indexing script
- `chunk.py`: Tool to estimate chunks in documents
- `data/markdown_results/`: Directory containing markdown files
