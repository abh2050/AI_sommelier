import os
import streamlit as st
import glob
from typing import List
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import time
from functools import lru_cache

# Set page config at the beginning
st.set_page_config(
    page_title="Wine Sommelier AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables - try both methods to work locally and in cloud
load_dotenv()  # Local development with .env file

# Helper function to get configuration from secrets or environment
def get_config(key, default=None):
    """Get config from Streamlit secrets or environment variables"""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except:
        return os.getenv(key, default)

# Get API keys and configuration
openai_api_key = get_config("OPENAI_API_KEY")
pinecone_api_key = get_config("PINECONE_API_KEY")
markdown_path = get_config("MARKDOWN_PATH", "data/markdown_results")
index_name = get_config("INDEX_NAME", "wine-sommelier")

# Check for required credentials
if not openai_api_key:
    st.error("OpenAI API key is missing. Please add it to your .env file or Streamlit secrets.")
    st.stop()

if not pinecone_api_key:
    st.error("Pinecone API key is missing. Please add it to your .env file or Streamlit secrets.")
    st.stop()

# Initialize clients
try:
    client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Test connection to validate keys
    # Make a simple embedding request to test OpenAI
    client.embeddings.create(
        input=["Test connection"],
        model="text-embedding-ada-002"
    )
    st.sidebar.success("✅ Connected to OpenAI")
except Exception as e:
    st.error(f"Error connecting to OpenAI: {str(e)}")
    st.stop()

# Test Pinecone connection
try:
    # List indexes to test connection
    pc.list_indexes()
    st.sidebar.success("✅ Connected to Pinecone")
except Exception as e:
    st.error(f"Error connecting to Pinecone: {str(e)}")
    st.stop()

# Helper functions
@lru_cache(maxsize=100)
def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding with caching."""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def estimate_tokens(text):
    """Roughly estimate the number of tokens in a text."""
    # A simple estimation: 1 token ≈ 4 characters in English
    return len(text) // 4

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a good break point
        if end < text_len:
            # Try to break at paragraph, sentence, or word boundary
            for break_char in ["\n\n", ".\n", ".", "\n", ". ", " "]:
                breakpoint = text.rfind(break_char, start, end)
                if breakpoint != -1:
                    end = breakpoint + len(break_char.rstrip())
                    break
        
        # Add the chunk
        if end > start:
            chunks.append(text[start:end].strip())
        
        # Move start with overlap
        start = end - chunk_overlap
    
    return chunks

# Status check functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_index_status():
    """Check if all files are properly indexed."""
    try:
        # Count markdown files
        md_files = glob.glob(f"{markdown_path}/**/*.md", recursive=True)
        file_count = len([f for f in md_files if not f.endswith("chunk.py")])
        
        # Check index status
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)
        
        return {
            "file_count": file_count,
            "vector_count": vector_count,
            "status": "Ready" if vector_count > 0 else "Not indexed"
        }
    except Exception as e:
        return {
            "file_count": "Error",
            "vector_count": "Error",
            "status": f"Error: {str(e)}"
        }

def truncate_context(texts, sources, max_tokens=6000):
    """Intelligently truncate context to fit within token limits."""
    if not texts or not sources:
        return [], []
        
    result_texts = []
    result_sources = []
    total_tokens = 0
    
    # First, include shorter contexts (prioritizing concise information)
    items = list(zip(texts, sources))
    items.sort(key=lambda x: len(x[0]))  # Sort by text length
    
    for text, source in items:
        tokens = estimate_tokens(text)
        if total_tokens + tokens <= max_tokens:
            result_texts.append(text)
            result_sources.append(source)
            total_tokens += tokens
        else:
            # If we can fit a partial chunk, do so for the first one only
            if not result_texts:
                # For the first chunk, include a truncated version
                chars_to_keep = (max_tokens * 4) - 100  # Leave some margin
                if chars_to_keep > 0:
                    truncated = text[:chars_to_keep] + "..."
                    result_texts.append(truncated)
                    result_sources.append(source)
            break
    
    return result_texts, result_sources

# Streamlit UI
st.title("Wine Sommelier AI Assistant")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    Welcome to the Wine Sommelier AI Assistant, your expert guide to the world of wines.
    Ask questions about our wines, wineries, regions, or anything related to wine.
    
    The assistant has been trained on our winery's complete knowledge base to provide 
    accurate and helpful information.
    """)
    
    st.header("Example Questions")
    st.write("""
    - What wines do you offer?
    - Tell me about your winery's history
    - What food pairs well with your Cabernet?
    - What's special about your vineyard?
    - How are your wines produced?
    """)

    # Display model selection
    st.header("Settings")
    use_gpt4 = st.toggle("Use GPT-4 (Higher quality but slower)", value=True)
    model_choice = "gpt-4-turbo" if use_gpt4 else "gpt-3.5-turbo-16k"

    st.header("Advanced Settings")
    context_size = st.slider(
        "Context Size", 
        min_value=1, 
        max_value=5, 
        value=3, 
        help="Number of documents to include in context"
    )

    auto_context = st.toggle("Auto-adjust context size", value=True, 
                            help="Automatically reduce context size for complex questions")

    # Show index status
    st.header("Index Status")
    status = check_index_status()
    
    st.metric("Documents", status["file_count"])
    st.metric("Vectors in DB", status["vector_count"])
    st.info(f"Status: {status['status']}")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
        ]
        st.rerun()

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! I'm your Wine Sommelier AI assistant. How can I help you with your wine questions today?"}
    ]

# Display chat history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Connect to existing index
try:
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Error connecting to Pinecone index: {e}")
    st.error("Please run the indexer.py script first to create and populate the index.")
    st.stop()

# Check if index is empty
try:
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) == 0:
        st.warning("The vector database is empty. Please run the indexer.py script to populate it.")
        st.code("python indexer.py", language="bash")
except Exception as e:
    st.error(f"Error checking index statistics: {e}")

# Chat input
query = st.chat_input("Ask me anything about our wines...")
if query:
    # Start timing
    start_time = time.time()
    
    # Add user message to chat
    st.session_state['messages'].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Get query embedding (cached)
                query_embedding = get_embedding(query)
                
                # Search Pinecone
                results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                
                if results.get("matches") and len(results["matches"]) > 0:
                    # Safely extract text and sources with defaults
                    retrieved_texts = [match["metadata"].get("text", "No text available") for match in results["matches"]]
                    retrieved_sources = [match["metadata"].get("source", "Unknown source") for match in results["matches"]]
                    
                    # Use context size from slider
                    limited_texts = retrieved_texts[:context_size]
                    limited_sources = retrieved_sources[:context_size]
                    
                    # Create context string
                    context = "\n\n".join([f"Source: {src}\n{txt}" for src, txt in zip(limited_sources, limited_texts)])

                    # Estimate token count
                    estimated_tokens = estimate_tokens(context) + 600  # Add buffer for system prompt

                    # Apply auto-context adjustment if enabled
                    if auto_context and estimated_tokens > 6000:
                        # Recursively reduce context size until it fits
                        new_context_size = context_size
                        while estimated_tokens > 6000 and new_context_size > 1:
                            new_context_size -= 1
                            limited_texts = retrieved_texts[:new_context_size]
                            limited_sources = retrieved_sources[:new_context_size]
                            context = "\n\n".join([f"Source: {src}\n{txt}" for src, txt in zip(limited_sources, limited_texts)])
                            estimated_tokens = estimate_tokens(context) + 600
                        
                        if new_context_size < context_size:
                            st.info(f"Automatically reduced context from {context_size} to {new_context_size} documents to fit token limits")
                    
                    # Generate response
                    chat_prompt = f"""You are an expert wine sommelier assistant for a winery. 
                    Answer the user's question based on the following context from the winery's knowledge base.
                    If the context doesn't contain relevant information, say "I don't have specific information about that in my knowledge base,
                    but I'd be happy to connect you with our staff for more details."
                    
                    Context:
                    {context}
                    
                    User question: {query}
                    
                    Please be helpful, conversational, and show your expertise about wines.
                    Include specific details from the context when available.
                    """
                    
                    response_placeholder = st.empty()
                    
                    try:
                        # Choose model based on context size
                        selected_model = model_choice
                        if estimated_tokens > 7000 and model_choice in ["gpt-4o", "gpt-4-turbo"]:
                            selected_model = "gpt-3.5-turbo-16k"  # Fallback to larger context model
                            st.warning(f"Context too large for {model_choice}, using {selected_model} instead")
                        
                        # Generate streaming response
                        full_response = ""
                        # Create a streaming response
                        stream = client.chat.completions.create(
                            model=selected_model,
                            messages=[
                                {"role": "system", "content": "You are a wine sommelier expert assistant for a winery. Help customers learn about wines."},
                                {"role": "user", "content": chat_prompt}
                            ],
                            max_tokens=1000,
                            stream=True  # Enable streaming
                        )
                        
                        # Process the stream
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                response_placeholder.markdown(full_response + "▌")
                        
                        # Final response without cursor
                        response_placeholder.markdown(full_response)
                        answer = full_response
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.warning(f"Error: {error_msg[:100]}..." if len(error_msg) > 100 else error_msg)
                        
                        # If it's a token limit error, try with much less context
                        if "context_length_exceeded" in error_msg or "maximum context length" in error_msg:
                            st.info("Reducing context due to length constraints...")
                            
                            # Take just the most relevant result and truncate it
                            if limited_texts and limited_sources:
                                shorter_context = f"Source: {limited_sources[0]}\n{limited_texts[0][:800]}"
                            else:
                                shorter_context = "No context available"
                            
                            shorter_prompt = f"Question about wine: {query}\nContext: {shorter_context}"
                            
                            try:
                                full_response = ""
                                stream = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a concise wine expert."},
                                        {"role": "user", "content": shorter_prompt}
                                    ],
                                    max_tokens=800,
                                    stream=True
                                )
                                
                                for chunk in stream:
                                    if chunk.choices[0].delta.content is not None:
                                        full_response += chunk.choices[0].delta.content
                                        response_placeholder.markdown(full_response + "▌")
                                
                                response_placeholder.markdown(full_response)
                                answer = full_response
                                
                            except Exception as e2:
                                answer = f"I apologize, but I'm having trouble processing your question due to technical limitations. Could you ask a more specific or shorter question?"
                                response_placeholder.markdown(answer)
                    
                    # Add response to chat history
                    st.session_state['messages'].append({"role": "assistant", "content": answer})
                    
                    # Show response timing
                    elapsed_time = time.time() - start_time
                    st.caption(f"Response generated in {elapsed_time:.2f} seconds")
                    
                    # Show sources in an expander - show all retrieved sources
                    with st.expander("View sources"):
                        for i, (text, source) in enumerate(zip(retrieved_texts, retrieved_sources)):
                            st.markdown(f"**Source {i+1}: {source}**")
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                            st.markdown("---")
                else:
                    no_info_msg = "I don't have specific information about that in my knowledge base, but I'd be happy to connect you with our staff for more details."
                    st.write(no_info_msg)
                    st.session_state['messages'].append({"role": "assistant", "content": no_info_msg})
            
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state['messages'].append({"role": "assistant", "content": 
                    f"I'm sorry, I encountered an error while processing your question. Please try again or contact support if the issue persists."})

# Footer
st.markdown("---")
st.caption("Wine Sommelier AI Assistant • Powered by AI • © 2024")
