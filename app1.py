import os
import streamlit as st
import glob
from typing import List
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
import time
import hashlib
from functools import lru_cache

# Load environment variables
load_dotenv()

# Initialize OpenAI client with the new API
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please set it in the .env file.")

client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone with the newer API
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Pinecone API key is not set. Please set it in the .env file.")

# Path to markdown files
MARKDOWN_PATH = os.getenv("MARKDOWN_PATH", "data/markdown_results")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Define index name
index_name = "wine-sommelier"

# Add caching for embeddings to speed up repeated queries
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding with caching"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Also cache file reading
@st.cache_data(ttl=86400)  # Cache for 24 hours
def read_markdown_files(directory):
    md_files = glob.glob(f"{directory}/**/*.md", recursive=True)
    md_contents = []
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                md_contents.append({
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "content": content
                })
        except Exception as e:
            st.error(f"Error reading file {file_path}: {e}")
    
    return md_contents

# Function to chunk text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a good break point
        if end < text_len:
            # Try to break at paragraph, sentence, or word boundary
            for break_char in ["\n\n", ".\n", ".\s", "\n", ". ", " "]:
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

# Cache index status check
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_index_status():
    """Check if all files are properly indexed."""
    try:
        # Count markdown files
        md_files = glob.glob(f"{MARKDOWN_PATH}/**/*.md", recursive=True)
        file_count = len(md_files)
        
        # Check index status
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        vector_count = stats["total_vector_count"] if stats else 0
        
        return {
            "file_count": file_count,
            "vector_count": vector_count,
            "status": "Complete" if vector_count > 0 else "Not indexed"
        }
    except Exception as e:
        return {
            "file_count": "Error",
            "vector_count": "Error",
            "status": f"Error: {str(e)}"
        }

# Function to index documents
def index_documents():
    # Check if index exists, create it if it doesn't
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        with st.spinner("Creating new Pinecone index..."):
            # Create a serverless index in AWS
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Give time for index to initialize
            time.sleep(20)  # Wait for the index to be ready
            st.success("Index created successfully!")
    
    # Connect to the index
    index = pc.Index(index_name)
    
    # Check if index is already populated
    stats = index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        st.success(f"Using existing index with {stats['total_vector_count']} vectors.")
        return index
    
    # Read markdown files
    with st.spinner("Reading markdown files..."):
        md_files = read_markdown_files(MARKDOWN_PATH)
        st.info(f"Found {len(md_files)} markdown files.")
    
    # Process and index files
    with st.spinner("Processing and indexing files..."):
        progress_bar = st.progress(0)
        
        total_chunks = 0
        for i, md_file in enumerate(md_files):
            # Chunk the text
            text_chunks = chunk_text(md_file["content"])
            total_chunks += len(text_chunks)
            
            # Get OpenAI embeddings for each chunk
            vectors_to_upsert = []
            for chunk_idx, chunk in enumerate(text_chunks):
                try:
                    # Use the cached embedding function
                    embedding = get_embedding(chunk)
                    
                    vectors_to_upsert.append({
                        "id": f"{md_file['filename']}-{chunk_idx}",
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "source": md_file["filename"],
                            "path": md_file["path"]
                        }
                    })
                except Exception as e:
                    st.warning(f"Error generating embedding for chunk {chunk_idx}: {e}")
                    continue
            
            # Upsert in batches
            batch_size = 100  # Increased batch size for speed
            for j in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[j:j + batch_size]
                index.upsert(vectors=batch)
                time.sleep(0.2)  # Reduced delay for speed
            
            # Update progress
            progress_bar.progress((i + 1) / len(md_files))
        
        st.success(f"Indexing complete! Created {total_chunks} chunks from {len(md_files)} files.")
    
    return index

# Add this function near the top with your other utility functions

def estimate_tokens(text):
    """Roughly estimate the number of tokens in a text."""
    # A simple estimation: 1 token ≈ 4 characters in English
    return len(text) // 4

# Replace the truncate_context function with this improved version:
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
st.title("Wine Sommelier AI Chat")

# Sidebar with about information
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

    # Add a button to force re-indexing
    st.header("Database Management")
    if st.button("Re-index All Files"):
        with st.spinner("Deleting old index..."):
            try:
                # Check if index exists and delete it
                existing_indexes = pc.list_indexes().names()
                if index_name in existing_indexes:
                    pc.delete_index(index_name)
                    st.success(f"Deleted existing index: {index_name}")
                    # Wait for deletion to complete
                    time.sleep(5)
            except Exception as e:
                st.error(f"Error deleting index: {e}")
            
            # Reset the initialization flag to force re-indexing
            if 'index_initialized' in st.session_state:
                del st.session_state['index_initialized']
            
            # Clear caches
            st.cache_data.clear()
            
            st.success("Index reset. New indexing will begin automatically.")
            st.rerun()  # Rerun the app to trigger re-indexing

    # Show index status
    st.header("Index Status")
    status = check_index_status()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files Found", status["file_count"])
    with col2:
        st.metric("Vectors in DB", status["vector_count"])
    
    st.info(f"Status: {status['status']}")
    
    # If the vector count is significantly lower than expected, show a warning
    if isinstance(status["file_count"], int) and isinstance(status["vector_count"], int):
        if status["vector_count"] < status["file_count"]:
            st.warning("The number of vectors seems low compared to the number of files. Consider re-indexing.")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! I'm your Wine Sommelier AI assistant. How can I help you with your wine questions today?"}
    ]

# Add this above your chat input
if len(st.session_state['messages']) > 10:  # Arbitrary threshold
    if st.button("Clear Chat History"):
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
        ]
        st.rerun()

# Display chat history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialize Pinecone index
if 'index_initialized' not in st.session_state:
    with st.spinner("Initializing knowledge base..."):
        index = index_documents()
        st.session_state['index_initialized'] = True
else:
    index = pc.Index(index_name)

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
            # Get query embedding (cached)
            query_embedding = get_embedding(query)
            
            # Search Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            if "matches" in results and results["matches"]:
                # Safely extract text and sources with defaults
                retrieved_texts = [match["metadata"].get("text", "No text available") for match in results["matches"]]
                retrieved_sources = [match["metadata"].get("source", "Unknown source") for match in results["matches"]]
                
                # Use context size from slider
                limited_texts = retrieved_texts[:context_size]
                limited_sources = retrieved_sources[:context_size]
                
                # Create context string
                context = "\n\n".join([f"Source: {src}\n{txt}" for src, txt in zip(limited_sources, limited_texts)])

                # Add this after creating the context string:
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
                    if estimated_tokens > 7000 and model_choice in ["gpt-4", "gpt-4-turbo"]:
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
