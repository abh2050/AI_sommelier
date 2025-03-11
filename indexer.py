import os
# Disable multiprocessing for OpenAI at the very beginning
os.environ["OPENAI_DISABLE_MULTIPROCESSING"] = "1"

import glob
import time
import logging
import psutil
import gc
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('indexing.log')]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please set it in the .env file.")
client = OpenAI(api_key=openai_api_key)

# Initialize Pinecone 
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Pinecone API key is not set. Please set it in the .env file.")
pc = Pinecone(api_key=pinecone_api_key)

# Default paths and settings - adjusted so that the actual chunks needed is ~356.
MARKDOWN_PATH = "/Users/abhishekshah/Desktop/Winery/rag-gemini-project/data/markdown_results"
INDEX_NAME = "wine-sommelier"
CHUNK_SIZE = 1000          # Chunk size remains at 1000 characters
CHUNK_OVERLAP = 250        # Reduced overlap to match estimated chunk counts
BATCH_SIZE = 30            # Smaller batches to minimize memory during upsert
FILE_LIMIT = None          # Process all files in the folder
RECREATE_INDEX = False     # Don't recreate the index by default

def log_memory_usage(step=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"{step} - Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    gc.collect()  # Force garbage collection after logging memory

def chunk_text_fixed(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into sensible chunks with a fixed algorithm."""
    if not text:
        return []
    
    logger.info(f"Original text length: {len(text)} characters")
    
    # Split text into paragraphs first
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    logger.info(f"Found {len(paragraphs)} paragraphs")
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Include some overlap by keeping the last sentence or two
            last_period = current_chunk.rfind('. ', max(0, len(current_chunk) - chunk_overlap))
            if last_period != -1:
                current_chunk = current_chunk[last_period + 2:]
            else:
                current_chunk = ""
        
        # Add paragraph to current chunk
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
            
        # If the chunk has become too large, split it
        while len(current_chunk) > chunk_size:
            # Find a good breakpoint
            breakpoint = current_chunk.rfind('. ', 0, chunk_size)
            if breakpoint == -1:  # No good breakpoint, use space
                breakpoint = current_chunk.rfind(' ', 0, chunk_size)
            
            if breakpoint == -1:  # No space either, just split at chunk_size
                chunks.append(current_chunk[:chunk_size])
                current_chunk = current_chunk[chunk_size:]
            else:
                chunks.append(current_chunk[:breakpoint + 1])
                current_chunk = current_chunk[breakpoint + 1:]
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Created {len(chunks)} chunks from text")
    
    # Verify chunks don't exceed size limit
    for i, chunk in enumerate(chunks):
        if len(chunk) > chunk_size * 1.5:  # Allow some flexibility
            logger.warning(f"Chunk {i} exceeds size limit: {len(chunk)} chars")
    
    return chunks

def yield_markdown_files(directory, limit=FILE_LIMIT):
    """Yield markdown files one-by-one."""
    logger.info(f"Reading markdown files from {directory}")
    md_files = glob.glob(f"{directory}/**/*.md", recursive=True)
    if limit is not None:
        md_files = md_files[:limit]
    logger.info(f"Found {len(md_files)} markdown files")
    
    for i, file_path in enumerate(md_files):
        try:
            logger.info(f"Reading file {i+1}: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_info = {
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "content": content
                }
                yield file_info
                # Clear references to allow garbage collection
                del content
                gc.collect()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

def get_embedding(text, model="text-embedding-ada-002", retry=3):
    """Generate an embedding with retry logic and memory management."""
    log_memory_usage("Before embedding")
    for attempt in range(retry):
        try:
            response = client.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            log_memory_usage("After embedding")
            return embedding
        except Exception as e:
            if attempt < retry - 1:
                logger.warning(f"Embedding failed (attempt {attempt+1}): {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"Failed to generate embedding after {retry} attempts: {e}")
                raise

def create_index(index_name, dimension=1536):
    """Create a Pinecone index if it doesn't exist."""
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        logger.info(f"Index '{index_name}' already exists")
        return pc.Index(index_name)
    
    logger.info(f"Creating new Pinecone index: {index_name}")
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info("Waiting for index to initialize...")
        for i in range(10):
            time.sleep(5)
            if index_name in pc.list_indexes().names():
                logger.info(f"Index '{index_name}' is now ready")
                return pc.Index(index_name)
            logger.info(f"Waiting... ({(i+1)*5}s)")
        logger.warning("Index creation timed out, but proceeding anyway")
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise

def delete_index(index_name):
    """Delete a Pinecone index if it exists."""
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        logger.info(f"Deleting index: {index_name}")
        pc.delete_index(index_name)
        return True
    return False

def process_file(file_info, index, batch_size=BATCH_SIZE):
    """Process a single file by streaming chunks to Pinecone."""
    filename = file_info["filename"]
    file_path = file_info["path"]
    content = file_info["content"]
    
    logger.info(f"Processing file: {filename}")
    log_memory_usage("Start of file processing")
    
    # Get chunks in a streaming fashion
    text_chunks = chunk_text_fixed(content)
    total_chunks = len(text_chunks)
    logger.info(f"Created {total_chunks} chunks from {filename}")
    
    # Clear content from memory
    del content
    gc.collect()
    
    # Process chunks in micro-batches
    current_batch = []
    total_processed = 0
    
    for chunk_idx, chunk in enumerate(tqdm(text_chunks, desc=f"Processing {filename}")):
        try:
            embedding = get_embedding(chunk)
            
            vector = {
                "id": f"{filename}-{chunk_idx}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": filename,
                    "path": file_path,
                    "chunk_idx": chunk_idx
                }
            }
            
            current_batch.append(vector)
            
            # When batch is full, upsert and clear memory
            if len(current_batch) >= batch_size:
                index.upsert(vectors=current_batch)
                total_processed += len(current_batch)
                logger.info(f"Upserted batch: {total_processed}/{total_chunks} vectors")
                
                # Clear batch and force garbage collection
                current_batch.clear()
                gc.collect()
                time.sleep(0.5)  # Small pause to allow system to recover
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
    
    # Upsert any remaining vectors
    if current_batch:
        try:
            index.upsert(vectors=current_batch)
            total_processed += len(current_batch)
            logger.info(f"Upserted final batch: {total_processed}/{total_chunks} vectors")
        except Exception as e:
            logger.error(f"Error upserting final batch: {e}")
    
    log_memory_usage("After processing file")
    return total_processed

def index_documents(markdown_path=MARKDOWN_PATH, index_name=INDEX_NAME, recreate=RECREATE_INDEX, 
                    batch_size=BATCH_SIZE, file_limit=FILE_LIMIT):
    """Index documents with optimized memory usage."""
    if recreate:
        if delete_index(index_name):
            logger.info(f"Deleted existing index: {index_name}")
            time.sleep(10)
    
    index = create_index(index_name)
    
    if not recreate:
        stats = index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)
        if vector_count > 0:
            logger.info(f"Index already contains {vector_count} vectors. Set RECREATE_INDEX = True to start fresh.")
            return index
    
    total_files = 0
    total_vectors = 0
    
    # Process one file at a time
    for file_info in yield_markdown_files(markdown_path, limit=file_limit):
        total_files += 1
        vectors_processed = process_file(file_info, index, batch_size)
        total_vectors += vectors_processed
        
        # Force garbage collection between files
        gc.collect()
        log_memory_usage(f"After file {total_files}")
    
    logger.info(f"Indexing complete! Processed {total_files} files, {total_vectors} vectors")
    stats = index.describe_index_stats()
    logger.info(f"Final index size: {stats.get('total_vector_count', 0)} vectors")
    
    return index

# Main execution - no argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index markdown files into Pinecone")
    parser.add_argument("--path", default=MARKDOWN_PATH, help="Path to markdown files directory")
    parser.add_argument("--index", default=INDEX_NAME, help="Pinecone index name")
    parser.add_argument("--recreate", action="store_true", help="Recreate index from scratch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for upserting vectors")
    parser.add_argument("--delete-only", action="store_true", help="Only delete the index, don't recreate")
    args = parser.parse_args()
    
    if args.delete_only:
        delete_index(args.index)
        logger.info(f"Deleted index: {args.index}")
    else:
        index_documents(args.path, args.index, args.recreate, args.batch_size)
