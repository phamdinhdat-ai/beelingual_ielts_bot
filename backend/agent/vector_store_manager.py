# app/vector_store_manager.py
import os
import shutil
import chromadb
from chromadb.config import Settings
from loguru import logger
from typing import List, Optional
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings # Or OpenAI if that's preferred
from langchain.docstore.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.chat_models.ollama import ChatOllama # For retriever tool LLM
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vector_stores")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest") # Match agent script
# Use Ollama embeddings if specified, otherwise default (e.g., SentenceTransformers via Chroma default or specify OpenAI)
# # NOTE: Ensure Ollama server is running if using OllamaEmbeddings
# try:
#     # Check if Ollama is intended (e.g., based on LLM model name)
#     # This logic might need adjustment based on how you want to select embeddings
#     if "mistral" in os.getenv("LLM_MODEL", "") or "ollama" in os.getenv("LLM_MODEL", ""):
#          logger.info(f"Using Ollama Embeddings: {EMBEDDING_MODEL_NAME}")
#          # Ensure the Ollama model name for embeddings matches one available in Ollama
#          embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#          # Test connection (optional)
#          # embeddings.embed_query("test")
#     else:
#          # Default or specify another provider like OpenAI
#          # logger.info("Using default SentenceTransformer embeddings (via Chroma).")
#          # embeddings = None # Let Chroma handle default
#          # OR use OpenAI
#          logger.info("Using OpenAI Embeddings.") # Assuming OpenAI is the alternative
#          embeddings = OpenAIEmbeddings()

# except Exception as e:
#     logger.error(f"Failed to initialize embeddings model '{EMBEDDING_MODEL_NAME}': {e}. Check model availability/Ollama server.", exc_info=True)
#     # Fallback or raise error
#     logger.warning("Falling back to default embeddings due to initialization error.")
#     embeddings = None # Let Chroma handle default
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME) # Default to Ollama for now

# LLM for potential use within tools (keep consistent with agent)
llm_for_tool = ChatOllama(model=os.getenv("LLM_MODEL", "deepseek-r1:1.5b"), temperature=0.1)

# ChromaDB client (persistent)
persistent_client = chromadb.PersistentClient(path=VECTOR_STORE_BASE_DIR)
logger.info(f"ChromaDB client initialized. Persistence directory: {VECTOR_STORE_BASE_DIR}")

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- Helper Functions ---

def get_collection_name(user_id: Optional[int] = None, guest_session_id: Optional[str] = None) -> str:
    """Generates a unique and valid ChromaDB collection name."""
    if user_id is not None:
        return f"user_{user_id}"
    elif guest_session_id is not None:
        # Chroma names need to be valid: 3-63 chars, start/end alphanumeric, only _. and - allowed
        # Use replace for session IDs which might have invalid chars (like hyphens in UUIDs)
        clean_session_id = guest_session_id.replace("-", "_")
        # Truncate if too long ( unlikely with UUIDs, but good practice)
        collection_name = f"guest_{clean_session_id}"
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
        # Ensure start/end alphanumeric (should be fine with guest_ prefix)
        return collection_name
    else:
        raise ValueError("Either user_id or guest_session_id must be provided")

def process_file_content(content: bytes, filename: str) -> List[Document]:
    """Processes file content into LangChain Documents using text splitting."""
    try:
        text = content.decode('utf-8', errors='ignore')
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t, metadata={"source": filename}) for t in texts]
        logger.info(f"Processed file '{filename}' into {len(docs)} chunks.")
        return docs
    except Exception as e:
        logger.error(f"Error processing file content for {filename}: {e}", exc_info=True)
        return []

def add_documents(docs: List[Document], collection_name: str):
    """Adds LangChain Documents to a specified ChromaDB collection."""
    if not docs:
        logger.warning(f"No documents provided for collection '{collection_name}'.")
        return

    try:
        logger.info(f"Attempting to add {len(docs)} document chunks to collection '{collection_name}'...")
        # Use LangChain's Chroma wrapper for easier integration
        vector_store = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings # Pass the initialized embeddings
        )
        vector_store.add_documents(docs)
        logger.info(f"Successfully added documents to collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to add documents to collection '{collection_name}': {e}", exc_info=True)
        # Decide on error handling: raise, log, return status?

def get_vector_retriever(collection_name: str, k: int = 4) -> Optional[BaseRetriever]:
     """Gets a retriever for a specific ChromaDB collection."""
     try:
        # Check if collection exists before creating retriever
        persistent_client.get_collection(collection_name) # Throws exception if not found

        vector_store = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        logger.info(f"Successfully created retriever for collection '{collection_name}'.")
        return retriever
     except Exception as e:
        # Catch specific ChromaDB collection not found error if possible,
        # otherwise, general exception.
        if "does not exist" in str(e).lower():
             logger.warning(f"Collection '{collection_name}' not found or empty. Cannot create retriever.")
        else:
             logger.error(f"Failed to create retriever for collection '{collection_name}': {e}", exc_info=True)
        return None

# --- Advanced Retriever (Optional - Ensemble with BM25 for keyword search) ---
def get_ensemble_retriever(collection_name: str, k_vector: int = 3, k_bm25: int = 3) -> Optional[BaseRetriever]:
    """Gets an ensemble retriever combining vector search and BM25 keyword search."""
    try:
        vector_retriever = get_vector_retriever(collection_name, k=k_vector)
        if not vector_retriever:
            return None # Vector store likely doesn't exist

        # Need the actual documents from the store to initialize BM25
        vector_store = Chroma(client=persistent_client, collection_name=collection_name, embedding_function=embeddings)
        all_docs = vector_store.get() # Fetch all docs - potentially inefficient for large stores!
        if not all_docs or not all_docs.get('documents'):
             logger.warning(f"No documents found in collection '{collection_name}' for BM25 initialization.")
             return vector_retriever # Fallback to just vector retriever

        # Reconstruct LangChain Documents for BM25Retriever
        langchain_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(all_docs['documents'], all_docs['metadatas'] or [])
        ]

        bm25_retriever = BM25Retriever.from_documents(langchain_docs, k=k_bm25)
        bm25_retriever.k = k_bm25

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6] # Adjust weights as needed (BM25, Vector)
        )
        logger.info(f"Successfully created ensemble retriever for collection '{collection_name}'.")
        return ensemble_retriever

    except Exception as e:
        logger.error(f"Failed to create ensemble retriever for '{collection_name}': {e}", exc_info=True)
        return get_vector_retriever(collection_name) # Fallback


# --- Retriever Tool Creation ---
def create_retriever_tool(collection_name: str, use_ensemble: bool = True) -> Optional[Tool]:
    """Creates a LangChain Tool wrapping the retriever for the specified collection."""
    if use_ensemble:
         retriever = get_ensemble_retriever(collection_name)
    else:
         retriever = get_vector_retriever(collection_name)

    if not retriever:
        # Return a dummy tool indicating no documents are available
        def dummy_func(query:str): return f"No documents found or index available for {collection_name}."
        tool_description = f"Placeholder tool for {collection_name} - no documents available or index error."
        logger.warning(f"Returning dummy tool for collection '{collection_name}'.")
        return Tool(name=f"{collection_name}_retriever", func=dummy_func, description=tool_description)

    # --- Optional: Contextual Compression ---
    # Improves relevance by filtering/compressing retrieved docs using the LLM
    # embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    # pipeline_compressor = DocumentCompressorPipeline(transformers=[embeddings_filter])
    # compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
    # Use compression_retriever below if enabled

    tool_description = f"Searches and retrieves relevant information from documents associated with {collection_name}. Use this for questions about content within those specific documents."
    tool_name = f"{collection_name}_retriever"

    # Create the tool - the function should take a string query
    # The retriever's get_relevant_documents method takes the query string.
    # We need to format the output for the agent.
    async def _arun_retriever(query: str):
        logger.info(f"Running retriever tool '{tool_name}' for query: '{query}'")
        try:
            # Use the chosen retriever (base or compressed)
            # docs = compression_retriever.get_relevant_documents(query) # If using compression
            docs = await retriever.aget_relevant_documents(query) # Use async version
            if not docs:
                return f"No relevant information found in documents for '{query}' in {collection_name}."
            # Format the results
            context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents for query.")
            return f"Retrieved context from documents in {collection_name}:\n{context}"
        except Exception as e:
            logger.error(f"Error during retriever tool execution for {tool_name}: {e}", exc_info=True)
            return f"Error retrieving documents for {collection_name}."

    # The Tool needs a synchronous function for CrewAI's default execution
    # We can wrap the async function if CrewAI doesn't handle async tools well directly
    # Or define a sync version alongside
    def _run_retriever(query: str):
        logger.info(f"Running retriever tool '{tool_name}' (sync) for query: '{query}'")
        try:
            docs = retriever.get_relevant_documents(query) # Sync version
            if not docs:
                return f"No relevant information found in documents for '{query}' in {collection_name}."
            context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents for query (sync).")
            return f"Retrieved context from documents in {collection_name}:\n{context}"
        except Exception as e:
            logger.error(f"Error during sync retriever tool execution for {tool_name}: {e}", exc_info=True)
            return f"Error retrieving documents for {collection_name}."

    return Tool(
        name=tool_name,
        func=_run_retriever, # Pass the sync version for compatibility
        # coroutine=_arun_retriever, # Provide async version if framework supports it
        description=tool_description,
    )


def cleanup_guest_store(guest_session_id: str):
    """Deletes the ChromaDB collection associated with a guest session."""
    collection_name = get_collection_name(guest_session_id=guest_session_id)
    try:
        logger.warning(f"Attempting to delete ChromaDB collection: '{collection_name}' for guest session cleanup.")
        persistent_client.delete_collection(collection_name)
        logger.info(f"Successfully deleted collection '{collection_name}'.")
        # Note: ChromaDB might not physically delete files immediately depending on config/version.
        # Physical directory cleanup might be needed separately if issues persist.
        # store_path = os.path.join(VECTOR_STORE_BASE_DIR, collection_name) # Path might differ slightly
        # if os.path.exists(store_path): shutil.rmtree(store_path)
    except Exception as e:
        # It's common for the collection to not exist if no docs were uploaded
        if "does not exist" in str(e).lower():
             logger.info(f"Collection '{collection_name}' did not exist, no deletion needed.")
        else:
             logger.error(f"Failed to delete collection '{collection_name}' during cleanup: {e}", exc_info=True)