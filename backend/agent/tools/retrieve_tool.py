# ./retrieval/rag_tools.py

from crewai_tools import BaseTool
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction # Try importing the base type if available
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import logging
from pydantic.v1 import BaseModel as V1BaseModel
# Logging setup remains the same
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Input Schemas (RetrieveInput, IngestInput) remain the same ---
class RetrieveInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant information")
    collection_name: str = Field(..., description="Name of the collection to search in")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    use_mmr: bool = Field(default=True, description="Whether to use Maximum Marginal Relevance for diverse results")
    mmr_diversity: float = Field(default=0.3, description="Diversity factor for MMR (0.0 to 1.0, higher means more diverse)")

class IngestInput(BaseModel):
    documents: List[str] = Field(..., description="List of documents (text content) to ingest")
    collection_name: str = Field(..., description="Name of the collection to store documents")
    metadatas: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional list of metadata dictionaries, one per document")
    ids: Optional[List[str]] = Field(default=None, description="Optional list of unique IDs for the documents")


class RetrieveTool(BaseTool):
    """
    RAG tool for document retrieval from ChromaDB with optional MMR reranking.
    Uses PrivateAttrs for internal state management to avoid Pydantic validation issues.
    """
    name: str = "ChromaDB Retriever Tool"
    description: str = ("Retrieves relevant information from a specified ChromaDB collection. "
                        "Can use standard similarity search or Maximum Marginal Relevance (MMR) for diversity.")
    args_schema: type[V1BaseModel] = RetrieveInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    _client: Optional[chromadb.ClientAPI] = PrivateAttr(default=None)
    _model_embed: Optional[SentenceTransformer] = PrivateAttr(default=None)
    
    _embedding_function: Optional[Any] = PrivateAttr(default=None) # Using Any as safer fallback

    # persist_dir can remain public if it's configuration
    persist_dir: str = Field(default="./chroma_db_store", description="Directory for persistent ChromaDB storage")

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2",
                 persist_dir: str = "./chroma_db_store", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initializing RetrieveTool with model '{embedding_model_name}' and persist_dir '{persist_dir}'")
        self.persist_dir = persist_dir
        
        self._setup_retriever(embedding_model_name, persist_dir)

    def _setup_retriever(self, embedding_model_name: str, persist_dir: str):
        """Setup the retriever, initializing private attributes."""
        try:
            logger.info(f"Loading embedding model: {embedding_model_name}")
            # Assign to the private attribute using underscore prefix
            self._model_embed = SentenceTransformer(embedding_model_name)

            logger.info(f"Setting up persistent ChromaDB client at: {persist_dir}")
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)

            logger.info("Setting up ChromaDB embedding function...")
            # Assign to the private attribute
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
            # You might need to explicitly cast or check type if using Any hint:
            # assert isinstance(self._embedding_function, embedding_functions.EmbeddingFunction)

            logger.info("Retriever setup complete.")
        except Exception as e:
            logger.error(f"Error during retriever setup: {e}", exc_info=True)
            # Ensure private attrs are None on failure
            self._client = None
            self._model_embed = None
            self._embedding_function = None
            raise RuntimeError(f"Failed to initialize RetrieveTool: {e}") from e

    

    def _run(self, query: str, collection_name: str, top_k: int = 5,
             use_mmr: bool = True, mmr_diversity: float = 0.3) -> str:
        # Access using underscore prefix
        if not self._client or not self._model_embed:
             error_msg = "Retriever Tool is not properly initialized. Setup failed."
             logger.error(error_msg)
             return error_msg
        # ... rest of the method remains the same, just ensure access is via self._client etc. ...
        logger.info(f"Running retrieval for query='{query}' in collection='{collection_name}' "
                    f"(top_k={top_k}, use_mmr={use_mmr}, mmr_diversity={mmr_diversity})")
        try:
            results = self.retrieve(query, collection_name, top_k, use_mmr, mmr_diversity) # Calls internal retrieve

            if not results:
                logger.info("No relevant documents found.")
                return "No relevant documents found matching the query."

            output = f"Found {len(results)} relevant document(s):\n\n"
            for i, result in enumerate(results, 1):
                output += f"--- Document {i} ---\n"
                output += f"Content: {result.get('document', '')}\n"
                metadata = result.get('metadata')
                if metadata:
                    output += f"Metadata: {metadata}\n"

            logger.info(f"Retrieval successful, returning {len(results)} documents.")
            return output.strip()

        except Exception as e:
            logger.error(f"Error during _run retrieval: {e}", exc_info=True)
            return f"An error occurred during retrieval: {str(e)}"


    def retrieve(self, query: str, collection_name: str, top_k: int = 5,
                 use_mmr: bool = True, mmr_diversity: float = 0.3) -> List[Dict[str, Any]]:
         # Access using underscore prefix
        if not self._client or not self._model_embed:
            raise RuntimeError("Retriever Tool is not properly initialized.")

        try:
            collection = self._get_collection(collection_name)
            if collection is None:
                 logger.warning(f"Collection '{collection_name}' not found.")
                 return []

            logger.debug(f"Retrieving from collection '{collection.name}'")
            if use_mmr:
                # Pass internal objects to MMR method
                return self._retrieve_with_mmr(query, collection, top_k, mmr_diversity)
            else:
                # Standard retrieval
                logger.debug(f"Performing standard query for '{query}' (top_k={top_k})")
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas"]
                )
                documents = results['documents'][0] if results.get('documents') else []
                metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [None] * len(documents)
                if len(metadatas) != len(documents):
                     metadatas = [None] * len(documents)
                retrieved_docs = [{"document": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]
                logger.debug(f"Standard retrieval found {len(retrieved_docs)} documents.")
                return retrieved_docs

        except Exception as e:
            logger.error(f"Error during retrieve method: {e}", exc_info=True)
            return []

    def _retrieve_with_mmr(self, query: str, collection: chromadb.Collection, top_k: int, diversity: float) -> List[Dict[str, Any]]:
        
        if not self._model_embed: # Check the private attribute
            raise RuntimeError("Embedding model not initialized for MMR.")
        
        
        collection_count = collection.count()
        if collection_count == 0:
            logger.debug("MMR: Collection is empty.")
            return []
        retrieve_k = min(max(top_k * 3, 10), collection_count)

        logger.debug(f"Performing MMR query for '{query}' (fetch k={retrieve_k}, final top_k={top_k}, diversity={diversity})")
        try:
            results = collection.query(
                query_texts=[query], n_results=retrieve_k, include=["documents", "metadatas", "embeddings"]
            )
        except Exception as e:
            logger.error(f"MMR: Error querying collection: {e}", exc_info=True)
            return []

        documents = results['documents'][0] if results.get('documents') and results['documents'] else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') and results['metadatas'] else [None] * len(documents)
        embeddings = results.get('embeddings', [[]])[0] if results.get('embeddings') and results['embeddings'] else []

        min_len = min(len(documents), len(metadatas) if metadatas else 0, len(embeddings) if embeddings else 0)
        if min_len == 0 or len(documents) == 0 or len(embeddings) == 0:
            logger.debug("MMR: Not enough valid data (docs/embeddings) found for MMR processing.")
            return []

        documents, embeddings = documents[:min_len], embeddings[:min_len]
        metadatas = metadatas[:min_len] if metadatas else [None] * min_len

        try:
            query_embedding = self._model_embed.encode(query, convert_to_numpy=True) # Use _model_embed
            doc_embeddings = np.array(embeddings, dtype=np.float32)

            if query_embedding is None or doc_embeddings.size == 0:
                 logger.error("MMR: Failed to encode query or process document embeddings.")
                 return []

            selected_indices, candidate_indices = [], list(range(len(documents)))
            actual_top_k = min(top_k, len(documents))

            while len(selected_indices) < actual_top_k and candidate_indices:
                remaining_embeddings = doc_embeddings[candidate_indices]
                if remaining_embeddings.size == 0: break

                relevance_scores = util.cos_sim(query_embedding, remaining_embeddings)[0].cpu().numpy()
                diversity_scores = np.zeros(len(candidate_indices))
                if selected_indices:
                    selected_embeddings = doc_embeddings[selected_indices]
                    if selected_embeddings.size > 0:
                        similarity_to_selected = util.cos_sim(remaining_embeddings, selected_embeddings).cpu().numpy()
                        diversity_scores = np.max(similarity_to_selected, axis=1)

                mmr_scores = (1 - diversity) * relevance_scores - diversity * diversity_scores
                best_candidate_local_idx = np.argmax(mmr_scores)
                best_candidate_global_idx = candidate_indices.pop(best_candidate_local_idx)
                selected_indices.append(best_candidate_global_idx)

            mmr_results = []
            for idx in selected_indices:
                 meta = metadatas[idx] if idx < len(metadatas) else None
                 mmr_results.append({"document": documents[idx], "metadata": meta})

            logger.debug(f"MMR processing complete, selected {len(mmr_results)} documents.")
            return mmr_results

        except Exception as e:
             logger.error(f"MMR: Error during MMR calculation: {e}", exc_info=True)
             return []


    def ingest_documents(self, documents: List[str], collection_name: str,
                        metadatas: Optional[List[Dict[str, Any]]] = None,
                        ids: Optional[List[str]] = None) -> Dict[str, Any]:
         # Access using underscore prefix
        if not self._client or not self._embedding_function:
            error_msg = "Retriever Tool is not properly initialized for ingestion."
            logger.error(error_msg)
            return {"status": "error", "message": error_msg, "count": 0}
        
        # (Keep the logic from previous version, just ensure access is via self._client/_embedding_function)
        logger.info(f"Ingesting {len(documents)} documents into collection '{collection_name}'")
        try:
            collection = self._get_or_create_collection(collection_name) # Uses private attrs internally

            # Validation logic... (same as before)
            if metadatas and len(metadatas) != len(documents):
                logger.warning(f"Mismatch between number of documents ({len(documents)}) and metadatas ({len(metadatas)}). Ignoring metadatas.")
                metadatas = None
            if ids and len(ids) != len(documents):
                logger.warning(f"Mismatch between number of documents ({len(documents)}) and IDs ({len(ids)}). Ignoring IDs.")
                ids = None
            final_ids = ids
            if final_ids is None:
                try: final_ids = [f"doc_{hash(doc) & 0xFFFFFFFFFFFFFFFF}" for doc in documents]
                except Exception: final_ids = [f"doc_idx_{i}" for i in range(len(documents))]
            final_metadatas = metadatas
            duplicates_found = False
            if collection.count() > 0 and final_ids:
                existing_ids = set(collection.get(include=[])['ids'])
                ids_to_add, docs_to_add, metas_to_add = [], [], [] if final_metadatas else None
                for i, doc_id in enumerate(final_ids):
                    if doc_id in existing_ids: duplicates_found = True; logger.warning(f"Skipping duplicate ID: {doc_id}")
                    else:
                        ids_to_add.append(doc_id); docs_to_add.append(documents[i])
                        if final_metadatas: metas_to_add.append(final_metadatas[i])
                if not docs_to_add: return {"status": "skipped", "message": "All provided documents had duplicate IDs.", "count": 0}
                final_ids, documents, final_metadatas = ids_to_add, docs_to_add, metas_to_add

            if not documents: return {"status": "skipped", "message": "No valid documents to add.", "count": 0}

            # Add using collection object
            collection.add(documents=documents, metadatas=final_metadatas, ids=final_ids)

            msg = f"Successfully ingested {len(documents)} documents into collection '{collection_name}'."
            if duplicates_found: msg += " Some duplicates were skipped."
            logger.info(msg)
            return {"status": "success", "message": msg, "count": len(documents)}

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}", exc_info=True)
            return {"status": "error", "message": f"Ingestion failed: {str(e)}", "count": 0}


    def _get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
         # Access using underscore prefix
        if not self._client or not self._embedding_function:
            raise RuntimeError("Retriever Tool is not properly initialized.")
        try:
            collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_function # Pass the internal embedding function
            )
            logger.info(f"Accessed or created collection: '{collection_name}'")
            return collection
        except Exception as e:
            logger.error(f"Failed to get or create collection '{collection_name}': {e}", exc_info=True)
            raise

    def _get_collection(self, collection_name: str) -> Optional[chromadb.Collection]:
         # Access using underscore prefix
         if not self._client:
            logger.error("Attempted to get collection, but client is not initialized.")
            return None
         try:
             # Pass the correct embedding function instance when getting
             collection = self._client.get_collection(
                 name=collection_name,
                 embedding_function=self._embedding_function # Important!
             )
             logger.debug(f"Successfully retrieved collection '{collection_name}'.")
             return collection
         except Exception as e:
             logger.warning(f"Collection '{collection_name}' not found or error retrieving it: {e}")
             return None


# --- IngestTool remains the same, but ensure its config has arbitrary_types_allowed=True ---
class IngestTool(BaseTool):
    name: str = "ChromaDB Document Ingest Tool"
    description: str = ("Ingests text documents, with optional metadata and IDs, "
                        "into a specified ChromaDB collection for later retrieval.")
    args_schema: type[V1BaseModel] = IngestInput
    model_config = ConfigDict(arbitrary_types_allowed=True) # Ensure this is here too
    retriever: RetrieveTool = Field(default=None, description="The RetrieveTool instance for document ingestion")

    def __init__(self, retriever_tool: RetrieveTool, **kwargs):
        super().__init__(**kwargs)
        # Check the private attribute _client on the passed tool instance
        if not isinstance(retriever_tool, RetrieveTool) or not retriever_tool._client:
             raise ValueError("IngestTool requires a properly initialized RetrieveTool instance.")
        self.retriever = retriever_tool
        logger.info("IngestTool initialized, using provided RetrieveTool instance.")

    def _run(self, documents: List[str], collection_name: str,
             metadatas: Optional[List[Dict[str, Any]]] = None,
             ids: Optional[List[str]] = None) -> str:
        logger.info(f"IngestTool executing ingestion for collection '{collection_name}'")
        if not documents:
             return "Ingestion failed: No documents provided."
        # Call the public method on the retriever instance
        result = self.retriever.ingest_documents(
            documents=documents, collection_name=collection_name, metadatas=metadatas, ids=ids
        )
        return f"Ingestion Result: {result.get('status', 'error')}. {result.get('message', 'Unknown error')}"