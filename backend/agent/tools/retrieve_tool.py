from crewai_tools import BaseTool
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

class RetrieveInput(BaseModel):
    """Input for the retrieval tool."""
    query: str = Field(..., description="The query to search for relevant information")
    collection_name: str = Field(..., description="Name of the collection to search in")
    top_k: int = Field(5, description="Number of documents to retrieve")
    use_mmr: bool = Field(True, description="Whether to use Maximum Marginal Relevance for diverse results")
    mmr_diversity: float = Field(0.3, description="Diversity factor for MMR (0.0 to 1.0)")

class IngestInput(BaseModel):
    """Input for document ingestion."""
    documents: List[str] = Field(..., description="List of documents to ingest")
    collection_name: str = Field(..., description="Name of the collection to store documents")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each document")
    ids: Optional[List[str]] = Field(None, description="Optional IDs for the documents")

class RetrieveTool(BaseTool):
    """RAG tool for document retrieval with advanced MMR reranking."""
    name: str = "Retriever Tool"
    description: str = "Retrieves relevant information from documents using RAG with MMR reranking"
    input_schema: type[BaseModel] = RetrieveInput
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", persist_dir: Optional[str] = "./chroma_db"):
        """Initialize the retriever tool with specified embedding model."""
        super().__init__()
        self.model = SentenceTransformer(embedding_model_name)
        
        # Setup persistent ChromaDB
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
    
    def _run(self, query: str, collection_name: str, top_k: int = 5, 
             use_mmr: bool = True, mmr_diversity: float = 0.3) -> str:
        """Run the retrieval with the specified parameters."""
        results = self.retrieve(query, collection_name, top_k, use_mmr, mmr_diversity)
        
        if not results:
            return "No relevant documents found."
        
        output = f"Found {len(results)} relevant documents:\n\n"
        for i, result in enumerate(results, 1):
            output += f"Document {i}:\n{result['document']}\n"
            if result['metadata']:
                output += f"Metadata: {result['metadata']}\n"
            output += "\n---\n"
            
        return output
    
    def retrieve(self, query: str, collection_name: str, top_k: int = 5, 
                 use_mmr: bool = True, mmr_diversity: float = 0.3) -> List[Dict]:
        """Retrieve documents using either standard search or MMR."""
        try:
            collection = self._get_or_create_collection(collection_name)
            
            if use_mmr:
                return self._retrieve_with_mmr(query, collection, top_k, mmr_diversity)
            else:
                # Standard retrieval
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                
                documents = results['documents'][0]
                metadatas = results.get('metadatas', [None] * len(documents))[0]
                
                return [
                    {"document": doc, "metadata": meta}
                    for doc, meta in zip(documents, metadatas)
                ]
        except Exception as e:
            return [{"document": f"Error during retrieval: {str(e)}", "metadata": None}]
    
    def _retrieve_with_mmr(self, query: str, collection, top_k: int, diversity: float) -> List[Dict]:
        """Retrieve documents with Maximum Marginal Relevance for diversity."""
        # Get more results than needed for MMR filtering
        retrieve_k = min(top_k * 2, 100)
        results = collection.query(
            query_texts=[query],
            n_results=retrieve_k,
            include=["documents", "metadatas", "embeddings"]
        )
        
        documents = results['documents'][0]
        metadatas = results.get('metadatas', [None] * len(documents))[0]
        embeddings = results.get('embeddings', [None] * len(documents))[0]
        
        if not documents or not embeddings:
            return []
            
        # Convert query to embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Convert document embeddings to numpy arrays
        doc_embeddings = np.array(embeddings)
        
        # MMR algorithm implementation
        already_selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break
                
            # Calculate MMR scores
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance score
                relevance = util.cos_sim(query_embedding, doc_embeddings[idx])[0][0]
                
                # Diversity score
                diversity_score = 0
                if already_selected_indices:
                    similarities = [util.cos_sim(doc_embeddings[idx], doc_embeddings[sel_idx])[0][0] 
                                   for sel_idx in already_selected_indices]
                    diversity_score = max(similarities)
                
                # MMR score calculation
                mmr_score = (1 - diversity) * relevance - diversity * diversity_score
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            already_selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        # Return documents in MMR order
        mmr_results = []
        for idx in already_selected_indices:
            mmr_results.append({
                "document": documents[idx],
                "metadata": metadatas[idx] if metadatas else None
            })
            
        return mmr_results
    
    def ingest_documents(self, documents: List[str], collection_name: str, 
                        metadatas: Optional[List[Dict[str, Any]]] = None, 
                        ids: Optional[List[str]] = None) -> Dict:
        """Ingest documents into the retrieval system."""
        # Get or create collection
        collection = self._get_or_create_collection(collection_name)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{hash(doc)}" for i, doc in enumerate(documents)]
        
        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "status": "success",
            "message": f"Ingested {len(documents)} documents into collection '{collection_name}'",
            "count": len(documents)
        }
    
    def _get_or_create_collection(self, collection_name: str):
        """Get a collection or create if it doesn't exist."""
        try:
            return self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            return self.client.create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )

class IngestTool(BaseTool):
    """Tool for ingesting documents into the retrieval system."""
    name: str = "Document Ingest Tool"
    description: str = "Ingests documents into the retrieval system"
    input_schema: type[BaseModel] = IngestInput
    
    def __init__(self, retriever_tool: RetrieveTool):
        """Initialize with reference to a retriever tool."""
        super().__init__()
        self.retriever = retriever_tool
    
    def _run(self, documents: List[str], collection_name: str, 
             metadatas: Optional[List[Dict[str, Any]]] = None, 
             ids: Optional[List[str]] = None) -> str:
        """Run document ingestion."""
        result = self.retriever.ingest_documents(
            documents=documents,
            collection_name=collection_name,
            metadatas=metadatas,
            ids=ids
        )
        
        return f"Ingestion result: {result['status']}. {result['message']}"