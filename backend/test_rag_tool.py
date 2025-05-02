import os
from agent.tools.retrieve_tool import RetrieveTool, IngestTool

# Setup test environment
def setup_test_environment():
    # Sample documents for testing
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require significant training data",
        "Python is a popular programming language for data science",
        "Natural language processing helps computers understand human language",
        "Deep learning is a subset of machine learning",
        "Artificial intelligence aims to mimic human intelligence",
        "Neural networks are inspired by the human brain",
        "Transfer learning leverages pre-trained models for new tasks",
        "Data preprocessing is crucial for machine learning success",
        "Feature engineering improves model performance",
    ]
    
    # Create metadata for each document
    metadatas = [
        {"category": "proverb", "complexity": "simple"},
        {"category": "machine learning", "complexity": "medium"},
        {"category": "programming", "complexity": "simple"},
        {"category": "nlp", "complexity": "medium"},
        {"category": "machine learning", "complexity": "medium"},
        {"category": "ai", "complexity": "medium"},
        {"category": "machine learning", "complexity": "advanced"},
        {"category": "machine learning", "complexity": "advanced"},
        {"category": "data science", "complexity": "simple"},
        {"category": "data science", "complexity": "medium"},
    ]
    
    # Create retriever tool and ingest tool
    retriever = RetrieveTool(embedding_model_name="all-MiniLM-L6-v2", persist_dir="./test_chroma_db")
    ingest_tool = IngestTool(retriever_tool=retriever)
    
    # Ingest documents
    ingest_result = ingest_tool._run(
        documents=documents,
        collection_name="test_collection",
        metadatas=metadatas
    )
    print(f"Ingestion: {ingest_result}\n")
    
    return retriever

def run_tests(retriever):
    # Test queries
    queries = [
        "What is machine learning?",
        "How does programming relate to data science?",
        "Explain artificial intelligence concepts"
    ]
    
    for query in queries:
        print(f"\n=== Testing Query: '{query}' ===\n")
        
        # Test 1: Standard retrieval (no MMR)
        print("Test 1: Standard Retrieval")
        results = retriever._run(
            query=query,
            collection_name="test_collection",
            top_k=3,
            use_mmr=False
        )
        print(results)
        
        # Test 2: MMR with low diversity (similar to standard)
        print("\nTest 2: MMR with Low Diversity (0.1)")
        results = retriever._run(
            query=query,
            collection_name="test_collection",
            top_k=3,
            use_mmr=True,
            mmr_diversity=0.1
        )
        print(results)
        
        # Test 3: MMR with medium diversity
        print("\nTest 3: MMR with Medium Diversity (0.5)")
        results = retriever._run(
            query=query,
            collection_name="test_collection",
            top_k=3,
            use_mmr=True,
            mmr_diversity=0.5
        )
        print(results)
        
        # Test 4: MMR with high diversity
        print("\nTest 4: MMR with High Diversity (0.8)")
        results = retriever._run(
            query=query,
            collection_name="test_collection",
            top_k=3,
            use_mmr=True,
            mmr_diversity=0.8
        )
        print(results)

if __name__ == "__main__":
    print("Setting up test environment and ingesting documents...")
    retriever = setup_test_environment()
    
    print("Running retrieval tests...")
    run_tests(retriever)
    
    print("\nTests completed!")