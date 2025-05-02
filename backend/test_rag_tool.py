# ./test_rag_tools.py

import os
import shutil
import time
from agent.tools.retrieve_tool import RetrieveTool, IngestTool

# --- Configuration ---
TEST_COLLECTION_NAME = "test_policy_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Make sure you have this model locally or it can be downloaded
PERSIST_DIR = "./_test_chroma_db"

# --- Helper Functions ---
def cleanup():
    """Removes the test ChromaDB directory."""
    if os.path.exists(PERSIST_DIR):
        print(f"\n--- Cleaning up test database at {PERSIST_DIR} ---")
        try:
            shutil.rmtree(PERSIST_DIR)
            print("Cleanup successful.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

# --- Test Execution ---
if __name__ == "__main__":
    print("--- Starting RAG Tool Test ---")
    cleanup() # Clean up previous runs if any

    try:
        # 1. Initialize Tools
        print("\n--- Initializing Tools ---")
        retriever_tool = RetrieveTool(
            embedding_model_name=EMBEDDING_MODEL,
            persist_dir=PERSIST_DIR
        )
        ingest_tool = IngestTool(retriever_tool=retriever_tool)
        print("Tools initialized successfully.")

        # 2. Prepare Sample Data
        print("\n--- Preparing Sample Data ---")
        documents = [
            "Shipping Policy: Standard shipping takes 3-5 business days.",
            "Return Policy: Items can be returned within 30 days of purchase.",
            "Product FAQ: The widget requires two AA batteries (not included).",
            "Shipping Policy: Express shipping is available for an additional $10.",
            "Company Info: Our headquarters are located in Tech City.",
            "Return Policy: Returns must be in original packaging.",
        ]
        metadatas = [
            {"source": "policy_doc_v1.pdf", "page": 1, "topic": "shipping"},
            {"source": "policy_doc_v1.pdf", "page": 2, "topic": "returns"},
            {"source": "product_manual_widget.txt", "topic": "usage"},
            {"source": "policy_doc_v2.pdf", "page": 1, "topic": "shipping"},
            {"source": "about_us.html", "topic": "location"},
            {"source": "policy_doc_v1.pdf", "page": 2, "topic": "returns"}, # Note: Same topic as another
        ]
        print(f"Sample documents prepared: {len(documents)}")

        # 3. Test Ingestion
        print("\n--- Testing Document Ingestion ---")
        ingestion_result = ingest_tool._run(
            documents=documents,
            collection_name=TEST_COLLECTION_NAME,
            metadatas=metadatas
            # ids are optional, letting the tool generate them
        )
        print(f"Ingestion Result: {ingestion_result}")
        # Basic check - In a real test suite, you'd assert specific outcomes
        if "success" not in ingestion_result.lower():
             raise RuntimeError("Ingestion test failed!")
        # Give ChromaDB a moment to process/persist
        print("Waiting briefly for index persistence...")
        time.sleep(2)


        # 4. Test Standard Retrieval
        print("\n--- Testing Standard Retrieval (use_mmr=False) ---")
        query_shipping = "Tell me about shipping times"
        retrieval_result_std = retriever_tool._run(
            query=query_shipping,
            collection_name=TEST_COLLECTION_NAME,
            top_k=2,
            use_mmr=False # Test standard search
        )
        print(f"\nQuery: '{query_shipping}' (Standard)")
        print("Result:\n", retrieval_result_std)
        # Check if expected content is present (simple check)
        if "shipping" not in retrieval_result_std.lower():
            print("WARNING: Standard retrieval might not have returned expected shipping info.")


        # 5. Test MMR Retrieval
        print("\n--- Testing MMR Retrieval (use_mmr=True) ---")
        query_policy = "What are the company policies?" # Broader query
        retrieval_result_mmr = retriever_tool._run(
            query=query_policy,
            collection_name=TEST_COLLECTION_NAME,
            top_k=3,
            use_mmr=True, # Test MMR
            mmr_diversity=0.7 # Higher diversity
        )
        print(f"\nQuery: '{query_policy}' (MMR, diversity=0.7)")
        print("Result:\n", retrieval_result_mmr)
        # Expect potentially diverse topics (shipping, returns, maybe location)
        if "shipping" not in retrieval_result_mmr.lower() or "return" not in retrieval_result_mmr.lower():
             print("WARNING: MMR retrieval might not have returned diverse policy info.")

        # 6. Test Retrieval for Non-existent Topic
        print("\n--- Testing Retrieval for Non-existent Topic ---")
        query_nonexistent = "Information about alien spacecraft"
        retrieval_result_none = retriever_tool._run(
            query=query_nonexistent,
            collection_name=TEST_COLLECTION_NAME,
            top_k=2
        )
        print(f"\nQuery: '{query_nonexistent}'")
        print("Result:\n", retrieval_result_none)
        if "no relevant documents found" not in retrieval_result_none.lower():
             print("WARNING: Retrieval for non-existent topic did not return the expected message.")

        # 7. Test Retrieval from Non-existent Collection
        print("\n--- Testing Retrieval from Non-existent Collection ---")
        retrieval_result_bad_coll = retriever_tool._run(
            query=query_shipping,
            collection_name="non_existent_collection",
            top_k=2
        )
        print(f"\nQuery: '{query_shipping}' (Bad Collection)")
        print("Result:\n", retrieval_result_bad_coll)
        if "no relevant documents found" not in retrieval_result_bad_coll.lower(): # Tool should return this if collection missing
             print("WARNING: Retrieval from bad collection did not return the expected message.")


        print("\n--- RAG Tool Test Completed Successfully ---")

    except Exception as e:
        print(f"\n--- RAG Tool Test Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 8. Cleanup
        cleanup()