# ./test_agent_with_rag.py

import os
import shutil
import time
import logging
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process

# LLM
from langchain_community.chat_models.ollama import ChatOllama

from agent.tools.retrieve_tool import RetrieveTool, IngestTool

# --- Configuration ---
load_dotenv() # Load environment variables if needed (e.g., API keys for other tools)
TEST_COLLECTION_NAME = "agent_test_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Ensure this model is available locally
PERSIST_DIR = "./_agent_test_chroma_db"
LLM_MODEL = "deepseek-r1:1.5b" # Or your preferred Ollama model
# llm = ChatOllama(model='deepseek-r1:1.5b', temperature=0.2, max_tokens=2000)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def setup_environment():
    """Initializes LLM, Tools, and cleans up old DB."""
    logger.info("--- Setting up test environment ---")
    cleanup() # Ensure clean state before starting

    try:
        # Initialize LLM
        logger.info(f"Loading LLM: {LLM_MODEL}")
        llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
        # Simple check if LLM is accessible (optional, Ollama might not have a direct check)
        # llm.invoke("Hi")
        logger.info("LLM loaded successfully.")

        # Initialize Tools
        logger.info("Initializing RAG Tools...")
        retriever_tool = RetrieveTool(
            embedding_model_name=EMBEDDING_MODEL,
            persist_dir=PERSIST_DIR
        )
        ingest_tool = IngestTool(retriever_tool=retriever_tool)
        logger.info("RAG Tools initialized successfully.")

        return llm, retriever_tool, ingest_tool

    except Exception as e:
        logger.error(f"Failed to set up environment: {e}", exc_info=True)
        raise

def cleanup():
    """Removes the test ChromaDB directory."""
    if os.path.exists(PERSIST_DIR):
        logger.info(f"--- Cleaning up test database at {PERSIST_DIR} ---")
        try:
            shutil.rmtree(PERSIST_DIR)
            logger.info("Cleanup successful.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    llm = None
    retriever_tool = None
    ingest_tool = None

    try:
        # 1. Setup Environment
        llm, retriever_tool, ingest_tool = setup_environment()
        tools = [retriever_tool, ingest_tool]

        # 2. Define Agent
        logger.info("--- Defining Knowledge Base Manager Agent ---")
        kb_manager_agent = Agent(
            role='Knowledge Base Manager',
            goal=f"Efficiently manage and retrieve information from the company's knowledge base stored in ChromaDB. Use the provided tools to ingest new documents into specific collections and retrieve relevant information based on queries.",
            backstory=(
                "You are an expert AI assistant responsible for maintaining the accuracy and accessibility "
                "of the company's document knowledge base. You meticulously ingest new information using the "
                "'ChromaDB Document Ingest Tool' and expertly query the database using the "
                "'ChromaDB Retriever Tool' to answer questions. Always specify the correct collection name."
            ),
            llm=llm,
            tools=tools,
            verbose=True, # Set to True to see LLM reasoning and tool calls
            allow_delegation=False,
            # memory=True # Optional: Enable memory for conversation context if needed
        )
        logger.info(f"Agent '{kb_manager_agent.role}' created.")

        # 3. Prepare Sample Data (Hardcoded for simplicity in this example)
        sample_docs = [
            "Workflow Policy: All project proposals must be submitted via the online portal by Friday 5 PM.",
            "Data Security Policy: Use strong, unique passwords for all company accounts and enable MFA.",
            "Workflow Policy: Status updates are due every Monday morning.",
            "Onboarding Guide: New hires should complete IT setup within their first two days.",
            "Data Security Policy: Do not share sensitive customer information via unencrypted email.",
        ]
        sample_metadatas = [
            {"source": "workflows_v2.doc", "topic": "proposals"},
            {"source": "security_policy_v4.pdf", "topic": "passwords"},
            {"source": "workflows_v2.doc", "topic": "updates"},
            {"source": "hr_onboarding.pdf", "topic": "setup"},
            {"source": "security_policy_v4.pdf", "topic": "communication"},
        ]
        # Format for potentially including in task description if needed (though relying on tool args is better)
        docs_repr = repr(sample_docs)
        metas_repr = repr(sample_metadatas)


        # 4. Define Tasks
        logger.info("--- Defining Tasks ---")

        # Task 1: Ingestion
        # The description guides the agent to use the tool and specifies *what* to ingest.
        # The agent's LLM should extract the necessary arguments (documents, collection_name, metadatas)
        # for the IngestTool based on its args_schema.
        task_ingest = Task(
            description=(
                f"Ingest the following set of documents into the '{TEST_COLLECTION_NAME}' collection "
                f"using the 'ChromaDB Document Ingest Tool'. Ensure you pass both the document texts "
                f"and their corresponding metadata.\n\n"
                f"Documents to ingest: {docs_repr}\n"
                f"Associated Metadatas: {metas_repr}"
            ),
            expected_output=(
                f"Confirmation that {len(sample_docs)} documents were successfully ingested "
                f"into the '{TEST_COLLECTION_NAME}' collection."
            ),
            agent=kb_manager_agent,
            tools=[ingest_tool] # Optional: Limit tools for this specific task
        )
        logger.info("Ingestion task defined.")

        # Task 2: Retrieval
        # This task depends on the first one completing successfully.
        query = "What is the policy regarding project proposals and data security?"
        task_retrieve = Task(
            description=(
                f"Search the '{TEST_COLLECTION_NAME}' collection using the 'ChromaDB Retriever Tool' "
                f"to find information relevant to the following query: '{query}'. "
                f"Retrieve the top 3 most relevant documents using MMR for diversity. " # Explicitly guide MMR usage
                f"Present the content of the retrieved documents clearly."
            ),
            expected_output=(
                "A summary or list of the content from the top 3 relevant documents found in the "
                f"'{TEST_COLLECTION_NAME}' collection related to '{query}', retrieved using MMR."
            ),
            agent=kb_manager_agent,
            context=[task_ingest], # Make this task depend on the ingestion task
            tools=[retriever_tool] # Optional: Limit tools for this specific task
        )
        logger.info("Retrieval task defined.")


        # 5. Create and Run Crew
        logger.info("--- Creating and Running the Crew ---")
        company_knowledge_crew = Crew(
            agents=[kb_manager_agent],
            tasks=[task_ingest, task_retrieve],
            process=Process.sequential, # Ensure tasks run in order: ingest then retrieve
            verbose=2 # Use verbose=2 to see detailed LLM thoughts and tool calls
        )

        # Kick off the crew execution
        # For tasks requiring complex input not easily put in description, use `inputs`
        # inputs = {'documents': sample_docs, 'metadatas': sample_metadatas, 'query': query}
        # result = company_knowledge_crew.kickoff(inputs=inputs)
        # However, here we've put the data in the task description for simplicity.
        result = company_knowledge_crew.kickoff()

        logger.info("--- Crew Execution Finished ---")
        print("\n\n===== Final Crew Result =====")
        print(result)
        print("============================")

    except Exception as e:
        logger.error(f"\n--- An error occurred during the agent test ---")
        logger.error(e, exc_info=True)

    finally:
        # 6. Cleanup
        cleanup()
        logger.info("--- Test script finished ---")