import os 
from typing import List, Dict, Any, Optional, TypedDict, Literal
from loguru import logger
import json 


from agent.utils.helper_function import extract_clean_json 
from agent.tools.retrieve_tool import RetrieveTool, IngestTool

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from crewai_tools import SerperDevTool

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import chromadb
from abc import ABC, abstractmethod
load_dotenv()

os.environ["SERPER_API_KEY"] = "a6ce4b5fc4ef3f9754144d519ecf9e418bc1c7bc"
os.environ["OPENAI_API_KEY"] = "NA"

TEST_COLLECTION_NAME = "agent_test_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Ensure this model is available locally
PERSIST_DIR = "./ai_agent_db"
LLM_MODEL = "deepseek-r1:1.5b" # 

persistent_client = chromadb.PersistentClient(path=PERSIST_DIR)
logger.info(f"ChromaDB client initialized. Persistence directory: {PERSIST_DIR}")

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# add logging file 
log_file = "logs/agent_log.log"
logger.add(log_file, rotation="1 MB", retention="7 days", compression="zip")
logger.info("Agent Graph Initialized")

def setup_enviroment() -> tuple: 
    logger.info("Setting up environment variables")
    try: 
        logger.info("Setting up environment variables")
        llm = ChatOllama(model = LLM_MODEL, temperature=0.2, max_tokens=2000)
        logger.info("LLM model initialized")
        embedding_model = OllamaEmbeddings(model = EMBEDDING_MODEL)
        logger.info("Embedding model initialized")
        retriever_tool = RetrieveTool(
            embedding_model_name=EMBEDDING_MODEL,
            persist_dir=PERSIST_DIR
        )
        logger.info("Retriever tool initialized")
        ingest_tool = IngestTool(
            embedding_model_name=EMBEDDING_MODEL,
            persist_dir=PERSIST_DIR
        )
        logger.info("Ingest tool initialized")
        return llm, embedding_model, retriever_tool, ingest_tool
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise e
    
class GradeDocument(BaseModel):
    """ Score the document based on its content. """
    score: float = Field(..., description="Score of the document")
    feedback: str = Field(..., description="Feedback on the document")
    document_id: int = Field(..., description="ID of the document")


class AgentState(TypedDict):
    original_query: str
    customer_id: Optional[str]
    rewritten_query: str
    documents : List[Dict[str, Any]]
    products : List[Dict[str, Any]]
    classified_agent: Literal["Company", "Customer", "Naive", "Unknown"]
    assigned_task : Literal["Retrieval Task", "Search Task", "Query Product","Response Task"]  # Optional field for assigned agent
    agent_response: str
    reflection: str
    is_final: bool
    error: Optional[str]
    retry_count: int
    suggested_questions: List[str]  # New field for suggested questions
    chat_history : Optional[List[dict]]  # Optional chat history for context

logger.info("AgentState class defined")

class BaseAgentTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tool = None


    @abstractmethod
    def setup_tool(self):
        logger.info("Setting up the tool")
        self.tool = {
            "name": self.name,
            "description": self.description
        }
        logger.info(f"Tool set up: {self.tool}")
    @abstractmethod
    def run(self, query, **kwargs):
        logger.info("Running the tool")
        # Placeholder for tool-specific logic
        pass

class SearchTool(BaseAgentTool):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        

    def setup_tool(self):
        logger.info("Setting up the search tool")
        from langchain_tavily import TavilySearch
        self.tool = {
            "name": self.name,
            "description": self.description
        }
        self.search = TavilySearch(max_results=5, topic="general")

        logger.info(f"Search tool set up: {self.tool}")

    def run(self, query):
        logger.info(f"Running search tool with query: {query}")
        # Placeholder for search logic
        resutls = self.search.invoke(query)
        logger.info(f"Search results: {resutls}")
        content_results = []
        for result in resutls["results"]:
            if result["score"] > 0.5:
                text = f"""Document Title: {result["title"]}\n
                Document URL: {result["url"]}\n
                Document Content: {result["content"]}\n
                Document Score: {result["score"]}\n
                """
                content_results.append(text)
        logger.info(f"Search results: {content_results}")
        return content_results

class RetrieveTool(BaseAgentTool):
    def __init__(self, embedding_model_name: str, persist_dir: str, collection_name: Optional[str] = None):
        super().__init__(name="Retrieve Tool", description="Retrieves documents from the vector store")
        self.embedding_model_name = embedding_model_name
        self.persist_dir = persist_dir
        self.vector_store = None
        self.collection_name = collection_name
        self.embedding_model = self.load_embedding_model(embedding_model_name)


    def load_embedding_model(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        try:
            embedding_model = OllamaEmbeddings(model=model_name)
            logger.info(f"Embedding model loaded: {embedding_model}")
            return embedding_model
        except Exception as e:
            logger.error(f"Error loading embedding model '{model_name}': {e}")
            raise e
        

    
    def setup_tool(self):
        logger.info("Setting up the retrieve tool")
        self.vector_store = Chroma(
            client=persistent_client,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )
        logger.info(f"Retrieve tool set up with vector store: {self.vector_store}")

    def run(self, query, **kwargs):
        logger.info(f"Running retrieve tool with query: {query}")
        
        # Placeholder for retrieval logic
        retriever = self.get_retriever()
        if not retriever:
            logger.error("Retriever is not available.")
            return []
        results = retriever.get_relevant_documents(query)
        if not results:
            logger.warning("No results found.")
            return []

        logger.info(f"Retrieved results: {results}")
        return self.format_docs(results)   
    
    def format_docs(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Formats documents for output."""
        formatted_docs = []
        for doc in docs:
            formatted_doc = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            formatted_docs.append(formatted_doc)
        logger.info(f"Formatted {len(docs)} documents.")
        return formatted_docs

    def get_retriever(self):
        """Returns the retriever object for the vector store."""
        try:
            retriever = self.vector_store.as_retriever()
            logger.info(f"Retriever object created: {retriever}")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {e}", exc_info=True)
            return None

    def process_file_content(self, content: bytes, filename: str) -> List[Document]:
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
    
    def add_documents(self, documents: List[Document]):
        """Adds documents to the vector store."""
        try:
            self.vector_store.add_documents(documents, collection_name=self.collection_name)
            logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding documents to collection '{self.collection_name}': {e}", exc_info=True)

    def delete_collection(self, collection_name: str):
        """Deletes a collection from the vector store."""
        try:
            self.vector_store.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}", exc_info=True)



class Nodes:

    def rewrite_query_node(state: AgentState) -> AgentState:
        """Rewrite the original query."""
        # Placeholder for rewriting logic
        state["rewritten_query"] = f"Rewritten: {state['original_query']}"
        return state

    def company_agent_node(state: AgentState) -> AgentState:
        """Route to the company agent."""
        # Placeholder for company-specific logic
        state["classified_agent"] = "Company"
        state["assigned_task"] = "Response Task"
        return state
    
    def customer_agent_node(state: AgentState) -> AgentState:
        """Route to the customer agent."""
        # Placeholder for customer-specific logic
        state["classified_agent"] = "Customer"
        state["assigned_task"] = "Response Task"
        return state
    
    def naive_agent_node(state: AgentState) -> AgentState:
        """Route to the naive agent."""
        # Placeholder for naive-specific logic
        state["classified_agent"] = "Naive"
        state["assigned_task"] = "Response Task"
        return state
    
    def product_sql_agent_node(state: AgentState) -> AgentState:
        """Route to the product agent."""
        # Placeholder for product-specific logic
        state["classified_agent"] = "Product"
        state["assigned_task"] = "Response Task"
        return state

    def generate_next_question_node(state: AgentState) -> AgentState:
        """Generate the next question."""
        # Placeholder for question generation logic
        state["suggested_questions"] = ["What is your name?", "How can I help you?"]
        return state
    
    def reflection_node(state: AgentState) -> AgentState:
        """Reflect on the agent's response."""
        # Placeholder for reflection logic
        state["reflection"] = "Good response"
        return state
    
    def router_node(state: AgentState) -> AgentState:
        """Route to the appropriate agent."""
        # Placeholder for routing logic
        if state["customer_id"]:
            state = Nodes.customer_agent_node(state)
        else:
            state = Nodes.company_agent_node(state)
        return state


class GSTAgent:




class AgentGraph:
    def __init__(self, llm, embedding_model, retriever_tool, ingest_tool):
        self.llm = llm
        self.embedding_model = embedding_model
        self.retriever_tool = retriever_tool
        self.ingest_tool = ingest_tool
        self.agent_state: AgentState = {
            "original_query": "",
            "customer_id": None,
            "rewritten_query": "",
            "documents": [],
            "products": [],
            "classified_agent": "Unknown",
            "assigned_task": "Retrieval Task",
            "agent_response": "",
            "reflection": "",
            "is_final": False,
            "error": None,
            "retry_count": 0,
            "suggested_questions": [],
            "chat_history": []
        }
        self.serper_tool = SerperDevTool()
        self.setup_tools()
        self.setup_prompt()
        self.setup_vector_store()
        self.setup_agent()
        self.setup_chat_prompt()
        self.setup_system_message()
        self.setup_human_message()
        self.setup_ai_message()
        self.setup_agent_response()
        self.setup_reflection()
        self.setup_final_response()
        self.setup_error_handling()
        self.setup_retry()
        self.setup_suggested_questions()
        self.setup_chat_history()
        self.setup_classified_agent()
        self.setup_assigned_task()
        self.setup_documents()
        self.setup_products()
        self.setup_customer_id()
        self.setup_original_query()
        self.setup_rewritten_query()
        self.setup_agent_state()
        self.setup_logger()
        self.setup_environment()
        self.setup_agent_graph()
        