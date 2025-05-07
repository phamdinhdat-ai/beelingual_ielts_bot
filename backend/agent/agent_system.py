# app/agent_system.py

import os
from typing import Optional, TypedDict, Literal, List, Tuple # Add Tuple
import loguru
from loguru import logger
import json
import asyncio

# --- Framework Imports ---
from crewai import Agent as CrewAgent, Task # Keep Task if using CrewAI's execution style
from crewai_tools import SerperDevTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- LangChain Imports ---
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain.sql_database import SQLDatabase # For Product SQL Agent
# from langchain_experimental.sql import SQLDatabaseChain # Old way
# For a more robust SQL agent, consider SQLDatabaseToolkit and create_sql_agent

# --- Application Imports ---
from agent.vector_store_manager import create_retriever_tool
from agent.utils.helper_function import extract_clean_json
# --- Configuration & Logging ---
from dotenv import load_dotenv
load_dotenv()
# logger.add("multi_agents.log", rotation="1 MB", retention="10 days", level="DEBUG") # Already added
# logger.info("Starting agent_system.py") # Already added

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "YOUR_SERPER_KEY_HERE")
# os.environ["OPENAI_API_KEY"] = "NA"

LLM_MODEL = os.getenv("LLM_MODEL", "mistral:latest")
PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL", "sqlite:///./product_database.db") # Load from .env

# --- Initialize LLM ---
try:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2, max_tokens=2000) # Lowered temp for more deterministic SQL/JSON
    logger.info(f"LLM '{LLM_MODEL}' loaded.")
except Exception as e:
    logger.error(f"Failed to load LLM '{LLM_MODEL}': {e}", exc_info=True)
    raise

# --- Initialize Product Database and SQL Tool ---
try:
    if not PRODUCT_DB_URL:
        raise ValueError("PRODUCT_DB_URL is not set in environment variables.")
    product_db = SQLDatabase.from_uri(PRODUCT_DB_URL)
    logger.info(f"Connected to Product Database: {product_db.dialect} - tables: {product_db.get_usable_table_names()}")

    # Custom SQL execution function for the tool
    def run_sql_query_for_tool(sql_query: str) -> str:
        """Executes a SQL query against the product database."""
        try:
            # Basic validation: only allow SELECT statements for safety in this example
            if not sql_query.strip().upper().startswith("SELECT"):
                logger.warning(f"Attempted to run non-SELECT SQL query: {sql_query}")
                return "Error: Only SELECT queries are allowed for product information."
            logger.info(f"Executing SQL against product DB: {sql_query}")
            result = product_db.run(sql_query)
            logger.info(f"Product DB SQL Result: {str(result)[:500]}...") # Log snippet
            return str(result)
        except Exception as e:
            logger.error(f"Error executing SQL query '{sql_query}' on product DB: {e}", exc_info=True)
            return f"Error during SQL execution: {e}. Please check your query or the database schema."

    sql_tool = Tool(
        name="ProductDatabaseSQLTool",
        func=run_sql_query_for_tool,
        description=f"Use this tool to execute a SQL query against the product database to find information about products. Input MUST be a valid SQL query. Available tables: {product_db.get_usable_table_names()}. Schema details: {product_db.get_table_info()}",
    )
    logger.info("Product SQL Tool initialized.")
except Exception as e:
    logger.error(f"Failed to initialize product database or SQL tool: {e}. ProductSQLAgent will be impaired.", exc_info=True)
    def dummy_sql(query:str): return "Product database tool is unavailable due to an initialization error."
    sql_tool = Tool(name="ProductDatabaseSQLTool", func=dummy_sql, description="Product DB tool unavailable.")

# --- Initialize Static Tools (Search Tool) ---
try:
    search_tool = SerperDevTool(n_results=3)
    logger.info("Search tool initialized.")
except Exception as e:
     logger.error(f"Failed to initialize SerperDevTool: {e}. Check SERPER_API_KEY.", exc_info=True)
     def dummy_search(query: str): return "Search tool unavailable."
     search_tool = Tool(name="search", func=dummy_search, description="Search tool unavailable.")


# --- Agent State Definition ---
class AgentState(TypedDict):
    original_query: str
    user_id: Optional[int]
    guest_session_id: Optional[str]
    customer_id: Optional[str] # For explicit DB customer ID (not login user ID)
    rewritten_query: str
    classified_agent: Literal["Company", "Customer", "Naive", "ProductSQL", "Unknown"] # Added ProductSQL
    agent_response: str
    reflection: str
    is_final: bool
    error: Optional[str]
    retry_count: int
    suggested_questions: List[str]
    chat_history: List[Tuple[str, str]]

logger.info("LangGraph State Defined.")

# --- CrewAI Agent Base Definitions ---
company_agent_def_base = {
    'role': 'Company Information Specialist',
    'goal': 'Provide accurate information about the company (e.g., Google) including location, products, services, history, and mission, based on search results and internal knowledge.',
    'backstory': 'You are an AI assistant dedicated to representing the company accurately. Use search tools to find the latest public information.',
    'llm': llm, 'verbose': True, 'allow_delegation': False
}

customer_agent_def_base = {
    'role': 'Customer Support Agent',
    'goal': 'Answer customer-specific questions by retrieving information from their uploaded documents using the document retriever tool OR from an internal customer database if an explicit customer_id is provided and a tool exists for it.',
    'backstory': 'You are a helpful support agent. You MUST prioritize using the document retriever tool if available for queries about uploaded content. For other customer-specific data, check if an explicit database customer_id is given.',
    'llm': llm, 'verbose': True, 'allow_delegation': False
}

naive_agent_def_base = {
    'role': 'General Knowledge Assistant',
    'goal': 'Answer general knowledge questions or queries that do not fall under specific company or customer information categories. Use search tools to find relevant information online.',
    'backstory': 'You are a general-purpose AI assistant capable of answering a wide range of topics by searching the internet.',
    'llm': llm, 'verbose': True, 'allow_delegation': False
}

# New: Product SQL Agent Base Definition
product_sql_agent_def_base = {
    'role': 'Product Database Query Specialist',
    'goal': 'Given a user question about products, first generate an accurate SQL query to fetch relevant data from the product database. Then, use the "ProductDatabaseSQLTool" to execute this SQL query. Finally, present the query results clearly to the user, or explain if no results were found.',
    'backstory': f'You are an expert in SQL and product data. You have a tool named "ProductDatabaseSQLTool" to execute SQL queries. Available tables and schema: {product_db.get_table_info() if "product_db" in globals() and product_db else "Schema unavailable"}. Focus on SELECT queries. If you need to generate SQL, ensure it is correct for the schema. After getting results from the tool, explain them in natural language.',
    'llm': llm, # This LLM will generate SQL and interpret results
    'tools': [sql_tool], # The tool that executes SQL
    'verbose': True,
    'allow_delegation': False
}
logger.info("CrewAI Agent Base Definitions Ready.")


# --- LangGraph Nodes ---

# Node 1: Rewriter (MODIFIED to include ProductSQL classification)
def rewrite_query_node(state: AgentState):
    logger.info("--- Node: Rewriter ---")
    state['error'] = None; state['agent_response'] = ""; state['reflection'] = ""; state['suggested_questions'] = []

    query = state['original_query']
    chat_history = state.get('chat_history', [])
    formatted_history = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history[-3:]]) # Last 3 turns

    # Refined system prompt for rewriter
    system_prompt = f"""You are an expert query processor. Consider the recent conversation history:
<history>
{formatted_history}
</history>

Your tasks for the LATEST user query are:
1. Correct spelling mistakes.
2. Rephrase for clarity if necessary.
3. Classify the query's intent. Base classification purely on the query's content:
    - 'Company': For questions about a specific company's general details (location, public products, services, mission etc.). Assume 'Google' if not specified.
    - 'Customer': ONLY if the query explicitly asks about user-specific data like "my orders", "my account details", "information in my uploaded document X", "summarize the file I sent".
    - 'ProductSQL': For questions specifically about product details, inventory, pricing, features, comparisons, or categories that would require querying a structured product database. Examples: "Do you have red shoes in size 10?", "List all laptops under $1000", "What are the specifications of product X?".
    - 'Naive': For general knowledge, definitions, facts, or queries not fitting other categories.
    - 'Unknown': If ambiguous or unclassifiable.
4. Extract an explicit customer ID (like cust123, user_id_789) IF AND ONLY IF it's mentioned for a *database lookup* (not a general user login/session). Return null otherwise.
Provide your output ONLY as a JSON object with keys: "rewritten_query", "agent_classification", "extracted_customer_id". Example:
{{"rewritten_query": "What are the specifications for the Pro Laptop?", "agent_classification": "ProductSQL", "extracted_customer_id": null}}"""

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Process latest user query: {query}")])
    rewriter_chain = prompt | llm

    try:
        response_content = rewriter_chain.invoke({"query": query}).content
        json_str = extract_clean_json(response_content)
        if json_str:
            response_data = json.loads(json_str)
            logger.debug(f"Rewriter Raw Output: {response_content}")
            logger.info(f"Rewriter Parsed JSON: {response_data}")

            classification = response_data.get('agent_classification', 'Unknown')
            extracted_db_customer_id = response_data.get('extracted_customer_id')

            # If ProductSQL is classified, and a DB customer ID is also extracted,
            # it's confusing. ProductSQL usually doesn't need a *customer* ID.
            # Let's prioritize ProductSQL if it's chosen.
            if classification == "ProductSQL" and extracted_db_customer_id:
                logger.warning("ProductSQL classified, but also extracted a DB customer_id. Ignoring extracted_db_customer_id for ProductSQL.")
                extracted_db_customer_id = None # ProductSQL agent doesn't use this.

            # If an explicit DB customer_id is extracted, and classification isn't already Customer or ProductSQL,
            # it's likely a 'Customer' query (for a specific customer DB, not document RAG).
            # This scenario is less common with the ProductSQL agent now.
            # We'll rely on the LLM's primary classification.

            valid_class = {"Company", "Customer", "Naive", "ProductSQL", "Unknown"}
            if classification not in valid_class:
                 logger.warning(f"Invalid classification '{classification}' from LLM. Defaulting to Unknown.")
                 classification = "Unknown"

            return {
                "rewritten_query": response_data.get('rewritten_query', query),
                "classified_agent": classification,
                "customer_id": extracted_db_customer_id, # This is for *explicit DB customer ID* for a CRM/Order DB
                "error": None
            }
        else: # Fallback if JSON parsing fails
            logger.warning(f"Rewriter output was not valid JSON. Raw: {response_content}. Applying fallback.")
            # Simple fallback
            if any(kw in query.lower() for kw in ["product", "item", "inventory", "price of", "stock"]):
                classification = "ProductSQL"
            elif any(kw in query.lower() for kw in ["my ", "i ", " me ", "account", "order", "document", "file"]):
                classification = "Customer"
            else:
                classification = "Naive" # Safer default than Unknown
            return {"rewritten_query": query, "classified_agent": classification, "customer_id": None, "error": "Invalid JSON from rewriter."}
    except Exception as e:
        logger.error(f"Error in Rewriter node: {e}", exc_info=True)
        # Fallback classification on error
        if any(kw in query.lower() for kw in ["product", "item", "inventory", "price of", "stock"]):
            classification = "ProductSQL"
        elif any(kw in query.lower() for kw in ["my ", "i ", " me ", "account", "order", "document", "file"]):
            classification = "Customer"
        else:
            classification = "Naive"
        return {"error": f"Failed to process query: {e}", "classified_agent": classification, "rewritten_query": query, "customer_id": None}


# --- Agent Execution Nodes ---
def execute_agent_node(
    state: AgentState,
    agent_base_config: dict,
    agent_name: str,
    static_tools: Optional[List[Tool]] = None
):
    logger.info(f"--- Node: Execute {agent_name} Agent Task ---")
    # ... (error checking as before) ...
    state["error"] = None # Clear error before execution

    query = state['rewritten_query']
    user_id = state.get('user_id')
    guest_session_id = state.get('guest_session_id')
    # `customer_id` from state is for explicit DB lookups (e.g. CRM), not for document RAG context.
    # Document RAG uses user_id or guest_session_id to name its collection.

    current_tools = list(static_tools) if static_tools else []
    retriever_tool_instance = None # Renamed to avoid conflict with global var

    if agent_name == "Customer":
        collection_context_id = None
        if user_id:
            collection_context_id = f"user_{user_id}"
        elif guest_session_id:
            clean_guest_id = guest_session_id.replace("-", "_")
            collection_context_id = f"guest_{clean_guest_id}"[:63]

        if collection_context_id:
            logger.info(f"Creating/getting retriever tool for Customer Agent, collection context: {collection_context_id}")
            retriever_tool_instance = create_retriever_tool(collection_name=collection_context_id) # This is from vector_store_manager
            if retriever_tool_instance:
                current_tools.append(retriever_tool_instance)
        else:
            logger.warning("Customer Agent: No user_id or guest_session_id provided for document retrieval.")
    
    # Add search tool if it's defined and not already in list (e.g. Customer might need it)
    if search_tool and hasattr(search_tool, 'name') and search_tool.name not in [t.name for t in current_tools]:
         if agent_name == "Customer" or agent_name == "Naive" or agent_name == "Company": # ProductSQL uses its own tool
              current_tools.append(search_tool)


    # Ensure agent_base_config is a dictionary
    if not isinstance(agent_base_config, dict):
        logger.error(f"agent_base_config for {agent_name} is not a dict: {agent_base_config}")
        return {"error": f"Internal configuration error for {agent_name}."}

    agent_instance = CrewAgent(**agent_base_config, tools=current_tools)

    chat_history = state.get('chat_history', [])
    formatted_history = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history[-3:]]) # Last 3 turns

    task_description = f"""Review the following recent conversation history:
<history>
{formatted_history}
</history>

Based on this history and your role, address the LATEST user query: '{query}'
"""
    if agent_name == "Customer":
        if state.get("customer_id"): # This is the explicit DB customer_id
            task_description += f"\nSpecific Context: If relevant and you have a tool for it, consider information for database customer ID: {state['customer_id']}."
        if retriever_tool_instance and "Placeholder" not in retriever_tool_instance.description:
            task_description += f"\nIMPORTANT: If the query relates to previously uploaded documents by this user/guest, you MUST prioritize using the '{retriever_tool_instance.name}' to find answers within those documents."
    
    if agent_name == "ProductSQL":
        task_description += "\nYour primary goal is to generate a SQL query based on the user's question about products. Use the 'ProductDatabaseSQLTool' to execute it. Then, explain the results to the user."


    expected_output = "Provide a comprehensive, accurate, and concise response directly addressing the user's query, strictly using information from your available tools and role. If using tools, clearly state what you found or if no information was found by the tool."

    task = Task(description=task_description, expected_output=expected_output, agent=agent_instance)

    try:
        logger.info(f"Executing CrewAI Task for {agent_name} with tools: {[t.name for t in agent_instance.tools]}")
        result = task.execute() # This is synchronous
        logger.info(f"Task for {agent_name} completed. Result snippet: {str(result)[:200]}...")
        return {"agent_response": str(result), "error": None}
    except Exception as e:
        logger.error(f"Error executing CrewAI task for {agent_name}: {e}", exc_info=True)
        return {"error": f"Agent '{agent_name}' task execution failed: {e}", "_last_error_agent": agent_name}

# Specific node functions
def company_agent_node(state: AgentState):
    return execute_agent_node(state, company_agent_def_base, "Company", static_tools=[search_tool])

def customer_agent_node(state: AgentState):
    return execute_agent_node(state, customer_agent_def_base, "Customer", static_tools=[])

def naive_agent_node(state: AgentState):
    return execute_agent_node(state, naive_agent_def_base, "Naive", static_tools=[search_tool])

# New Product SQL Agent Node - this uses the CrewAI agent paradigm
def product_sql_agent_node(state: AgentState):
    # The product_sql_agent_def_base already includes the sql_tool.
    # The agent's LLM, guided by its role and goal, should decide to use the tool.
    return execute_agent_node(state, product_sql_agent_def_base, "ProductSQL", static_tools=[])


class ReflectionOutput(BaseModel):
    feedback: str = Field(description="Constructive feedback on the response's relevance and correctness relative to the original query.")
    is_final_answer: bool = Field(description="True if the answer is satisfactory and directly addresses the original query, False otherwise.")

def reflection_node(state: AgentState):
    logger.info("--- Node: Reflection ---")
    if state.get("error") and "task execution failed" in state.get("error", ""):
        logger.warning(f"Reflecting on execution error: {state['error']}")
        return {"reflection": f"Processing failed: {state['error']}", "is_final": False, "retry_count": state.get('retry_count', 0) + 1}

    original_query = state['original_query']
    agent_response = state['agent_response']
    if not agent_response:
         logger.warning("No agent response to reflect on.")
         return {"reflection": "No response generated.", "is_final": True, "error": "Reflection: No agent response."}

    system_prompt = """You are a QA reviewer. Evaluate the AI's response based ONLY on the user's original query and the response.
    Assess: Relevance, Correctness, Completeness.
    Provide feedback. Is the answer final or needs retry?
    Output ONLY JSON: {"feedback": "...", "is_final_answer": <true|false>}""" # Simplified prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original Query: {original_query}\n\nAgent Response:\n{agent_response}\n\nEvaluate (JSON only):")
    ])
    reflection_chain = prompt | llm

    try:
        response_content = reflection_chain.invoke({"original_query": original_query, "agent_response": agent_response}).content
        json_str = extract_clean_json(response_content)
        logger.debug(f"Reflection Raw: {response_content}")
        if json_str:
            data = json.loads(json_str)
            logger.info(f"Reflection Parsed: {data}")
            return {"reflection": data.get('feedback', 'N/A'), "is_final": data.get('is_final_answer', False), "retry_count": state.get('retry_count', 0) + 1, "error": None}
        else:
            logger.warning(f"Reflection invalid JSON. Raw: {response_content}")
            return {"reflection": "Reflection format error.", "is_final": False, "retry_count": state.get('retry_count', 0) + 1, "error": "Reflection JSON error."}
    except Exception as e:
        logger.error(f"Reflection node error: {e}", exc_info=True)
        return {"error": f"Reflection failed: {e}", "is_final": False, "retry_count": state.get('retry_count', 0) + 1}


def question_generator_node(state: AgentState):
    logger.info("--- Node: Question Generator ---")
    original_query, agent_response = state['original_query'], state['agent_response']
    agent_type, db_customer_id = state['classified_agent'], state.get('customer_id')
    context_id = state.get('user_id') or state.get('guest_session_id')

    if not agent_response or len(agent_response) < 20:
        logger.warning("Response too short, skipping question gen.")
        return {"suggested_questions": []}

    system_prompt = """Generate 3 relevant follow-up questions based on query and response.
    Tailor to Agent Type. Output ONLY JSON: {"suggested_questions": ["q1", "q2", "q3"]}"""
    human_template = f"Query: {original_query}\nType: {agent_type}\nCtxID: {context_id}\nDbCustID: {db_customer_id}\nResponse: {agent_response}\nGen Questions (JSON only):"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_template)])
    chain = prompt | llm
    try:
        response_content = chain.invoke({}).content
        json_str = extract_clean_json(response_content)
        logger.debug(f"QGen Raw: {response_content}")
        if json_str:
            data = json.loads(json_str)
            qs = data.get('suggested_questions', [])
            logger.info(f"Generated {len(qs)} questions.")
            return {"suggested_questions": qs[:3]}
        else: # Fallback parsing
            logger.warning(f"QGen invalid JSON. Raw: {response_content}")
            questions = [line.strip() for line in response_content.split('\n') if '?' in line and len(line.strip()) > 10]
            return {"suggested_questions": questions[:3], "error": "QGen JSON error." if not questions else None}
    except Exception as e:
        logger.error(f"QGen node error: {e}", exc_info=True)
        return {"suggested_questions": [], "error": f"QGen failed: {e}"}


# --- Error Handler Node (as before) ---
# ... (copy your error_handler_node, ensure it uses logger) ...
def error_handler_node(state: AgentState):
    logger.info("--- Node: Error Handler ---")
    error_msg = state.get("error", "Unknown error")
    logger.error(f"Error Handler triggered: {error_msg}")
    response = state.get("agent_response", "")
    if response and len(response) > 30 and "failed" not in response.lower():
        logger.info("Error, but usable response found. Generating questions.")
        return {"error": error_msg, "agent_response": f"{response}\n\n[Note: Error during processing: {error_msg}]"}
    else:
        logger.warning("Error with no usable response. Ending workflow.")
        return {"error": error_msg, "is_final": True, "agent_response": f"Error: {error_msg}", "suggested_questions": []}


# --- Conditional Edges ---
def route_query(state: AgentState):
    logger.info(f"--- Edge: Routing Query ---")
    if state.get("error") and ("Failed to process query" in state["error"] or "Invalid JSON from rewriter" in state["error"]):
        logger.error(f"Routing to error_handler due to rewriter error: {state['error']}")
        return "error_handler"
    classification = state['classified_agent']
    logger.info(f"Routing based on classification: {classification}")
    if classification == "Company": return "execute_company"
    elif classification == "Customer": return "execute_customer"
    elif classification == "Naive": return "execute_naive"
    elif classification == "ProductSQL": return "execute_product_sql" # Added
    else: # Unknown or other error
        logger.warning(f"Unknown classification '{classification}'. Routing to reflect.")
        state["error"] = f"Cannot route: Unknown classification '{classification}'."
        state["agent_response"] = "I'm unsure how to handle that. Could you rephrase?"
        return "reflect" # Let reflection decide or lead to error handler

def reflection_router(state: AgentState):
    logger.info(f"--- Edge: Reflection Router ---")
    max_retries, retries = 2, state.get('retry_count', 0)
    if state.get("error") and ("task execution failed" in state["error"] or "Reflection processing failed" in state["error"]):
        logger.error(f"Error post-execution/during-reflection: {state['error']}")
        return "error_handler" if retries >= max_retries else "rewrite_query"
    if state.get('is_final', False):
        logger.info("Reflection: Final. To question generator.")
        return "generate_questions"
    elif retries >= max_retries:
        logger.warning(f"Max retries ({retries}) reached. To question generator with current best.")
        state["agent_response"] += f"\n\n[System: Max retries. Last feedback: {state.get('reflection', 'N/A')}]"
        state["is_final"] = True # Mark as final now
        return "generate_questions"
    else:
        logger.info(f"Reflection: Retry ({retries}/{max_retries}). To rewriter.")
        return "rewrite_query"

def error_final_route(state: AgentState):
    logger.info(f"--- Edge: Error Final Route ---")
    return END if state.get("is_final", False) else "generate_questions"


# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("execute_company", company_agent_node)
workflow.add_node("execute_customer", customer_agent_node)
workflow.add_node("execute_naive", naive_agent_node)
workflow.add_node("execute_product_sql", product_sql_agent_node) # Added
workflow.add_node("reflect", reflection_node)
workflow.add_node("generate_questions", question_generator_node)
workflow.add_node("error_handler", error_handler_node)

workflow.set_entry_point("rewrite_query")
workflow.add_conditional_edges("rewrite_query", route_query, {
    "execute_company": "execute_company", "execute_customer": "execute_customer",
    "execute_naive": "execute_naive", "execute_product_sql": "execute_product_sql", # Added
    "reflect": "reflect", "error_handler": "error_handler"
})
workflow.add_edge("execute_company", "reflect")
workflow.add_edge("execute_customer", "reflect")
workflow.add_edge("execute_naive", "reflect")
workflow.add_edge("execute_product_sql", "reflect") # Added
workflow.add_conditional_edges("reflect", reflection_router, {
    "generate_questions": "generate_questions", "rewrite_query": "rewrite_query",
    "error_handler": "error_handler"
})
workflow.add_conditional_edges("error_handler", error_final_route, {
    "generate_questions": "generate_questions", END: END
})
workflow.add_edge("generate_questions", END)

checkpointer = InMemorySaver()
app_agent = workflow.compile(checkpointer=checkpointer)
logger.info("LangGraph Agent System Compiled with ProductSQL agent.")


# --- Agent Runner Function (for API) ---
def run_agent_sync(
    query: str, thread_id: str, user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None,
    chat_history: Optional[List[Tuple[str, str]]] = None
) -> AgentState:
    initial_state = AgentState(
        original_query=query, user_id=user_id, guest_session_id=guest_session_id,
        customer_id=None, rewritten_query="", classified_agent="Unknown",
        agent_response="", reflection="", is_final=False, error=None,
        retry_count=0, suggested_questions=[], chat_history=chat_history or []
    )
    logger.info(f"Running agent: Q='{query}' User={user_id}, Guest={guest_session_id}, Thread={thread_id}, HistLen={len(initial_state['chat_history'])}")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        final_state = app_agent.invoke(initial_state, config, recursion_limit=15)
        logger.info(f"Agent finished for thread {thread_id}. Final class: {final_state.get('classified_agent')}")
        logger.debug(f"Final State (Thread {thread_id}): {final_state}")
        if final_state.get('error'): logger.error(f"Agent Error (Thread {thread_id}): {final_state['error']}")
        return final_state
    except Exception as e:
        logger.error(f"Unhandled agent invocation error (Thread {thread_id}): {e}", exc_info=True)
        error_state = initial_state.copy()
        error_state.update({'error': f"Agent invocation error: {e}", 'agent_response': "Critical error.", 'is_final': True, 'suggested_questions': []})
        return error_state


# Standalone test (comment out for API use)
if __name__ == "__main__":
    logger.info("--- Running Standalone Agent Test ---")
    test_thread = "standalone_test_001"

    # Test ProductSQL Agent
    # Make sure product_database.db exists and is populated from setup_product_db.py
    # product_queries = [
    #     "How many laptops are in stock?",
    #     "List all products under $50",
    #     "What is the price of the 'Wireless Mouse'?",
    #     "Show me red t-shirts",
    #     "are there any blue hoodies in size L?"
    # ]
    # for q in product_queries:
    #     print(f"\n--- Testing ProductSQL Query: {q} ---")
    #     response = run_agent_sync(query=q, thread_id=test_thread)
    #     print(f"Response: {response.get('agent_response')}")
    #     print(f"Suggested: {response.get('suggested_questions')}")
    #     print(f"Error: {response.get('error')}")
    #     print("------------------------------------")

    # Test Company Agent
    # print("\n--- Testing Company Query ---")
    # response = run_agent_sync(query="What is Google's main business?", thread_id=test_thread)
    # print(f"Response: {response.get('agent_response')}")
    # print(f"Suggested: {response.get('suggested_questions')}")

    # Test Naive Agent
    # print("\n--- Testing Naive Query ---")
    # response = run_agent_sync(query="What is the capital of Canada?", thread_id=test_thread)
    # print(f"Response: {response.get('agent_response')}")
    # print(f"Suggested: {response.get('suggested_questions')}")

    # Test Customer Agent (without documents initially - it should use search or say it can't answer)
    # print("\n--- Testing Customer Query (no docs, no specific cust_id for DB) ---")
    # response = run_agent_sync(query="Tell me about my recent activity", guest_session_id="guest_test_sql", thread_id=test_thread)
    # print(f"Response: {response.get('agent_response')}")
    # print(f"Suggested: {response.get('suggested_questions')}")
    # This might be classified as Naive or Unknown by the rewriter if "my" isn't strong enough.
    # Or if classified as Customer, it would likely use search_tool or state it needs more info / can't find docs.

    logger.info("--- Standalone Agent Test Finished ---")