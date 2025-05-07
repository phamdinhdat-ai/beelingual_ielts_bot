# app/agent_system.py
import os
from typing import Optional, TypedDict, Literal, List, Tuple # Add Tuple if using chat history
import loguru
from loguru import logger
import json
import asyncio # Import asyncio for potential async tool calls

# --- Framework Imports ---
from crewai import Agent as CrewAgent, Task
from crewai_tools import SerperDevTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver # Use if needed
from crewai import Crew
# --- LangChain Imports ---
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from pydantic import BaseModel, Field

# --- Application Imports ---
from agent.utils.helper_function import extract_clean_json
from agent.vector_store_manager import create_retriever_tool # Ensure this is correctly implemented
# from agent.tools import RetrieveTool, IngestTool # Re-importing if needed

# --- Configuration & Logging ---
from dotenv import load_dotenv
load_dotenv()
logger.add("multi_agents.log", rotation="1 MB", retention="10 days", level="DEBUG")
logger.info("Starting agent_system.py")
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "YOUR_SERPER_KEY_HERE") # Load from env
# os.environ["OPENAI_API_KEY"] = "NA" # Set as needed

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")

# --- Initialize LLM ---
try:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7, max_tokens=2000)
    logger.info(f"LLM '{LLM_MODEL}' loaded.")
except Exception as e:
    logger.error(f"Failed to load LLM '{LLM_MODEL}': {e}", exc_info=True)
    raise # Stop if LLM fails

# --- Initialize Static Tools ---
try:
    search_tool = SerperDevTool(n_results=3) # Increased results slightly
    logger.info("Search tool initialized.")
except Exception as e:
     logger.error(f"Failed to initialize SerperDevTool: {e}. Check SERPER_API_KEY.", exc_info=True)
     # Provide a dummy tool if search fails?
     def dummy_search(query: str): return "Search tool unavailable."
     search_tool = Tool(name="search", func=dummy_search, description="Search tool unavailable.")


# --- Agent State Definition ---
class AgentState(TypedDict):
    original_query: str
    user_id: Optional[int] # Added for API integration
    guest_session_id: Optional[str] # Added for API integration
    customer_id: Optional[str] # Extracted customer ID for DB lookups (separate from user/guest ID)
    rewritten_query: str
    classified_agent: Literal["Company", "Customer", "Naive", "Unknown"]
    agent_response: str
    reflection: str
    is_final: bool
    error: Optional[str]
    retry_count: int
    suggested_questions: List[str]
    # Optional: Add chat history context
    # chat_history: List[Tuple[str, str]]

logger.info("LangGraph State Defined.")


# --- CrewAI Agent Definitions (Base Config) ---
# Store base configs, tools will be added dynamically where needed
from langchain_openai import ChatOpenAI
ollama_llm = ChatOpenAI(
    model = "ollama/deepseek-r1:1.5b",
    base_url = "http://localhost:11434/v1",
    temperature = 0.7,
    max_tokens = 2000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

company_agent_def_base = {
    'role': 'Company Information Specialist',
    'goal': 'Provide accurate information about the company (e.g., Google) including location, products, services, history, and mission, based on search results and internal knowledge.',
    'backstory': 'You are an AI assistant dedicated to representing the company accurately. Use search tools to find the latest public information.',
    'llm': ollama_llm,
    'verbose': True, # Verbose for debugging
    'allow_delegation': False
}

customer_agent_def_base = {
    'role': 'Customer Support Agent',
    'goal': 'Answer customer-specific questions by retrieving their information from the customer database OR provided documents. Handle queries about plans, history, tickets, or document content.',
    'backstory': 'You are a helpful support agent. Use the customer ID provided for DB lookups OR use the document retriever tool if the query relates to uploaded documents.',
    'llm': ollama_llm,
    'verbose': True,
    'allow_delegation': False
    # Tools added dynamically in the node
}

naive_agent_def_base = {
    'role': 'General Knowledge Assistant',
    'goal': 'Answer general knowledge questions or queries that do not fall under specific company or customer information categories. Use search tools to find relevant information online.',
    'backstory': 'You are a general-purpose AI assistant capable of answering a wide range of topics by searching the internet.',
    'llm': ollama_llm,
    'verbose': True,
    'allow_delegation': False
}

product_agent_def_base = {
    'role': 'Product Information Specialist',
    'goal': 'Provide detailed information about specific products, including features, specifications, and comparisons.',
    'backstory': 'You are an AI assistant specialized in product knowledge. Use search tools to find the latest product information.',
    'llm': ollama_llm,
    'verbose': True,
    'allow_delegation': False
}


agent_definitions = {
    'company': company_agent_def_base,
    'customer': customer_agent_def_base,
    'naive': naive_agent_def_base,
    'product': product_agent_def_base
}

logger.info("Product Agent Base Definition Ready.")
logger.info("CrewAI Agent Base Definitions Ready.")


# --- LangGraph Nodes ---

# Node 1: Rewriter (Copy from your multi_agents.py, ensure logger is used)
def rewrite_query_node(state: AgentState):
    """Rewrites the user query and classifies which agent should handle it."""
    logger.info("--- Node: Rewriter ---")
    # Clear previous errors/responses if retrying
    state['error'] = None
    state['agent_response'] = ""
    state['reflection'] = "" # Clear reflection too
    state['suggested_questions'] = [] # Clear questions
    is_user_context = state.get("user_id") is not None
    is_guest_context = state.get("guest_session_id") is not None

    query = state['original_query']

    system_prompt = """You are an expert query processor. Your tasks are:
    1.  Correct any spelling mistakes in the user query.
    2.  Rephrase the query for maximum clarity if necessary.
    3.  Classify the query's intent to determine the best agent to handle it:
        - 'Company': For questions about a specific company's details (location, products, services, mission etc.). Assume the company is 'Google' if not specified.
        - 'Customer': For questions related to a specific customer's account, history, plan, support tickets etc. These queries MUST contain or imply a customer ID.
        - 'Naive': For general knowledge questions, queries unrelated to the company or a specific customer.
        - 'Unknown': If the query is ambiguous or cannot be clearly classified.
    4. Extract any customer ID mentioned (like cust123, user45, id: 7890). Return null if no ID is found.
    Provide your output in the specified JSON format.
    For example, for the query 'What is the status of my order with customer ID cust123?', your output should be:
    <output>
    {{
        "rewritten_query": "What is the status of my order?",
        "agent_classification": "Customer",
        "extracted_customer_id": "cust123"
    }}
    </output>
    If no customer ID is found, return null for that field.
    If the query is not clear or cannot be classified, return 'Unknown' for the agent classification and null for the customer ID.
    Your output should be a JSON object with the following fields:

    {{
        "rewritten_query": "<corrected and clarified query>",
        "agent_classification": "<Company|Customer|Naive|Unknown>",
        "extracted_customer_id": "<customer ID or null>"
        
    }}

    **CRITICAL:** You MUST format your output ONLY as a JSON object that strictly adheres to the following schema. Do NOT include any other text, explanations, or markdown formatting like ```json ``` around the JSON object itself."""


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Process the following user query: {query}")
    ])

    rewriter_chain = prompt | llm

    try:
        response = rewriter_chain.invoke({"query": query})
        json_str = extract_clean_json(response.content)
        if json_str:
            response_data = json.loads(json_str)
            logger.debug(f"Rewriter Raw Output: {response.content}")
            logger.info(f"Rewriter Parsed JSON: {response_data}")

            classification = response_data.get('agent_classification', 'Unknown')
            # --- Logic Adjustment: If user/guest context exists, lean towards Customer ---
            if (is_user_context or is_guest_context) and classification != 'Company':
                 # If user/guest has uploaded docs, queries are likely about them or generic customer service
                 if classification == 'Naive' or classification == 'Unknown':
                      logger.info(f"User/Guest context present, overriding classification from '{classification}' to 'Customer'.")
                      classification = 'Customer'

            # Validate classification
            valid_class = {"Company", "Customer", "Naive", "Unknown"}
            if classification not in valid_class:
                 logger.warning(f"Invalid classification '{classification}' received. Defaulting to Unknown.")
                 classification = "Unknown"

            return {
                "rewritten_query": response_data.get('rewritten_query', query),
                "classified_agent": classification,
                 # Prioritize extracted DB ID, keep existing state ID otherwise (user/guest)
                "customer_id": response_data.get('extracted_customer_id'), # This is for explicit DB lookup ID
                 # user_id and guest_session_id remain unchanged in the state
                "error": None
            }
        else:
            logger.warning(f"Rewriter output was not valid JSON. Raw: {response.content}")
            # Fallback logic if JSON fails
            classification = "Customer" if (is_user_context or is_guest_context) else "Unknown"
            return {"rewritten_query": query, "classified_agent": classification, "error": "Invalid JSON from rewriter."}
    except Exception as e:
        logger.error(f"Error in Rewriter node: {e}", exc_info=True)
        classification = "Customer" if (is_user_context or is_guest_context) else "Unknown"
        return {"error": f"Failed to process query: {e}", "classified_agent": classification, "rewritten_query": query}


# --- Node 2, 3, 4: Agent Execution ---
# Uses dynamic tools for Customer Agent

def execute_agent_node(
    state: AgentState,
    agent_base_config: dict,
    agent_name: str,
    static_tools: Optional[List[Tool]] = None
):
    """Executes a CrewAI Task using dynamically configured agent and tools."""
    logger.info(f"--- Node: Execute {agent_name} Agent Task ---")
    if state.get("error") and "Failed to process query" in state["error"]: # Check for critical rewriter error
         logger.warning("Skipping agent execution due to critical rewriter error.")
         return {}
    if state.get("error") and agent_name == state.get("_last_error_agent"): # Avoid immediate re-run if *this* agent failed last time
         logger.warning(f"Skipping {agent_name} execution due to error in previous attempt.")
         return {}

    state["error"] = None # Clear previous errors

    query = state['rewritten_query']
    user_id = state.get('user_id')
    guest_session_id = state.get('guest_session_id')
    db_customer_id = state.get('customer_id') # Explicit ID for DB lookup

    # --- Dynamic Tool Loading (Especially for Customer Agent) ---
    current_tools = list(static_tools) if static_tools else []
    retriever_tool = None
    collection_name = None

    if agent_name == "Customer":
        if user_id:
            collection_name = f"user_{user_id}"
            logger.info(f"Creating retriever tool for user collection: {collection_name}")
            retriever_tool = create_retriever_tool(collection_name=collection_name)
        elif guest_session_id:
            collection_name = f"guest_{guest_session_id.replace('-', '_')}"[:63] # Ensure valid length
            logger.info(f"Creating retriever tool for guest collection: {collection_name}")
            retriever_tool = create_retriever_tool(collection_name=collection_name)

        if retriever_tool:
            current_tools.append(retriever_tool)
        else:
            logger.warning("No user/guest ID provided or failed to create retriever tool for Customer Agent.")
            # Agent will run without document retrieval capability

    # Add standard search tool if not already present (e.g., for Customer agent)
    if search_tool and search_tool.name not in [t.name for t in current_tools]:
         current_tools.append(search_tool)

    # Create Agent instance with the right tools for this run
    agent_instance = CrewAgent(**agent_base_config)

    # --- Task Definition ---
    task_description = f"User Query: '{query}'"
    if agent_name == "Customer":
        if db_customer_id:
            task_description += f"\nContext: Check internal database for customer ID: {db_customer_id} if relevant and tools allow."
        if retriever_tool and "Placeholder" not in retriever_tool.description:
            task_description += f"\nContext: You MUST prioritize searching the uploaded documents using the '{retriever_tool.name}' tool for information relevant to the query."
        elif user_id or guest_session_id:
             task_description += "\nContext: User/guest might have documents, but no specific document retriever tool is available (maybe none uploaded/indexed yet)."

    expected_output = "Provide a concise and accurate response based ONLY on the information retrieved from your available tools (search, document retriever) and your internal knowledge. Address the user's query directly."

    task = Task(
        name=f"{agent_name} Task",
        
        description=task_description,
        expected_output=expected_output,
        agent=agent_instance,
        tools=current_tools,
        # async_execution=True, # Set to True for async execution
        output_file=f"{agent_name}_output.md", 
        
    )
    # --- Execute Task ---
    try:
        logger.info(f"Executing CrewAI Task for {agent_name} with tools: {[t.name for t in current_tools]}")
        # crew_agent = Crew(agent=agent_instance, tools=current_tools)
        
        # async def execute_task(task: Task):
        #     # Use async execution
        #     return  agent_instance.execute_task(task)
        # result = asyncio.run(execute_task(task)) # Run the task asynchronously
        result = agent_instance.execute_task(task) # Run the task synchronously
        logger.info(f"Task execution completed for {agent_name}. Result snippet: {str(result)[:200]}...")
        return {"agent_response": str(result), "error": None}
    except Exception as e:
        logger.error(f"Error executing CrewAI task for {agent_name}: {e}", exc_info=True)
        # Record which agent failed to prevent immediate retry loop on persistent error
        return {"error": f"Agent '{agent_name}' task execution failed: {e}", "_last_error_agent": agent_name}


# Specific node functions calling the generic executor
def company_agent_node(state: AgentState):
    return execute_agent_node(state, company_agent_def_base, "Company", static_tools=[search_tool])

def customer_agent_node(state: AgentState):
    # Tools are determined dynamically inside execute_agent_node
    return execute_agent_node(state, customer_agent_def_base, "Customer", static_tools=[search_tool]) # Pass empty list, tools added inside

def naive_agent_node(state: AgentState):
    return execute_agent_node(state, naive_agent_def_base, "Naive", static_tools=[search_tool])



# Ensure it uses logger and extract_clean_json
class ReflectionOutput(BaseModel):
    """Structured output for the Reflection node."""
    feedback: str = Field(description="Constructive feedback on the response's relevance and correctness relative to the original query.")
    is_final_answer: bool = Field(description="True if the answer is satisfactory and directly addresses the original query, False otherwise.")

def reflection_node(state: AgentState):
    """Reflects on the generated answer, checking relevance and correctness."""
    logger.info("--- Node: Reflection ---")
    # If error occurred during agent execution, reflect on that state
    if state.get("error") and "task execution failed" in state.get("error", ""):
        logger.warning(f"Reflecting on execution error: {state['error']}")
        return {
            "reflection": f"Processing failed due to agent execution error: {state['error']}",
            "is_final": False, # Allow retry unless max retries hit
            "retry_count": state.get('retry_count', 0) + 1 # Still counts as an attempt
            # Keep the error state
            }

    original_query = state['original_query']
    agent_response = state['agent_response']

    if not agent_response:
         logger.warning("No agent response to reflect on.")
         return {"reflection": "No response generated.", "is_final": True, "error": "Reflection failed: No agent response found."}

    system_prompt = """You are a meticulous quality assurance reviewer. Your task is to evaluate an AI agent's response based on the user's original query.
                    Assess the following:
                    1.  **Relevance:** Does the response directly address the user's original question?
                    2.  **Correctness:** Is the information likely correct (based on general knowledge or provided context)? You don't need to verify external facts exhaustively, but check for obvious flaws or contradictions.
                    3.  **Completeness:** Does the response sufficiently answer the question?

                    Provide  constructive feedback and determine if the answer is final (good enough) or needs revision/retry. Use the specified JSON format.
                    For example, if the original query was 'What is the capital of France?' and the agent response was 'The capital of France is Paris.', your output should be:
                    <output>
                    {{
                        "feedback": "The response is relevant, correct, and complete. It directly answers the question.",
                        "is_final_answer": true
                    }}
                    </output>
                    If the original query was 'What is the capital of France?' and the agent response was 'The capital of France is Berlin.', your output should be:
                    <output>
                    {{
                        "feedback": "The response is relevant but incorrect. The capital of France is Paris, not Berlin.",
                        "is_final_answer": false
                    }}
                    </output>
                    If the original query was 'What is the capital of France?' and the agent response was 'I don't know.', your output should be:
                    <output>
                    {{
                        "feedback": "The response is not relevant and does not answer the question. It should provide the correct information.",
                        "is_final_answer": false
                    }}
                    </output>
                    Your output should be a JSON object with the following fields:
                    {{
                        "feedback": "<constructive feedback on relevance, correctness, and completeness>",
                        "is_final_answer": <true|false>
                    }}
                    **CRITICAL:** You MUST format your output ONLY as a JSON object that strictly adheres to the above schema. Do NOT include any other text, explanations, or markdown formatting like ```json ``` around the JSON object itself.
                    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original Query: {original_query}\n\nAgent Response:\n{agent_response}\n\nPlease evaluate and provide feedback. Your output should be a JSON object as specified above.")
    ])
    reflection_chain = prompt | llm

    try:
        response = reflection_chain.invoke({
            "original_query": original_query,
            "agent_response": agent_response
        })
        json_str = extract_clean_json(response.content)
        logger.debug(f"Reflection Raw Output: {response.content}")
        if json_str:
            response_data = json.loads(json_str)
            logger.info(f"Reflection Parsed JSON: {response_data}")
            return {
                "reflection": response_data.get('feedback', 'No feedback provided.'),
                "is_final": response_data.get('is_final_answer', False),
                "retry_count": state.get('retry_count', 0) + 1,
                "error": None # Clear execution error if reflection succeeds
            }
        else:
            logger.warning(f"Reflection output was not valid JSON. Raw: {response.content}")
            # Decide fallback - assume needs retry if reflection fails?
            return {"reflection": "Reflection failed (invalid format). Assuming retry needed.", "is_final": False, "retry_count": state.get('retry_count', 0) + 1, "error": "Invalid JSON from reflection."}
    except Exception as e:
        logger.error(f"Error in Reflection node: {e}", exc_info=True)
        return {"error": f"Reflection processing failed: {e}", "is_final": False, "retry_count": state.get('retry_count', 0) + 1}


# Ensure it uses logger and extract_clean_json
def question_generator_node(state: AgentState):
    """Generates relevant follow-up questions based on previous query and response context."""
    logger.info("--- Node: Question Generator ---")

    original_query = state['original_query']
    agent_response = state['agent_response']
    agent_type = state['classified_agent']
    db_customer_id = state.get('customer_id') # DB specific ID
    context_id = state.get('user_id') or state.get('guest_session_id') # User/Guest ID

    if not agent_response or len(agent_response) < 20: # Avoid generating for short/error responses
        logger.warning("Agent response too short or missing, skipping question generation.")
        return {"suggested_questions": []}

    system_prompt = """You are an expert question generator. Your task is to generate 3-5 highly relevant follow-up questions 
    based on the original query and the response provided.
    
    The questions should:
    1. Help explore the topic in more depth
    2. Be directly related to the response content
    3. Include specific data points mentioned in the response
    4. Focus on the most interesting or valuable aspects
    5. Be tailored to the agent type (Company, Customer, or General Knowledge)
    
    For Company questions: Focus on products, services, history, competition, or market position
    For Customer questions: Focus on account details, usage patterns, recommendations, or support issues
    For General Knowledge: Focus on broader context, practical applications, or recent developments
    
    
    For example, if the original query was "What is the mission of Google?" and the response was "Google's mission is to organize the world's information and make it universally accessible and useful.", your output should be:
    <output>
    {{
        "suggested_questions": [
            "What are some recent initiatives by Google to improve accessibility?",
            "How does Google ensure the accuracy of the information it organizes?",
            "What challenges does Google face in making information universally accessible?",
            "How does Google's mission impact its product development?",
            "What are the implications of Google's mission for user privacy?"]
    }}
    </output>
    If the original query was "What is the status of my order with customer ID cust123?" and the response was "Your order is being processed and will be shipped soon.", your output should be:
    <output>
    {{
        "suggested_questions": [
            "What is the estimated delivery date for my order?",
            "Can I change the shipping address for my order?",
            "What should I do if my order doesn't arrive on time?",
            "How can I track my order status?",
            "What are the return policies for my order?"
        ]
    }}
    </output>
    Your output should be a JSON object with the following fields:
    {{
        "suggested_questions": [
            "<question 1>",
            "<question 2>",
            "<question 3>"
        ]
    }}
    **NOTE:** The questions should be relevant to the context of the original query and the agent type.
    **CRITICAL:** You MUST format your output ONLY as a JSON object that strictly adheres to the above schema. Do NOT include any other text, explanations, or markdown formatting like ```json ``` around the JSON object itself.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"""Original Query: {original_query}
                
        Agent Type: {agent_type}
        {f'Customer ID: {db_customer_id}' if db_customer_id else ''}

        Agent Response: 
        {agent_response}

        Please generate relevant follow-up questions based on this context.""")
    ])
    
    question_chain = prompt | llm

    try:
        response = question_chain.invoke({
            "original_query": original_query,
            "agent_response": agent_response,
            "agent_type": agent_type,
            "customer_id": db_customer_id
        })
        json_str = extract_clean_json(response.content)
        logger.debug(f"Question Gen Raw Output: {response.content}")
        if json_str:
            response_data = json.loads(json_str)
            questions = response_data.get('suggested_questions', [])
            logger.info(f"Generated {len(questions)} follow-up questions.")
            return {"suggested_questions": questions[:3]} # Limit to 3
        else:
            logger.warning(f"Question generator output was not valid JSON. Raw: {response.content}")
            return {"suggested_questions": [], "error": "Invalid JSON from question generator."} # No questions on format error
    except Exception as e:
        logger.error(f"Error in question generator node: {e}", exc_info=True)
        return {"suggested_questions": [], "error": f"Question generation failed: {e}"}


def error_handler_node(state: AgentState):
    """Handles errors, decides if question generation is still possible."""
    logger.info("--- Node: Error Handler ---")
    error_message = state.get("error", "Unknown error")
    logger.error(f"Entering error handler with error: {error_message}")

    agent_response = state.get("agent_response", "")

    # Check if there's a somewhat usable response despite the error
    if agent_response and len(agent_response) > 30 and "failed" not in agent_response.lower():
        logger.info("Error occurred, but attempting question generation based on partial response.")
        # Keep the error, modify response slightly, allow flow to question gen
        return {
            "error": error_message,
            "agent_response": f"{agent_response}\n\n[Note: An error occurred during processing: {error_message}]",
            # is_final will be determined by whether question gen runs
        }
    else:
        logger.warning("Error occurred with no usable response. Ending workflow.")
        # Set final state with error message
        return {
            "error": error_message,
            "is_final": True,
            "agent_response": f"Sorry, an error occurred: {error_message}",
            "suggested_questions": []
        }

def route_query(state: AgentState):
    """Routes to the appropriate agent node based on classification."""
    logger.info(f"--- Conditional Edge: Routing Query ---")
    # Check for critical errors from rewriter first
    if state.get("error") and ("Failed to process query" in state["error"] or "Invalid JSON from rewriter" in state["error"]):
        logger.error(f"Routing to error handler due to rewriter error: {state['error']}")
        return "error_handler"

    classification = state['classified_agent']
    logger.info(f"Routing based on classification: {classification}")

    if classification == "Company":
        return "execute_company"
    elif classification == "Customer":
        # Check for user/guest context or explicit customer ID
        if state.get('user_id') or state.get('guest_session_id') or state.get('customer_id'):
            return "execute_customer"
        else:
            logger.warning("Customer classification chosen, but no user/guest/DB context found. Routing to reflect.")
            state["error"] = "Customer agent requires user context or a specific Customer ID."
            state["agent_response"] = "Please log in or provide a specific Customer ID for this type of query."
            return "reflect"
    elif classification == "Naive":
        return "execute_naive"
    else: # Unknown
        logger.warning(f"Routing to reflect due to Unknown classification.")
        state["error"] = f"Could not determine the right agent (classification: {classification})."
        state["agent_response"] = "I'm not sure how to handle that query. Could you rephrase?"
        return "reflect"

def reflection_router(state: AgentState):
    """Routes after reflection: retry, generate questions, or handle error."""
    logger.info(f"--- Conditional Edge: Reflection Router ---")
    max_retries = 2
    retry_count = state.get('retry_count', 0) # Retry count increments *in* reflection node

    # Prioritize checking for execution errors noted during reflection
    if state.get("error") and ("task execution failed" in state["error"] or "Reflection processing failed" in state["error"]):
         if retry_count >= max_retries:
              logger.error(f"Max retries reached after execution/reflection error. Routing to error handler. Error: {state['error']}")
              return "error_handler" # Let error handler decide final state
         else:
              logger.warning(f"Execution/reflection error occurred, attempting retry ({retry_count}/{max_retries}). Routing to rewrite_query. Error: {state['error']}")
              return "rewrite_query" # Retry if under limit

    is_final = state.get('is_final', False) # Get decision from reflection node

    if is_final:
        logger.info("Reflection approved. Routing to generate questions.")
        return "generate_questions"
    elif retry_count >= max_retries:
        logger.warning(f"Max retries ({max_retries}) reached after reflection feedback. Using current response for question generation.")
        state["agent_response"] += f"\n\n[System Note: Max retries reached. Last feedback: {state.get('reflection', 'N/A')}]"
        # Mark as final *before* generating questions based on this final attempt
        state["is_final"] = True
        return "generate_questions"
    else: # Needs retry and under limit
        logger.info(f"Reflection suggests retry ({retry_count}/{max_retries}). Routing back to rewrite_query.")
        return "rewrite_query"

def error_final_route(state: AgentState):
     """Determines final routing after error handler."""
     # If error handler decided it's final, end. Otherwise, go generate questions.
     if state.get("is_final", False):
          logger.info("Error handler marked state as final. Ending.")
          return END
     else:
          logger.info("Error handler allows proceeding to question generation.")
          return "generate_questions"


# --- Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("execute_company", company_agent_node)
workflow.add_node("execute_customer", customer_agent_node)
workflow.add_node("execute_naive", naive_agent_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("generate_questions", question_generator_node)
workflow.add_node("error_handler", error_handler_node)

workflow.set_entry_point("rewrite_query")

workflow.add_conditional_edges("rewrite_query", route_query, {
    "execute_company": "execute_company",
    "execute_customer": "execute_customer",
    "execute_naive": "execute_naive",
    "reflect": "reflect",
    "error_handler": "error_handler"
})

workflow.add_edge("execute_company", "reflect")
workflow.add_edge("execute_customer", "reflect")
workflow.add_edge("execute_naive", "reflect")

workflow.add_conditional_edges("reflect", reflection_router, {
    "generate_questions": "generate_questions",
    "rewrite_query": "rewrite_query",
    "error_handler": "error_handler"
})

# Route from error handler: either END or generate questions
workflow.add_conditional_edges("error_handler", error_final_route, {
    "generate_questions": "generate_questions",
    END: END
})

workflow.add_edge("generate_questions", END) # Final step

# Compile the graph
checkpointer = InMemorySaver() # Use checkpointer if needed for state persistence between calls
app_agent = workflow.compile(checkpointer=checkpointer)

logger.info("LangGraph Agent System Compiled.")




def run_agent_sync(
    query: str,
    thread_id: str, # Use thread_id for conversation memory with checkpointer
    user_id: Optional[int] = None,
    guest_session_id: Optional[str] = None
    # chat_history: Optional[List[Tuple[str, str]]] = None # Pass history if needed
    ) -> AgentState:
    """Runs the agentic system synchronously for API calls."""

    initial_state = AgentState(
        original_query=query,
        user_id=user_id,
        guest_session_id=guest_session_id,
        customer_id=None, # Rewriter extracts this if present
        rewritten_query="",
        classified_agent="Unknown",
        agent_response="",
        reflection="",
        is_final=False,
        error=None,
        retry_count=0,
        suggested_questions=[],
        # chat_history=chat_history or []
    )

    logger.info(f"Running agent for query: '{query}' [User: {user_id}, Guest: {guest_session_id}, Thread: {thread_id}]")
    config = {"configurable": {"thread_id": 1}}

    try:
        # Use sync invoke
        final_state = app_agent.invoke(initial_state, config) # Increased limit slightly
        logger.info(f"Agent invocation finished for thread {thread_id}.")
        # Log final details before returning
        logger.debug(f"Final State for thread {thread_id}: {final_state}")
        if final_state.get('error'): logger.error(f"Agent finished with error for thread {thread_id}: {final_state['error']}")
        return final_state
    except Exception as e:
        logger.error(f"Unhandled error invoking agent for thread {thread_id}: {e}", exc_info=True)
        # Return a state indicating catastrophic failure
        error_state = initial_state.copy() # Don't modify original initial_state
        error_state['error'] = f"Agent invocation failed catastrophically: {e}"
        error_state['agent_response'] = "Sorry, a critical error occurred. Please contact support."
        error_state['is_final'] = True
        error_state['suggested_questions'] = []
        return error_state

# Example test call (won't run automatically in API context)
if __name__ == "__main__":
    logger.info("Running standalone test...")
    test_thread_id = "1"
    # Example 1: Company query
    run_agent_sync("What is Google's mission?", thread_id=test_thread_id)
    # Example 2: Naive query
    # run_agent_sync("Who invented the lightbulb?", thread_id=test_thread_id)
    # Example 3: Guest query (no docs initially)
    # run_agent_sync("Tell me about my account", guest_session_id="guest_abc", thread_id=test_thread_id)
    # Example 4: User query (assuming user 1 exists, no docs initially)
    # run_agent_sync("Summarize my documents", user_id=1, thread_id=test_thread_id)
    # Example 5: Query needing specific DB ID (will likely fail w/o DB tool)
    # run_agent_sync("What's the status for customer cust123?", thread_id=test_thread_id)
    logger.info("Standalone test finished.")