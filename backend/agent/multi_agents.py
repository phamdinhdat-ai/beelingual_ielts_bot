import os
from typing import Optional, TypedDict, Literal, List
import loguru
from loguru import logger
import json


# Import CrewAI's  class
from crewai.tools import BaseTool
from agent.utils.helper_function import extract_clean_json
from agent.tools.retrieve_tool import RetrieveTool, IngestTool
# from agent.tools.search_tool import SearchTool
# from agent.tools.nl2sql_tool import NL2SQLTool

# import langchain ollama
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import Tool
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from crewai_tools import SerperDevTool
# from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from crewai import Agent as CrewAgent # Alias to avoid confusion with node names

from pydantic import BaseModel, Field
# from langchain_core.output_parsers import StructuredOutputParser
from dotenv import load_dotenv
load_dotenv() # Load environment variables if needed (e.g., API keys for other tools)
os.environ["SERPER_API_KEY"] = "a6ce4b5fc4ef3f9754144d519ecf9e418bc1c7bc"
os.environ["OPENAI_API_KEY"] = "NA"

TEST_COLLECTION_NAME = "agent_test_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Ensure this model is available locally
PERSIST_DIR = "./_agent_test_chroma_db"
LLM_MODEL = "mistral:latest" # Or your preferred Ollama model
# llm = ChatOllama(model='deepseek-r1:1.5b', temperature=0.2, max_tokens=2000)
logger.add("multi_agents.log", rotation="1 MB", retention="10 days", level="DEBUG")
logger.info("Starting multi_agents.py")

def setup_environment() -> tuple:
    """Initializes LLM, Tools, and cleans up old DB."""
    logger.info("--- Setting up test environment ---")

    try:
        # Initialize LLM
        logger.info(f"Loading LLM: {LLM_MODEL}")

        # llm = ChatOpenAI(
        #     model = "ollama/deepseek-r1:1.5b",
        #     base_url = "http://localhost:11434/v1",
        #     temperature = 0.7,
        #     max_tokens = 2000)
        llm = ChatOllama(model=LLM_MODEL, temperature=0.7, max_tokens=2000)
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

class AgentState(TypedDict):
    original_query: str
    customer_id: Optional[str]
    rewritten_query: str
    classified_agent: Literal["Company", "Customer", "Naive", "Unknown"]
    assigned_agent: Optional[str]  # Optional field for assigned agent
    agent_response: str
    reflection: str
    is_final: bool
    error: Optional[str]
    retry_count: int
    suggested_questions: List[str]  # New field for suggested questions
    chat_history : Optional[List[dict]]  # Optional chat history for context

logger.info("LangGraph State Defined.")



logger.info("Environment setup starting ......")
llm, retriever_tool, ingest_tool = setup_environment()
logger.info("Environment setup complete.")
# --- Initialize Tools ---
search_tool = SerperDevTool(n_results=2)
# Agent 1: Company Agent (Definition)
company_agent_def = CrewAgent(
    role='Company Information Specialist',
    goal='Provide accurate information about the company (e.g., Google) including location, products, services, history, and mission, based on search results and internal knowledge.',
    backstory='You are an AI assistant dedicated to representing the company accurately and helpfully to external queries. You use search tools to find the latest public information.',
    tools=[search_tool],
    llm=llm,
    verbose=False
)

# Agent 2: Customer Agent (Definition)
customer_agent_def = CrewAgent(
    role='Customer Support Agent',
    goal='Answer customer-specific questions by retrieving their information from the customer database using their customer ID. Handle queries about plans, purchase history, support tickets etc.',
    backstory='You are a helpful customer support agent with access to the customer database. You must use the customer ID provided to retrieve relevant information using your specialized tool.',
    tools=[search_tool, retriever_tool], # Has retriever and search
    llm=llm,
    verbose=False
)

# Agent 3: Naive Agent (Definition)
naive_agent_def = CrewAgent(
    role='General Knowledge Assistant',
    goal='Answer general knowledge questions or queries that do not fall under specific company or customer information categories. Use search tools to find relevant information online.',
    backstory='You are a general-purpose AI assistant capable of answering a wide range of topics by searching the internet.',
    tools=[search_tool],
    llm=llm,
    verbose=False
)

logger.info("CrewAI Agent Definitions Ready.")

# --- Initialize CrewAI Crew ---

def rewrite_query_node(state: AgentState):
    """Rewrites the user query and classifies which agent should handle it."""
    logger.info("--- Node: Rewriter ---")
    if state.get("customer_id"):
        logger.info(f"Customer ID found: {state['customer_id']}")
        return {
                    "rewritten_query": state['original_query'], # No rewriting needed
                    "classified_agent": "Customer", # Default to Customer if customer_id is present
                    "customer_id": state['customer_id'], # Use state customer_id directly
                    "error": None
                }

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

    # Use OpenAI Functions for structured output
    rewriter_chain = prompt | llm

    try:
        response = rewriter_chain.invoke({"query": query})
        response_dict = extract_clean_json(response.content)
        if response_dict:
                # response = json.loads(response_dict)
                print(f"Rewriter Output: {response_dict}")
                return {
                    "rewritten_query": response_dict['rewritten_query'],
                    "classified_agent": response_dict['agent_classification'],
                    "customer_id": response_dict.get('extracted_customer_id') or state.get('customer_id'), # Prioritize extracted, fallback to state
                    "error": None
                }
        else:
            print("Rewriter output was not valid JSON.")
            return {
                "rewritten_query": query,  # Fallback to original query
                "classified_agent": "Unknown",
                "customer_id": None,
                "error": "Invalid JSON response from rewriter."
            }
    except Exception as e:
        print(f"Error in Rewriter node: {e}")
        return {"error": f"Failed to process query: {e}"}


def execute_agent_node(state: AgentState, agent_def: CrewAgent, agent_name: str, ):
    """Generic function to execute the logic of a specific agent."""
    print(f"--- Node: Execute {agent_name} Agent ---")
    if state.get("error"): return {} # Don't run if prior error

    query = state['rewritten_query']
    customer_id = state.get('customer_id') # Relevant for Customer agent

    # Construct a prompt using the agent's definition
    prompt_messages = [
        SystemMessage(content=f"Role: {agent_def.role}\nGoal: {agent_def.goal}\nBackstory: {agent_def.backstory}"),
        HumanMessage(content=f"User Query: {query}")
    ]
    if agent_name == "Customer" and customer_id:
         prompt_messages.append(HumanMessage(content=f"Context: Apply this query to customer ID: {customer_id}."))
    elif agent_name == "Customer" and not customer_id:
         return {"agent_response": "Cannot answer customer question without a Customer ID.", "error": "Missing Customer ID"}


    # Simplified tool handling for LangGraph node
    # A more robust implementation would use LangChain's agent executors or tool calling
    available_tools = {tool.name: tool for tool in agent_def.tools}
    tool_response = ""

    # Basic tool check (can be expanded)
    # This is a placeholder - real tool use requires more complex agent logic (like ReAct or OpenAI Functions Agent)
    if "tavily_search" in available_tools and agent_name != "Customer": # Example: Use search for non-customer
        try:
            tool_response = f"\nSearch Results: {search_tool.run(query)}"
            prompt_messages.append(AIMessage(content=f"Tool Used: duckduckgo_search\nResult: {tool_response[:500]}...")) # Add tool result snippet
        except Exception as e:
            print(f"Error using search tool: {e}")
            tool_response = "\nSearch tool failed."

    if "ChromaDB Retriever Tool" in available_tools and agent_name == "Customer":
        try:
            logger.info(f"Using customer retriever tool for query: {query} with customer ID: {customer_id}")
            # Pass query potentially enriched with customer ID context
            if retriever_tool._get_collection(customer_id) is not None:
                retriever_query = f"Info for customer {customer_id}: {query}"
                
                tool_response = f"\nCustomer DB Info: {retriever_tool._run(query=retriever_query, collection_name=customer_id, use_mmr=False)}"
                logger.info(f"Tool Response: {tool_response}")
                prompt_messages.append(AIMessage(content=f"Tool Used: ChromaDB Retriever Tool\nResult: {tool_response}..."))
            else: 
                tool_response = "\nCustomer have no database, please add more documents or just do question-answering task"
        except Exception as e:
            print(f"Error using customer retriever tool: {e}")
            tool_response = "\nCustomer retrieval tool failed."

    # Final LLM call to generate response based on persona and potential tool output
    final_prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = final_prompt | llm
    try:
        response = chain.invoke({}) # Query is already in messages
        print(f"{agent_name} Agent Response Snippet: {response.content[:200]}...")
        return {"agent_response": response.content, "error": None}
    except Exception as e:
        print(f"Error during {agent_name} agent LLM call: {e}")
        return {"error": f"{agent_name} agent failed during generation: {e}"}

#  Specific node functions calling the generic executor
def company_agent_node(state: AgentState):
    return execute_agent_node(state, company_agent_def, "Company")

def customer_agent_node(state: AgentState):
    return execute_agent_node(state, customer_agent_def, "Customer")

def naive_agent_node(state: AgentState):
    return execute_agent_node(state, naive_agent_def, "Naive")


# --- Node 5: Reflection ---
class ReflectionOutput(BaseModel):
    """Structured output for the Reflection node."""
    feedback: str = Field(description="Constructive feedback on the response's relevance and correctness relative to the original query.")
    is_final_answer: bool = Field(description="True if the answer is satisfactory and directly addresses the original query, False otherwise.")

def reflection_node(state: AgentState):
    """Reflects on the generated answer, checking relevance and correctness."""
    print("--- Node: Reflection ---")
    if state.get("error"): return {"is_final": True} # If error occurred, end the loop

    original_query = state['original_query']
    agent_response = state['agent_response']
    # rewritten_query = state['rewritten_query'] # Could also be used for context

    if not agent_response:
         print("No agent response to reflect on.")
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
        response_dict = extract_clean_json(response.content)
        logger.info(f"Reflection Chain Response: {response.content}")
        logger.info(f"Reflection Chain JSON: {response_dict}")
        if response_dict:
            # response = json.loads(response_dict)

            logger.info(f"Reflection Chain Response: {response_dict}")
            return {
                "reflection": response_dict['feedback'],
                "is_final": response_dict['is_final_answer'],
                "retry_count": state.get('retry_count', 0) + 1, # Increment retry count
                "error": None
            }
        else:
            logger.info("Reflection output was not valid JSON.")
            return {
                "reflection": "Invalid JSON response from reflection.",
                "is_final": False,
                "retry_count": state.get('retry_count', 0) + 1,
                "error": "Invalid JSON response from reflection."
            }
    except Exception as e:
        logger.info(f"Error in Reflection node: {e}")
        # Decide how to handle reflection error - maybe treat as non-final?
        return {"error": f"Reflection failed: {e}", "is_final": False, "retry_count": state.get('retry_count', 0) + 1}

logger.info("LangGraph Nodes Defined.")


from langgraph.graph import StateGraph, END

def route_query(state: AgentState):
    """Routes to the appropriate agent node based on classification."""
    logger.info(f"--- Conditional Edge: Routing Query ---")
    if state.get("error"):
        logger.info("Error detected before routing.")
        return "error_handler" # Or END directly

    classification = state['classified_agent']
    logger.info(f"Routing based on classification: {classification}")
    if classification == "Company":
        return "execute_company"
    elif classification == "Customer":
        # Add check for customer ID existence
        if state.get("customer_id"):
             return "execute_customer"
        else:
             logger.info("Customer classification but no ID found.")
             # Update state to reflect missing ID issue and go to reflection/end
             state["error"] = "Query classified as Customer, but no Customer ID was provided or found."
             state["agent_response"] = "I need a customer ID to answer that question."
             return "reflect" # Go to reflection to potentially end gracefully
    elif classification == "Naive":
        return "execute_naive"
    else: # Unknown or error during classification
        logger.info("Classification is Unknown or invalid.")
        state["error"] = f"Could not determine the right agent for the query (classification: {classification})."
        state["agent_response"] = "I'm not sure how to handle that query. Could you please rephrase?"
        return "reflect" # Go to reflection

def decide_after_reflection(state: AgentState):
    """Decides whether to end the process or retry based on reflection."""
    logger.info(f"--- Conditional Edge: After Reflection ---")
    is_final = state['is_final']
    retry_count = state['retry_count']
    max_retries = 2 # Set a limit for retries

    if state.get("error") and "Reflection failed" not in state["error"]: # Handle agent errors first
        logger.info(f"Ending due to execution error: {state['error']}")
        return END # End directly on execution error

    logger.info(f"Reflection result: is_final={is_final}, Retry count={retry_count}")

    if is_final:
        logger.info("Reflection approved. Ending.")
        return END
    elif retry_count >= max_retries:
        logger.info(f"Max retries ({max_retries}) reached. Ending.")
        # Optionally, provide the last reflection feedback
        state["agent_response"] += f"\n\n[System Note: Max retries reached. Last feedback: {state.get('reflection', 'N/A')}]"
        return END
    else:
        print("Reflection suggests retry. Looping back to Rewriter.")
        # Add reflection feedback to the original query for context in the next loop? Optional.
        # state['original_query'] = f"{state['original_query']}\n\n[Retry Context: Previous attempt failed. Feedback: {state.get('reflection', 'N/A')}]"
        return "rewrite_query" # Loop back to the start

def question_generator_node(state: AgentState):
    """Generates relevant follow-up questions based on previous query and response context."""
    logger.info("--- Node: Question Generator ---")

    if state.get("error"):
        return {"suggested_questions": [], "error": state["error"]}

    original_query = state['original_query']
    agent_response = state['agent_response']
    agent_type = state['classified_agent']
    customer_id = state.get('customer_id')

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

    Format your response as a JSON array of suggested questions:
    {
        "suggested_questions": [
            "Question 1?",
            "Question 2?",
            "Question 3?"
        ]
    }
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"""Original Query: {original_query}

Agent Type: {agent_type}
{f'Customer ID: {customer_id}' if customer_id else ''}

Agent Response:
{agent_response}

Please generate relevant follow-up questions based on this context.""")
    ])

    question_chain = prompt | llm

    try:
        response = question_chain.invoke({})
        response_dict = extract_clean_json(response.content)

        if response_dict:
            # response_data = json.loads(response_dict)
            logger.info(f"Generated {len(response_dict.get('suggested_questions', []))} follow-up questions")
            return {
                "suggested_questions": response_dict.get("suggested_questions", []),
                "error": None
            }
        else:
            logger.warning("Question generator output was not valid JSON")

            # Fallback parsing - try to extract questions directly
            questions = []
            for line in response_dict['suggested_questions']:
                # Try to identify numbered questions or questions with question marks
                if (line.strip().startswith(('1.', '2.', '3.', '4.', '5.')) or
                    '?' in line) and len(line.strip()) > 5:
                    questions.append(line.strip())

            if questions:
                return {"suggested_questions": questions[:5], "error": None}
            else:
                return {
                    "suggested_questions": [],
                    "error": "Failed to parse question generator output"
                }
    except Exception as e:
        logger.error(f"Error in question generator node: {e}")
        return {"suggested_questions": [], "error": f"Question generation failed: {e}"}


# --- Build the Graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("execute_company", company_agent_node)
workflow.add_node("execute_customer", customer_agent_node)
workflow.add_node("execute_naive", naive_agent_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("generate_questions", question_generator_node)
# Optional: Add a specific error handling node if needed
# workflow.add_node("error_handler", ...)

# Define edges
workflow.set_entry_point("rewrite_query")

# Routing from Rewriter
workflow.add_conditional_edges(
    "rewrite_query",
    route_query,
    {
        "execute_company": "execute_company",
        "execute_customer": "execute_customer",
        "execute_naive": "execute_naive",
        "reflect": "reflect", # Handle Unknown/Error cases by going directly to reflection
        # "error_handler": "error_handler" # Route explicit errors
    }
)

# Edges from execution nodes to reflection
workflow.add_edge("execute_company", "reflect")
workflow.add_edge("execute_customer", "reflect")
workflow.add_edge("execute_naive", "reflect")

# Conditional edge from Reflection (Loop or End)
workflow.add_conditional_edges(
    "reflect",
    lambda state: "generate_questions" if state["is_final"] else decide_after_reflection(state),
    {
        "generate_questions": "generate_questions",
        "rewrite_query": "rewrite_query",
        END: END
    }
)
workflow.add_edge("generate_questions", END)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
# Compile the graph
app = workflow.compile(checkpointer=checkpointer) # Set recursion limit for safety


logger.info("LangGraph Compiled.")


def run_agentic_system(query: str, cust_id: Optional[str] = None):
    """Runs the agentic system with a user query."""
    initial_state = AgentState(
        original_query=query,
        customer_id=cust_id,
        rewritten_query="",
        classified_agent="Unknown", # Start as Unknown
        agent_response="",
        reflection="",
        is_final=False,
        error=None,
        retry_count=0
    )

    logger.info(f"\n🚀 Starting Agentic System for Query: '{query}'" + (f" (Customer ID: {cust_id})" if cust_id else ""))
    # config = {"configurable": {"thread_id": f"thread_{query[:10]}"}} # Example thread ID if using checkpointer

    try:
        # final_state = app.invoke(initial_state, config=config)
        config = {"configurable": {"thread_id": "1"}}
        final_state = app.invoke(initial_state, config) # Example thread ID if using checkpointer


        logger.info("\n🏁 Agentic System Finished!")
        logger.info("------ Final State ------")
        # Pretty print relevant parts of the final state
        logger.info(f"Original Query: {final_state['original_query']}")
        if final_state.get('rewritten_query'): print(f"Rewritten Query: {final_state['rewritten_query']}")
        if final_state.get('classified_agent'): print(f"Agent Used: {final_state['classified_agent']}")
        if final_state.get('error'):
            logger.info(f"Error Occurred: {final_state['error']}")
        logger.info(f"\nFinal Response:\n{final_state['agent_response']}")
        if final_state.get('reflection') and not final_state.get('is_final') and final_state.get('retry_count') > 0:
             logger.info(f"\nLast Reflection Feedback (Process Ended): {final_state['reflection']}")
        # Add display of suggested questions if present
        if final_state.get('suggested_questions'):
            logger.info("\n✨ Suggested follow-up questions:")
            for i, question in enumerate(final_state['suggested_questions'], 1):
                logger.info(f"  {i}. {question}")
        return final_state

    except Exception as e:
        logger.info(f"\n💥 An unhandled error occurred during graph execution: {e}")
        import traceback
        traceback.print_exc()
        return None

agent_response_object = run_agentic_system("What is CoPali?", cust_id="dat1")
agent_response = agent_response_object['agent_response']
logger.info(f"Agent Response: {agent_response}")
logger.info(f"Suggested Questions: {agent_response_object['suggested_questions']}")

