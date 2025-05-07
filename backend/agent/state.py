from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal, Optional, TypedDict, Dict, List, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from loguru import logger
from agent.utils.helper_function import extract_clean_json

import os 
import json 

# import cr

# logger.add("agent_state.log", rotation="1 MB", retention="10 days", level="DEBUG")
# # --- State Definition ---


# class AgentState(TypedDict):
#     original_query: str
#     customer_id: Optional[str] # Allow optional customer ID
#     rewritten_query: str
#     classified_agent: Literal["Company", "Customer", "Naive", "Unknown"] # Agent types
#     agent_response: str
#     reflection: str
#     is_final: bool
#     error: Optional[str]
#     retry_count: int

# logger.debug("LangGraph State Defined.")
# # --- Node 1: Rewriter ---
# class RewriterOutput(BaseModel):
#     """Structured output for the Rewriter node."""
#     rewritten_query: str = Field(description="The user's query, corrected for spelling and clarity.")
#     agent_classification: Literal["Company", "Customer", "Naive", "Unknown"] = Field(description="The best agent type to handle the rewritten query.")
#     extracted_customer_id: Optional[str] = Field(description="Customer ID extracted from the query, if relevant and present (e.g., 'cust123'). Null otherwise.")

# def rewrite_query_node(state: AgentState):
#     """Rewrites the user query and classifies which agent should handle it."""
#     logger.debug("--- Node: Rewriter ---")
#     query = state['original_query']

#     system_prompt = """You are an expert query processor. Your tasks are:
#     1.  Correct any spelling mistakes in the user query.
#     2.  Rephrase the query for maximum clarity if necessary.
#     3.  Classify the query's intent to determine the best agent to handle it:
#         - 'Company': For questions about a specific company's details (location, products, services, mission etc.). Assume the company is 'Google' if not specified.
#         - 'Customer': For questions related to a specific customer's account, history, plan, support tickets etc. These queries MUST contain or imply a customer ID.
#         - 'Naive': For general knowledge questions, queries unrelated to the company or a specific customer.
#         - 'Unknown': If the query is ambiguous or cannot be clearly classified.
#     4. Extract any customer ID mentioned (like cust123, user45, id: 7890). Return null if no ID is found.
#     Provide your output in the specified JSON format.
#     For example, for the query 'What is the status of my order with customer ID cust123?', your output should be:
#     <output>
#     {{
#         "rewritten_query": "What is the status of my order?",
#         "agent_classification": "Customer",
#         "extracted_customer_id": "cust123"
#     }}
#     </output>
#     If no customer ID is found, return null for that field.
#     If the query is not clear or cannot be classified, return 'Unknown' for the agent classification and null for the customer ID.
#     Your output should be a JSON object with the following fields:

#     {{
#         "rewritten_query": "<corrected and clarified query>",
#         "agent_classification": "<Company|Customer|Naive|Unknown>",
#         "extracted_customer_id": "<customer ID or null>"
        
#     }}

#     **CRITICAL:** You MUST format your output ONLY as a JSON object that strictly adheres to the following schema. Do NOT include any other text, explanations, or markdown formatting like ```json ``` around the JSON object itself."""


#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "Process the following user query: {query}")
#     ])

#     # Use OpenAI Functions for structured output
#     rewriter_chain = prompt | llm 
    
#     try:
#         response = rewriter_chain.invoke({"query": query})
#         json_str = extract_clean_json(response.content)
#         if json_str:
#                 response = json.loads(json_str)
#                 print(f"Rewriter Output: {response}")
#                 return {
#                     "rewritten_query": response['rewritten_query'],
#                     "classified_agent": response['agent_classification'],
#                     "customer_id": response.get('extracted_customer_id') or state.get('customer_id'), # Prioritize extracted, fallback to state
#                     "error": None
#                 }
#         else:
#             print("Rewriter output was not valid JSON.")
#             return {
#                 "rewritten_query": query,  # Fallback to original query
#                 "classified_agent": "Unknown",
#                 "customer_id": None,
#                 "error": "Invalid JSON response from rewriter."
#             }
#     except Exception as e:
#         print(f"Error in Rewriter node: {e}")
#         return {"error": f"Failed to rewrite/classify query: {e}"}

# # --- Nodes 2, 3, 4: Specialized Agent Execution ---
# # These nodes will simulate the execution of the conceptually defined CrewAI agents.

# def execute_agent_node(state: AgentState, agent_def: CrewAgent, agent_name: str):
#     """Generic function to execute the logic of a specific agent."""
#     print(f"--- Node: Execute {agent_name} Agent ---")
#     if state.get("error"): return {} # Don't run if prior error

#     query = state['rewritten_query']
#     customer_id = state.get('customer_id') # Relevant for Customer agent

#     # Construct a prompt using the agent's definition
#     prompt_messages = [
#         SystemMessage(content=f"Role: {agent_def.role}\nGoal: {agent_def.goal}\nBackstory: {agent_def.backstory}"),
#         HumanMessage(content=f"User Query: {query}")
#     ]
#     if agent_name == "Customer" and customer_id:
#          prompt_messages.append(HumanMessage(content=f"Context: Apply this query to customer ID: {customer_id}."))
#     elif agent_name == "Customer" and not customer_id:
#          return {"agent_response": "Cannot answer customer question without a Customer ID.", "error": "Missing Customer ID"}


#     # Simplified tool handling for LangGraph node
#     # A more robust implementation would use LangChain's agent executors or tool calling
#     available_tools = {tool.name: tool for tool in agent_def.tools}
#     tool_response = ""

#     # Basic tool check (can be expanded)
#     # This is a placeholder - real tool use requires more complex agent logic (like ReAct or OpenAI Functions Agent)
#     if "duckduckgo_search" in available_tools and agent_name != "Customer": # Example: Use search for non-customer
#         try:
#             tool_response = f"\nSearch Results: {search_tool.run(query)}"
#             prompt_messages.append(AIMessage(content=f"Tool Used: duckduckgo_search\nResult: {tool_response[:500]}...")) # Add tool result snippet
#         except Exception as e:
#             print(f"Error using search tool: {e}")
#             tool_response = "\nSearch tool failed."

#     if "customer_info_retriever" in available_tools and agent_name == "Customer":
#         try:
#             # Pass query potentially enriched with customer ID context
#             retriever_query = f"Info for customer {customer_id}: {query}"
#             tool_response = f"\nCustomer DB Info: {customer_retriever_tool.run(retriever_query)}"
#             prompt_messages.append(AIMessage(content=f"Tool Used: customer_info_retriever\nResult: {tool_response[:500]}..."))
#         except Exception as e:
#             print(f"Error using customer retriever tool: {e}")
#             tool_response = "\nCustomer retrieval tool failed."

#     # Final LLM call to generate response based on persona and potential tool output
#     final_prompt = ChatPromptTemplate.from_messages(prompt_messages)
#     chain = final_prompt | llm
#     try:
#         response = chain.invoke({}) # Query is already in messages
#         print(f"{agent_name} Agent Response Snippet: {response.content[:200]}...")
#         return {"agent_response": response.content, "error": None}
#     except Exception as e:
#         print(f"Error during {agent_name} agent LLM call: {e}")
#         return {"error": f"{agent_name} agent failed during generation: {e}"}


# # Specific node functions calling the generic executor
# def company_agent_node(state: AgentState):
#     return execute_agent_node(state, company_agent_def, "Company")

# def customer_agent_node(state: AgentState):
#     return execute_agent_node(state, customer_agent_def, "Customer")

# def naive_agent_node(state: AgentState):
#     return execute_agent_node(state, naive_agent_def, "Naive")

# # --- Node 5: Reflection ---
# class ReflectionOutput(BaseModel):
#     """Structured output for the Reflection node."""
#     feedback: str = Field(description="Constructive feedback on the response's relevance and correctness relative to the original query.")
#     is_final_answer: bool = Field(description="True if the answer is satisfactory and directly addresses the original query, False otherwise.")

# def reflection_node(state: AgentState):
#     """Reflects on the generated answer, checking relevance and correctness."""
#     print("--- Node: Reflection ---")
#     if state.get("error"): return {"is_final": True} # If error occurred, end the loop

#     original_query = state['original_query']
#     agent_response = state['agent_response']
#     # rewritten_query = state['rewritten_query'] # Could also be used for context

#     if not agent_response:
#          print("No agent response to reflect on.")
#          return {"reflection": "No response generated.", "is_final": True, "error": "Reflection failed: No agent response found."}


#     system_prompt = """You are a meticulous quality assurance reviewer. Your task is to evaluate an AI agent's response based on the user's original query.
#                     Assess the following:
#                     1.  **Relevance:** Does the response directly address the user's original question?
#                     2.  **Correctness:** Is the information likely correct (based on general knowledge or provided context)? You don't need to verify external facts exhaustively, but check for obvious flaws or contradictions.
#                     3.  **Completeness:** Does the response sufficiently answer the question?

#                     Provide  constructive feedback and determine if the answer is final (good enough) or needs revision/retry. Use the specified JSON format.

#                     For example, if the response is relevant and correct, your output should be:
#                     <output>
#                     {{
#                         "feedback": "The response is relevant and correct. It directly addresses the user's query about the company's mission.",
#                         "is_final_answer": true
#                     }}
#                     </output>
#                     If the response is not relevant or correct, your output should be: 
#                     <output>
#                     {{
#                         "feedback": "The response is not relevant. It does not address the user's query about the company's mission.",
#                         "is_final_answer": false
#                     }}
#                     </output>
#                     If the response is partially correct but needs more detail, your output should be:
#                     <output>
#                     {{
#                         "feedback": "The response is partially correct but lacks detail. It mentions the company's mission but does not explain it.",
#                         "is_final_answer": false
#                     }}
#                     </output>
#                     If the response is completely incorrect, your output should be:
#                     <output>
#                     {{
#                         "feedback": "The response is completely incorrect. It does not address the user's query at all.",
#                         "is_final_answer": false
#                     }}
#                     </output>
#                     If the response is satisfactory and directly addresses the original query, your output should be:
#                     <output>
#                     {{
#                         "feedback": "The response is satisfactory and directly addresses the user's query.",
#                         "is_final_answer": true
#                     }}
#                     </output>
#                     Your output should be a JSON object with the following fields:
#                     {{
#                         "feedback": "<constructive feedback on relevance, correctness, and completeness>",
#                         "is_final_answer": <true|false>
#                     }}
#                     **CRITICAL:** You MUST format your output ONLY as a JSON object that strictly adheres to the above schema. Do NOT include any other text, explanations, or markdown formatting like ```json ``` around the JSON object itself.
#                     """

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "Original Query: {original_query}\n\nAgent Response:\n{agent_response}\n\nPlease evaluate and provide feedback. Your output should be a JSON object as specified above.")
#     ])

#     reflection_chain = prompt | llm
#     try:
#         response = reflection_chain.invoke({
#             "original_query": original_query,
#             "agent_response": agent_response
#         })
#         json_str = extract_clean_json(response.content)
#         logger.info(f"Reflection Chain Response: {response.content}")
#         logger.info(f"Reflection Chain JSON: {json_str}")
#         if json_str:
#             response = json.loads(json_str)
        
#             logger.info(f"Reflection Chain Response: {response}")
#             return {
#                 "reflection": response['feedback'],
#                 "is_final": response['is_final_answer'],
#                 "retry_count": state.get('retry_count', 0) + 1, # Increment retry count
#                 "error": None
#             }
#         else:
#             print("Reflection output was not valid JSON.")
#             return {
#                 "reflection": "Invalid JSON response from reflection.",
#                 "is_final": False,
#                 "retry_count": state.get('retry_count', 0) + 1,
#                 "error": "Invalid JSON response from reflection."
#             }
#     except Exception as e:
#         print(f"Error in Reflection node: {e}")
#         # Decide how to handle reflection error - maybe treat as non-final?
#         return {"error": f"Reflection failed: {e}", "is_final": False, "retry_count": state.get('retry_count', 0) + 1}

# print("LangGraph Nodes Defined.")