from crewai import Agent, Task, Crew
from langchain_community.chat_models.ollama import ChatOllama
from agent.tools.nl2sql_tool import NL2SQLTool
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Create the NL2SQL tool
nl2sql_tool = NL2SQLTool()
# test_nl2sql.py
from agent.tools.nl2sql_tool import NL2SQLTool, ValidateSQLQueryTool
from crewai import Agent, Task, Crew
from langchain_community.chat_models.ollama import ChatOllama
from pydantic import BaseModel, Field

from typing import Dict, Optional, Any

class OutputAgent(BaseModel):
    """Output for the NL2SQL tool."""
    sql_query: str = Field(..., description="The generated SQL query")
    validation_result: str = Field(..., description="Validation result of the SQL query")



# Initialize the LLM
llm = ChatOllama(model='deepseek-r1:1.5b', temperature=0.2, max_tokens=2000)

# Create test cases
test_cases = [
    {
        "name": "Simple Query",
        "question": "Find all customers who purchased more than 5 items last month",
        "database_schema": None,
        "dialect": "sqlite"
    },
    {
        "name": "Query with Schema",
        "question": "Find the top 5 products by revenue",
        "database_schema": """
        Customers(customer_id, name, email, signup_date)
        Orders(order_id, customer_id, order_date, total_amount)
        OrderItems(order_item_id, order_id, product_id, quantity, price)
        Products(product_id, name, category, price)
        """,
        "dialect": "sqlite"
    },
    {
        "name": "PostgreSQL Query",
        "question": "Find average monthly sales by category for the last year",
        "database_schema": """
        Products(product_id, name, category, price)
        Orders(order_id, customer_id, order_date, total_amount)
        OrderItems(order_item_id, order_id, product_id, quantity, price)
        """,
        "dialect": "postgresql"
    }
]



# Create the NL2SQL tool
nl2sql_tool = NL2SQLTool()
validate_sql_tool = ValidateSQLQueryTool()

data_analyst = Agent(
        role="SQL Data Analyst",
        goal="Convert natural language questions to SQL queries",
        backstory="I help users generate SQL queries from their natural language questions.",
        verbose=True,
        tools=[nl2sql_tool],
        llm=llm,
    )
    
sql_validator = Agent(
    role="SQL Validator",
    goal="Validate SQL queries for correctness",
    backstory="I ensure SQL queries are syntactically correct and optimized.",
    verbose=True,
    tools=[validate_sql_tool],
    llm=llm,
)


# Run tests for each case
# for test_case in test_cases:
test_case = test_cases[0]
print(f"\n--- Testing: {test_case['name']} ---")

# Create an agent that uses the tool
data_analyst = Agent(
    role="SQL Data Analyst",
    goal="Convert natural language questions to SQL queries",
    backstory="I help users generate SQL queries from their natural language questions.",
    verbose=True,
    tools=[nl2sql_tool],
    llm=llm,
    
)

# Tasks
sql_generation_task = Task(
    description=f"Convert the following question to SQL: '{test_case['question']}'",
    agent=data_analyst,
    prompt_context=f"""
    Question: {test_case['question']}
    Database Schema: {test_case['database_schema'] or 'Not provided'}
    SQL Dialect: {test_case['dialect']}
    """,
    expected_output="SQL Query",
)

validation_task = Task(

    description="Validate the SQL query for syntax and structure",
    agent=sql_validator,
    prompt_context="Validate the following SQL query",
    expected_output="Validation result",
    context=[sql_generation_task],
    output_pydantic=OutputAgent,

)

# # Create and run the crew
# crew = Crew(
#     agents=[data_analyst],
#     tasks=[sql_generation_task],
#     verbose=True
# )

# Create and run the crew
crew = Crew(
    agents=[data_analyst, sql_validator],
    tasks=[sql_generation_task, validation_task],
    verbose=True,
    planing=True,

)

result = crew.kickoff()
print(result)