from crewai import Agent
from crewai_tools import BaseTool
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
import re
from langchain_community.chat_models.ollama import ChatOllama
from agent.config.load_config import agents_config
# from agent.agent_crew.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from agent.tools.nl2sql_tool import NL2SQLTool,  ValidateSQLQueryTool
from os.path import dirname, join, abspath
from dotenv import load_dotenv
import os 


env_path = join(dirname(dirname(abspath(__file__))), '.env')
load_dotenv(env_path)

nl2sql_tool = NL2SQLTool()
validate_sql_tool = ValidateSQLQueryTool()
search_tool = SerperDevTool(api_key=os.environ['SERPER_API_KEY'])
tools = [tool for tool in [search_tool, nl2sql_tool, validate_sql_tool] if tool is not None]

class CustomerAgent:
    """
    A class to represent a company agent.
    """

    def __init__(self, model_name: str, tools: list[BaseTool], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CompanyAgent with a model name and tools.

        Args:
            model_name (str): The name of the model to be used.
            tools (list[BaseTool]): A list of tools to be used by the agent.
        """
        self.tools = tools
        self.llm = self.load_llm(model_name)
        self.config = config 

    def load_llm(self, model_name: str):
        """
        Load the LLM model.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            LLM: Loaded LLM model.
        """
        # Load the LLM model here
        llm = ChatOllama(model=model_name, temperature=0.2, max_tokens=2000)
        return llm
    
    def create_agent(self):
        """
        Create an agent with the loaded LLM and tools.

        Returns:
            Agent: The created agent.
        """
        agent = Agent.from_tools(
            self.llm,
            self.tools,
            verbose=True,
            config=self.config
        )
        return agent