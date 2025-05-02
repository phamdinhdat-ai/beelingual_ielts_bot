from crewai import Agent
from crewai_tools import BaseTool
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
import re
from langchain_community.chat_models.ollama import ChatOllama
from agent.config.load_config import agents_config


class BaseAgent():
    
    def __init__(self, model_name: str, tools: list[BaseTool], config: Optional[Dict[str, Any]] = None):
        
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
        # Create the agent here
        agent = Agent(
            llm=self.llm,
            tools=self.tools,
            config=self.config
        )
        return agent