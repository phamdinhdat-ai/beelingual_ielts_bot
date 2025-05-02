import crewai_tools
from crewai_tools import BaseTool
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
import re
import sqlparse
class NL2SQLInput(BaseModel):
    """Input for the NL2SQL tool."""
    question: str = Field(..., description="The natural language question to convert to SQL")
    database_schema: Optional[str] = Field(None, description="Description of the database schema to help with SQL generation")
    dialect: Optional[str] = Field("sqlite", description="SQL dialect to use (e.g., 'sqlite', 'postgresql', 'mysql')")

class NL2SQLTool(BaseTool):
    """Tool for converting natural language questions to SQL queries."""
    name: str = "NL2SQL Tool"
    description: str = "Converts natural language questions to SQL queries based on the provided database schema"
    input_schema: type[BaseModel] = NL2SQLInput
    
    def _run(self, sql_query: str, dialect: Optional[str] = "sqlite") -> str:
        """
        Convert natural language question to SQL query.
        
        Args:
            question: Natural language question to convert
            database_schema: Optional description of database tables and columns
            dialect: SQL dialect to use
            
        Returns:
            str: Generated SQL query
        """
         # Remove any commented lines
        sql_query = self._remove_comments(sql_query)
        # Use the CrewAI's agent LLM to generate SQL
        # We'll use the crew's agent's built-in LLM capabilities
        
         # Skip empty queries
        if not sql_query.strip():
            return "Invalid query: Empty query"
        
        # Basic syntax check using sqlparse
        try:
            formatted_query = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            parsed = sqlparse.parse(sql_query)
            
            if not parsed:
                return "Invalid query: Could not parse SQL"
            
            # Basic structure validation
            result = self._validate_structure(parsed[0], dialect)
            if result != "Valid":
                return result
                
            return f"Valid SQL query:\n\n```sql\n{formatted_query}\n```"
            
        except Exception as e:
            return f"Invalid query: {str(e)}"
        
    
    def _create_prompt(self, question: str, database_schema: Optional[str], dialect: str) -> str:
        """Create a prompt for the LLM to generate SQL from."""
        prompt = f"Convert the following question to a {dialect} SQL query:\n\nQuestion: {question}\n\n"
        
        if database_schema:
            prompt += f"Database Schema:\n{database_schema}\n\n"
        
        prompt += "SQL Query:"
        return prompt
    
    def _generate_sql(self, prompt: str) -> str:
        """Generate SQL using the built-in LLM capabilities of CrewAI."""
        # In a real implementation, this would use the agent's LLM
        # For now, we'll use a placeholder that would be replaced with actual LLM call
        
        # Example: return self.agent.llm.generate(prompt)
        # Since we don't have direct access to the agent's LLM in this context,
        # this would need to be implemented based on how CrewAI tools access the LLM
        
        # Placeholder - in actual use, would call LLM
        return "-- SQL query would be generated here based on the natural language question"
    


class ValidateSQLQueryInput(BaseModel):
    """Input for the SQL query validation tool."""
    sql_query: str = Field(..., description="The SQL query to validate")
    database_schema: Optional[str] = Field(None, description="Description of the database schema to validate against")
    dialect: Optional[str] = Field("sqlite", description="SQL dialect to use (e.g., 'sqlite', 'postgresql', 'mysql')")



class ValidateSQLQueryTool(BaseTool):
    """Tool for validating SQL queries."""
    name: str = "Validate SQL Query Tool"
    description: str = "Validates SQL queries against the provided database schema"
    input_schema: type[BaseModel] = ValidateSQLQueryInput
    
    def _run(self, sql_query: str, database_schema: Optional[str] = None, dialect: Optional[str] = "sqlite") -> bool:
        """
        Validate the SQL query against the database schema.
        
        Args:
            sql_query: SQL query to validate
            database_schema: Optional description of database tables and columns
            dialect: SQL dialect to use
            
        Returns:
            bool: True if the query is valid, False otherwise
        """
        sql_query = self._remove_comments(sql_query)
        try: 
            formatted_query = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return "Invalid query: Could not parse SQL"
            
            result = self._validate_structure(parsed[0], dialect)
            if result != "Valid":
                return result
                
            return f"Valid SQL query:\n\n```sql\n{formatted_query}\n```"
            
        except Exception as e:
            return f"Invalid query: {str(e)}"
            
     

    def _remove_comments(self, sql_query: str) -> str:
        """Remove comments from the SQL query."""
        # Placeholder for actual comment removal logic
        # In a real implementation, this would parse the SQL and remove comments
        sql_query = re.sub(r'--.*?$', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
        return sql_query
    
     
    def _validate_structure(self, parsed_query, dialect: str) -> str:
        """Validate the structure of the SQL query."""
        query_type = parsed_query.get_type()
        
        # Check for basic query types
        if query_type not in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']:
            return f"Invalid query: Unrecognized query type '{query_type}'"
        
        # Additional dialect-specific checks can be added here
        if dialect == 'postgresql':
            # PostgreSQL specific validations
            if query_type == 'SELECT' and 'FOR UPDATE' in str(parsed_query) and 'NOWAIT' not in str(parsed_query):
                return "Warning: PostgreSQL 'FOR UPDATE' without 'NOWAIT' might cause blocking"
        
        return "Valid"