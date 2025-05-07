import unittest
from unittest.mock import patch, MagicMock
import pytest
import json
import os
from typing import Dict, Any, Optional
from backend.agent.multi_agents import run_agentic_system, AgentState

# filepath: backend/agent/test_multi_agents.py


# Import the function to test

class TestMultiAgents(unittest.TestCase):
    """Test class for multi_agents.py's run_agentic_system function."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment once before all tests."""
        # Ensure any environment variables or global setup is done
        os.environ["SERPER_API_KEY"] = "test_key"  # Mock API key for testing
        
    def setUp(self):
        """Setup before each test."""
        # Create patcher for app.invoke to avoid actual LLM calls
        self.app_patcher = patch('backend.agent.multi_agents.app.invoke')
        self.mock_app_invoke = self.app_patcher.start()
        
    def tearDown(self):
        """Cleanup after each test."""
        self.app_patcher.stop()

    def _mock_final_state(self, 
                          original_query: str = "test query",
                          agent_response: str = "This is a test response",
                          classified_agent: str = "Naive",
                          is_final: bool = True,
                          suggested_questions: Optional[list] = None,
                          error: Optional[str] = None) -> Dict[str, Any]:
        """Helper to create a mock final state for app.invoke."""
        if suggested_questions is None:
            suggested_questions = ["Follow-up question 1?", "Follow-up question 2?"]
            
        return {
            "original_query": original_query,
            "rewritten_query": "rewritten " + original_query,
            "classified_agent": classified_agent,
            "agent_response": agent_response,
            "reflection": "Good response",
            "is_final": is_final,
            "error": error,
            "retry_count": 1,
            "suggested_questions": suggested_questions,
            "customer_id": None
        }
    
    def test_basic_successful_query(self):
        """Test a basic successful query through the agentic system."""
        # Setup mock response
        mock_final_state = self._mock_final_state(
            original_query="What is AI?",
            agent_response="AI stands for Artificial Intelligence...",
            classified_agent="Naive"
        )
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("What is AI?")
        
        # Assertions
        self.assertEqual(result["agent_response"], mock_final_state["agent_response"])
        self.assertEqual(result["classified_agent"], "Naive")
        self.assertTrue(len(result["suggested_questions"]) > 0)
        self.mock_app_invoke.assert_called_once()
    
    def test_with_customer_id(self):
        """Test query with customer ID routing to Customer agent."""
        # Setup mock response
        mock_final_state = self._mock_final_state(
            original_query="What's in my account?",
            agent_response="Your account has the following details...",
            classified_agent="Customer",
        )
        mock_final_state["customer_id"] = "cust123"
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("What's in my account?", cust_id="cust123")
        
        # Assertions
        self.assertEqual(result["classified_agent"], "Customer")
        self.assertEqual(result["customer_id"], "cust123")
        
        # Check that the initial state was created with the correct customer ID
        args, kwargs = self.mock_app_invoke.call_args
        initial_state = args[0]
        self.assertEqual(initial_state["customer_id"], "cust123")
    
    def test_error_handling(self):
        """Test handling of errors during processing."""
        # Setup mock to raise exception
        self.mock_app_invoke.side_effect = Exception("Test error")
        
        # Execute function - should return None on error
        result = run_agentic_system("Problematic query")
        
        # Assertions
        self.assertIsNone(result)
    
    def test_question_generation(self):
        """Test that follow-up questions are generated."""
        # Setup mock with specific questions
        custom_questions = [
            "What are the latest AI advancements?",
            "How does AI impact society?",
            "What are the ethical concerns with AI?"
        ]
        mock_final_state = self._mock_final_state(
            original_query="Tell me about AI",
            suggested_questions=custom_questions
        )
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("Tell me about AI")
        
        # Assertions
        self.assertEqual(result["suggested_questions"], custom_questions)
        self.assertEqual(len(result["suggested_questions"]), 3)
    
    def test_without_suggested_questions(self):
        """Test behavior when no suggested questions are returned."""
        # Setup mock with no questions
        mock_final_state = self._mock_final_state(suggested_questions=[])
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("Simple query")
        
        # Assertions
        self.assertEqual(result["suggested_questions"], [])
    
    def test_company_agent_classification(self):
        """Test query routed to Company agent."""
        # Setup mock for company classification
        mock_final_state = self._mock_final_state(
            original_query="What products does Google offer?",
            agent_response="Google offers various products including Search, Gmail...",
            classified_agent="Company"
        )
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("What products does Google offer?")
        
        # Assertions
        self.assertEqual(result["classified_agent"], "Company")
    
    def test_multiple_retries(self):
        """Test scenario where multiple retries occurred."""
        # Setup mock with retry count > 1
        mock_final_state = self._mock_final_state()
        mock_final_state["retry_count"] = 2
        mock_final_state["reflection"] = "Improved after retry"
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("Complex query")
        
        # Assertions
        self.assertEqual(result["retry_count"], 2)

    @patch('backend.agent.multi_agents.logger.info')
    def test_logging(self, mock_logger):
        """Test that important information is logged."""
        # Setup mock response
        self.mock_app_invoke.return_value = self._mock_final_state()
        
        # Execute function
        run_agentic_system("Test query")
        
        # Check logging calls
        mock_logger.assert_any_call("\n🚀 Starting Agentic System for Query: 'Test query'")
        mock_logger.assert_any_call("\n🏁 Agentic System Finished!")
    
    def test_with_complex_error(self):
        """Test handling when an error occurs during processing but app.invoke completes."""
        # Setup mock with error in final state
        mock_final_state = self._mock_final_state(
            error="Classification failed",
            agent_response="I couldn't process your query properly."
        )
        self.mock_app_invoke.return_value = mock_final_state
        
        # Execute function
        result = run_agentic_system("Problematic query")
        
        # Assertions
        self.assertEqual(result["error"], "Classification failed")
        self.assertFalse(result["is_final"])  # Assuming errors are treated as non-final

if __name__ == '__main__':
    unittest.main()