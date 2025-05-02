import re
import sqlparse
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict
import difflib

class NL2SQLEvaluator:
    """Evaluator class for NL2SQL systems with multiple metrics."""
    
    def __init__(self, db_executor=None, embedding_model=None):
        """
        Initialize the evaluator.
        
        Args:
            db_executor: Optional database execution engine to run queries
            embedding_model: Optional model to create embeddings for semantic similarity
        """
        self.db_executor = db_executor
        self.embedding_model = embedding_model
    
    def normalize_query(self, query: str) -> str:
        """Normalize SQL query for better comparison."""
        # Remove comments
        query = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        
        # Format with sqlparse
        query = sqlparse.format(
            query.strip(),
            keyword_case='upper',
            identifier_case='lower',
            strip_comments=True,
            reindent=True,
            comma_first=False
        )
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'\( ', '(', query)
        query = re.sub(r' \)', ')', query)
        query = re.sub(r' ,', ',', query)
        
        return query.strip()
    
    def exact_match_accuracy(self, 
                            predicted_queries: List[str], 
                            reference_queries: List[str]) -> Dict:
        """
        Calculate exact match accuracy after normalizing the queries.
        
        Args:
            predicted_queries: List of generated SQL queries
            reference_queries: List of reference (ground truth) SQL queries
            
        Returns:
            Dict with accuracy score and matches
        """
        if len(predicted_queries) != len(reference_queries):
            raise ValueError("Predicted and reference query lists must have the same length")
        
        matches = []
        for pred, ref in zip(predicted_queries, reference_queries):
            norm_pred = self.normalize_query(pred)
            norm_ref = self.normalize_query(ref)
            match = norm_pred == norm_ref
            matches.append(match)
        
        accuracy = sum(matches) / len(matches) if matches else 0
        return {
            "metric": "exact_match_accuracy",
            "score": accuracy,
            "matches": matches,
            "per_query": [{"match": m, "normalized_pred": self.normalize_query(p), 
                           "normalized_ref": self.normalize_query(r)}
                          for m, p, r in zip(matches, predicted_queries, reference_queries)]
        }
    
    def execution_accuracy(self, 
                          predicted_queries: List[str], 
                          reference_queries: List[str]) -> Dict:
        """
        Calculate execution accuracy by comparing query results.
        Requires a database executor to be provided.
        
        Args:
            predicted_queries: List of generated SQL queries
            reference_queries: List of reference SQL queries
            
        Returns:
            Dict with accuracy score and matches
        """
        if not self.db_executor:
            raise ValueError("Database executor is required for execution accuracy")
        
        matches = []
        results = []
        
        for pred, ref in zip(predicted_queries, reference_queries):
            try:
                pred_result = self.db_executor.execute_query(pred)
                ref_result = self.db_executor.execute_query(ref)
                
                # Compare results (should handle different column orders)
                pred_df = pd.DataFrame(pred_result)
                ref_df = pd.DataFrame(ref_result)
                
                # Sort both dataframes to handle different row orders
                if not pred_df.empty and not ref_df.empty:
                    pred_df = pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True)
                    ref_df = ref_df.sort_values(by=list(ref_df.columns)).reset_index(drop=True)
                
                match = pred_df.equals(ref_df)
                results.append((pred_result, ref_result))
            except Exception as e:
                match = False
                results.append((str(e), None))
                
            matches.append(match)
        
        accuracy = sum(matches) / len(matches) if matches else 0
        return {
            "metric": "execution_accuracy",
            "score": accuracy,
            "matches": matches,
            "details": results
        }
    
    def parse_query_components(self, query: str) -> Dict[str, str]:
        """Parse SQL query into its components."""
        try:
            parsed = sqlparse.parse(query)[0]
            components = {}
            
            # Extract SELECT clause
            select_tokens = []
            from_tokens = []
            where_tokens = []
            group_tokens = []
            having_tokens = []
            order_tokens = []
            
            current_component = None
            for token in parsed.tokens:
                token_upper = token.value.upper() if hasattr(token, 'value') else ''
                
                if token_upper == 'SELECT':
                    current_component = select_tokens
                elif token_upper == 'FROM':
                    current_component = from_tokens
                elif token_upper == 'WHERE':
                    current_component = where_tokens
                elif token_upper == 'GROUP BY':
                    current_component = group_tokens
                elif token_upper == 'HAVING':
                    current_component = having_tokens
                elif token_upper == 'ORDER BY':
                    current_component = order_tokens
                elif current_component is not None:
                    current_component.append(str(token))
            
            if select_tokens:
                components['select'] = ''.join(select_tokens[1:])
            if from_tokens:
                components['from'] = ''.join(from_tokens[1:])
            if where_tokens:
                components['where'] = ''.join(where_tokens[1:])
            if group_tokens:
                components['group'] = ''.join(group_tokens[1:])
            if having_tokens:
                components['having'] = ''.join(having_tokens[1:])
            if order_tokens:
                components['order'] = ''.join(order_tokens[1:])
                
            return components
        except Exception:
            return {}
    
    def component_matching(self, 
                          predicted_queries: List[str], 
                          reference_queries: List[str]) -> Dict:
        """
        Calculate component-level accuracy.
        
        Args:
            predicted_queries: List of generated SQL queries
            reference_queries: List of reference SQL queries
            
        Returns:
            Dict with component-level matching scores
        """
        component_matches = defaultdict(list)
        overall_matches = []
        
        for pred, ref in zip(predicted_queries, reference_queries):
            try:
                pred_components = self.parse_query_components(pred)
                ref_components = self.parse_query_components(ref)
                
                # Gather all component types
                all_components = set(pred_components.keys()) | set(ref_components.keys())
                
                matches_for_query = {}
                for component in all_components:
                    pred_comp = self.normalize_query(pred_components.get(component, ""))
                    ref_comp = self.normalize_query(ref_components.get(component, ""))
                    
                    match = pred_comp == ref_comp
                    component_matches[component].append(match)
                    matches_for_query[component] = match
                
                # A query matches overall if all its components match
                overall_match = all(matches_for_query.values()) if matches_for_query else False
                overall_matches.append(overall_match)
                
            except Exception:
                for component in ['select', 'from', 'where', 'group', 'having', 'order']:
                    component_matches[component].append(False)
                overall_matches.append(False)
        
        # Calculate accuracy for each component
        component_scores = {
            component: sum(matches) / len(matches) if matches else 0
            for component, matches in component_matches.items()
        }
        
        # Calculate overall component-wise accuracy
        overall_accuracy = sum(overall_matches) / len(overall_matches) if overall_matches else 0
        
        return {
            "metric": "component_matching",
            "overall_score": overall_accuracy,
            "component_scores": component_scores,
            "matches_by_component": dict(component_matches)
        }
    
    def sql_validity_rate(self, queries: List[str]) -> Dict:
        """
        Calculate the rate of syntactically valid SQL queries.
        
        Args:
            queries: List of SQL queries to validate
            
        Returns:
            Dict with validity score and details
        """
        valid_queries = []
        errors = []
        
        for query in queries:
            try:
                # Try to parse the query
                parsed = sqlparse.parse(query)
                if parsed and len(parsed) > 0:
                    valid_queries.append(True)
                    errors.append(None)
                else:
                    valid_queries.append(False)
                    errors.append("Empty parse result")
            except Exception as e:
                valid_queries.append(False)
                errors.append(str(e))
        
        validity_rate = sum(valid_queries) / len(valid_queries) if valid_queries else 0
        
        return {
            "metric": "sql_validity_rate",
            "score": validity_rate,
            "valid_queries": valid_queries,
            "errors": errors
        }
    
    def semantic_similarity(self, 
                           predicted_queries: List[str], 
                           reference_queries: List[str],
                           include_per_query: bool = False) -> Dict:
        """
        Calculate semantic similarity between queries.
        Uses embedding model if provided, otherwise uses difflib.
        
        Args:
            predicted_queries: List of generated SQL queries
            reference_queries: List of reference SQL queries
            include_per_query: Whether to include per-query similarity scores
            
        Returns:
            Dict with similarity score and details
        """
        if self.embedding_model:
            # Use embeddings for similarity if model is available
            pred_embeddings = self.embedding_model.encode(predicted_queries)
            ref_embeddings = self.embedding_model.encode(reference_queries)
            
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(float(similarity))
        else:
            # Fall back to string similarity with difflib
            similarities = []
            for pred, ref in zip(predicted_queries, reference_queries):
                norm_pred = self.normalize_query(pred)
                norm_ref = self.normalize_query(ref)
                similarity = difflib.SequenceMatcher(None, norm_pred, norm_ref).ratio()
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        result = {
            "metric": "semantic_similarity",
            "score": avg_similarity
        }
        
        if include_per_query:
            result["per_query_similarity"] = similarities
            
        return result
    
    def evaluate_all(self, 
                    predicted_queries: List[str], 
                    reference_queries: List[str],
                    questions: Optional[List[str]] = None,
                    run_execution: bool = True) -> Dict:
        """
        Run all evaluation metrics and return combined results.
        
        Args:
            predicted_queries: List of generated SQL queries
            reference_queries: List of reference SQL queries
            questions: Optional list of natural language questions
            run_execution: Whether to run execution accuracy (requires db_executor)
            
        Returns:
            Dict with all evaluation metrics
        """
        results = {}
        
        # Always run these metrics
        results["exact_match"] = self.exact_match_accuracy(predicted_queries, reference_queries)
        results["component_match"] = self.component_matching(predicted_queries, reference_queries)
        results["validity"] = self.sql_validity_rate(predicted_queries)
        results["similarity"] = self.semantic_similarity(predicted_queries, reference_queries)
        
        # Run execution accuracy if requested and possible
        if run_execution and self.db_executor:
            results["execution"] = self.execution_accuracy(predicted_queries, reference_queries)
        
        # Calculate overall score (weighted average of all metrics)
        metrics = []
        weights = []
        
        if "exact_match" in results:
            metrics.append(results["exact_match"]["score"])
            weights.append(0.35)  # Highest weight to exact matches
            
        if "execution" in results:
            metrics.append(results["execution"]["score"])
            weights.append(0.35)  # Equally high weight to execution
            
        if "component_match" in results:
            metrics.append(results["component_match"]["overall_score"])
            weights.append(0.15)
            
        if "validity" in results:
            metrics.append(results["validity"]["score"])
            weights.append(0.1)
            
        if "similarity" in results:
            metrics.append(results["similarity"]["score"])
            weights.append(0.05)
            
        overall_score = sum(m * w for m, w in zip(metrics, weights)) / sum(weights) if weights else 0
        
        results["overall_score"] = overall_score
        
        # Include per-question evaluation if questions are provided
        if questions:
            results["per_question"] = [
                {
                    "question": q,
                    "predicted_query": p,
                    "reference_query": r,
                    "exact_match": em,
                    "components_matched": cm
                }
                for q, p, r, em, cm in zip(
                    questions,
                    predicted_queries,
                    reference_queries,
                    results["exact_match"]["matches"],
                    [all(matches) for matches in zip(*results["component_match"]["matches_by_component"].values())] 
                    if results["component_match"]["matches_by_component"].values() else [False] * len(predicted_queries)
                )
            ]
            
        return results
    


# Example usage
# from nl2sql_evaluator import NL2SQLEvaluator

# Simple evaluation without database execution
evaluator = NL2SQLEvaluator()

# Example data
predicted_queries = [
    "SELECT name FROM customers WHERE purchase_count > 5",
    "SELECT p.name, SUM(i.price) FROM products p JOIN order_items i ON p.id = i.product_id GROUP BY p.name ORDER BY SUM(i.price) DESC LIMIT 5"
]

reference_queries = [
    "SELECT name FROM customers WHERE purchase_count > 5",
    "SELECT products.name, SUM(order_items.price) FROM products JOIN order_items ON products.id = order_items.product_id GROUP BY products.name ORDER BY SUM(order_items.price) DESC LIMIT 5"
]

questions = [
    "Find all customers who purchased more than 5 items",
    "Find the top 5 products by revenue"
]

# Run basic evaluation
results = evaluator.evaluate_all(
    predicted_queries=predicted_queries,
    reference_queries=reference_queries,
    questions=questions,
    run_execution=False  # No database executor provided
)

print(f"Overall evaluation score: {results['overall_score']:.2f}")
print(f"Exact match accuracy: {results['exact_match']['score']:.2f}")
print(f"Component matching: {results['component_match']['overall_score']:.2f}")
print(f"SQL validity rate: {results['validity']['score']:.2f}")

# Print per-question results
for idx, question_result in enumerate(results['per_question']):
    print(f"\nQuestion {idx+1}: {question_result['question']}")
    print(f"Exact match: {'✓' if question_result['exact_match'] else '✗'}")