"""
Code Similarity Analysis Module
Analyzes similarity between code snippets using various metrics
"""
import ast
import difflib
from typing import Dict, Any


def calculate_similarity(correct_code: str, generated_code: str) -> Dict[str, float]:
    """
    Calculate similarity between two code snippets using difflib.
    
    Args:
        correct_code: The reference/correct code
        generated_code: The code to compare against
        
    Returns:
        Dictionary with line_similarity and char_similarity scores
    """
    # Line-based similarity
    correct_lines = correct_code.splitlines()
    generated_lines = generated_code.splitlines()

    matcher = difflib.SequenceMatcher(None, correct_lines, generated_lines)
    line_similarity = matcher.ratio()

    # Character-based similarity
    char_matcher = difflib.SequenceMatcher(None, correct_code, generated_code)
    char_similarity = char_matcher.ratio()

    return {
        'line_similarity': line_similarity,
        'char_similarity': char_similarity
    }


def count_ast_nodes(code: str) -> int:
    """
    Count AST nodes in the code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Number of AST nodes, or -1 if syntax error
    """
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return -1  # Indicates syntax error


def analyze_code_structure(correct_code: str, generated_code: str) -> Dict[str, Any]:
    """
    Analyze the structure of both code snippets.
    
    Args:
        correct_code: The reference/correct code
        generated_code: The code to compare against
        
    Returns:
        Dictionary with AST node counts and differences
    """
    correct_ast_nodes = count_ast_nodes(correct_code)
    generated_ast_nodes = count_ast_nodes(generated_code)
    
    return {
        'correct': correct_ast_nodes,
        'generated': generated_ast_nodes,
        'difference': generated_ast_nodes - correct_ast_nodes
    } 
if __name__ == "__main__":
    correct_code = open("example_code/corrected_code/correct_code.py", "r").read()
    generated_code = open("example_code/generated_code/generated_code.py", "r").read()
    print(analyze_code_structure(correct_code, generated_code))
    print(calculate_similarity(correct_code, generated_code))