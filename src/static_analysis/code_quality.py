"""
Code Quality Analysis Module
Analyzes code quality using static analysis tools like pylint
"""
import os
import tempfile
from io import StringIO
from typing import Dict, Any
from pylint.lint import Run
from pylint.reporters.text import TextReporter
import pprint

def run_pylint(code: str) -> float:
    """
    Run pylint on the code and return the score.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Pylint score (0-10), or -1 if error
    """
    try:
        # Create a temporary file for pylint
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code)

        # Capture pylint output
        pylint_output = StringIO()
        reporter = TextReporter(pylint_output)

        # Run pylint with captured output
        Run([temp_file_path, '--output-format=text'], reporter=reporter, exit=False)

        # Get the output as string
        output = pylint_output.getvalue()

        # Extract score
        score_line = [line for line in output.split('\n') if 'Your code has been rated at' in line]
        if score_line:
            try:
                score_part = score_line[0].split('at ')[1].split('/')[0]
                score = float(score_part)
            except (IndexError, ValueError):
                score = 0
        else:
            score = 0

        # Clean up
        os.remove(temp_file_path)

        return score
    except Exception as e:
        print(f"Error running pylint: {e}")
        return -1


def analyze_code_quality(correct_code: str, generated_code: str) -> Dict[str, Any]:
    """
    Analyze code quality of both code snippets.
    
    Args:
        correct_code: The reference/correct code
        generated_code: The code to compare against
        
    Returns:
        Dictionary with pylint scores and differences
    """
    correct_pylint = run_pylint(correct_code)
    generated_pylint = run_pylint(generated_code)
    
    return {
        'correct': correct_pylint,
        'generated': generated_pylint,
        'difference': generated_pylint - correct_pylint
    } 
if __name__ == "__main__":
    correct_code = open("example_code/corrected_code/correct_code.py", "r").read()
    generated_code = open("example_code/generated_code/generated_code.py", "r").read()
    pprint.pprint(analyze_code_quality(correct_code, generated_code))