#!/usr/bin/env python3
"""
Main script to run the comprehensive code evaluation
Run this from the project root directory
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluators.comprehensive_evaluator import evaluate_code, print_evaluation_report


def main():
    """Main function to run the comprehensive evaluation."""
    
    # Default file paths
    correct_code_path = "example_code/corrected_code/correct_code.py"
    generated_code_path = "example_code/generated_code/generated_code.py"
    
    # Use command line arguments if provided
    if len(sys.argv) >= 3:
        correct_code_path = sys.argv[1]
        generated_code_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(correct_code_path):
        print(f"Error: Correct code file not found: {correct_code_path}")
        sys.exit(1)
    
    if not os.path.exists(generated_code_path):
        print(f"Error: Generated code file not found: {generated_code_path}")
        sys.exit(1)
    
    print(f"Evaluating code files:")
    print(f"  Correct: {correct_code_path}")
    print(f"  Generated: {generated_code_path}")
    print()
    
    try:
        # Run comprehensive evaluation
        results = evaluate_code(correct_code_path, generated_code_path)
        
        # Print detailed report
        print_evaluation_report(results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 