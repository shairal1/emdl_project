#%%
"""
Comprehensive Code Evaluator
Combines static and dynamic analysis for complete code evaluation
"""
import sys
import os
from typing import Dict, Any
from pathlib import Path
from typing import List
import logging

from pandas.plotting import plot_params
# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modular components
from static_analysis.code_similarity import calculate_similarity, analyze_code_structure
from static_analysis.code_quality import analyze_code_quality
from static_analysis.ml_analysis import analyze_ml_components, detect_ml_issues
from dynamic_analysis.execution_metrics import (
    measure_execution_time_and_memory, 
    analyze_execution_differences
)
from utils.file_utils import load_code_from_file
import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_code(correct_code: str, generated_code: str, 
                 run_dynamic_analysis: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of code combining static and dynamic analysis.
    
    Args:
        correct_code_path: Path to the reference/correct code
        generated_code_path: Path to the generated code to evaluate
        run_dynamic_analysis: Whether to run dynamic analysis (default: True)
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    # Load code files
    
    
    if correct_code is None or generated_code is None:
        raise ValueError("Could not load one or both code files")

    # Static Analysis
    similarity = calculate_similarity(correct_code, generated_code)
    code_structure = analyze_code_structure(correct_code, generated_code)
    #code_quality = analyze_code_quality(correct_code, generated_code)
    ml_analysis = analyze_ml_components(correct_code, generated_code)
    ml_issues = detect_ml_issues(ml_analysis)

    # Compile static analysis results
    static_results = {
        'similarity': similarity,
        'code_structure': code_structure,
        #'code_quality': code_quality,
        'ml_analysis': ml_analysis,
        'ml_issues': ml_issues
    }

    # Dynamic Analysis (optional)
    dynamic_results = {}
    if run_dynamic_analysis:
        try:
            correct_metrics = measure_execution_time_and_memory(correct_code)
            generated_metrics = measure_execution_time_and_memory(generated_code)
            dynamic_results = analyze_execution_differences(correct_metrics, generated_metrics)
        except Exception as e:
            print(f"Warning: Dynamic analysis failed: {e}")
            dynamic_results = {'error': str(e)}

    # Compile final results
    results = {
        'static_analysis': static_results,
        'dynamic_analysis': dynamic_results,
        #'overall_assessment': calculate_overall_assessment(static_results, dynamic_results)
    }

    return results


def calculate_overall_assessment(static_results: Dict[str, Any], 
                               dynamic_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overall assessment scores based on static and dynamic analysis.
    
    Args:
        static_results: Results from static analysis
        dynamic_results: Results from dynamic analysis
        
    Returns:
        Dictionary with overall assessment scores
    """
    # Calculate similarity score (weighted average)
    similarity_score = (0.7 * static_results['similarity']['char_similarity'] + 
                       0.3 * static_results['similarity']['line_similarity'])

    # Calculate code quality score
    quality_ratio = 1.0
    if static_results['code_quality']['correct'] > 0:
        quality_ratio = static_results['code_quality']['generated'] / static_results['code_quality']['correct']
        quality_ratio = max(0, min(2, quality_ratio))  # Cap at 2x

    # Calculate ML correctness score
    ml_issues = len(static_results['ml_issues'])
    ml_score = max(0, 1.0 - (ml_issues * 0.2))  # Deduct 0.2 for each issue, minimum 0

    # Calculate execution success score
    execution_score = 1.0
    if dynamic_results and 'success_rate' in dynamic_results:
        correct_success = dynamic_results['success_rate']['correct']
        generated_success = dynamic_results['success_rate']['generated']
        if correct_success and not generated_success:
            execution_score = 0.0
        elif not correct_success and generated_success:
            execution_score = 1.5  # Bonus for fixing broken code
        elif correct_success and generated_success:
            execution_score = 1.0
        else:
            execution_score = 0.5  # Both failed

    # Overall score (weighted average)
    weights = {
        'similarity': 0.25,
        'quality': 0.20,
        'ml_correctness': 0.30,
        'execution': 0.25
    }
    
    overall_score = (
        weights['similarity'] * similarity_score +
        weights['quality'] * quality_ratio +
        weights['ml_correctness'] * ml_score +
        weights['execution'] * execution_score
    ) * 10

    # Determine verdict
    if overall_score >= 8.5:
        verdict = "Excellent"
    elif overall_score >= 7.0:
        verdict = "Good"
    elif overall_score >= 5.0:
        verdict = "Acceptable"
    elif overall_score >= 3.0:
        verdict = "Poor"
    else:
        verdict = "Unacceptable"

    return {
        'similarity_score': similarity_score,
        'quality_ratio': quality_ratio,
        'ml_correctness_score': ml_score,
        'execution_score': execution_score,
        'overall_score': overall_score,
        'verdict': verdict
    }


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted comprehensive evaluation report.
    
    Args:
        results: Results from evaluate_code function
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE CODE EVALUATION REPORT")
    print("="*70)

    # Static Analysis Section
    static = results['static_analysis']
    
    print("\n" + "-"*30)
    print("STATIC ANALYSIS")
    print("-"*30)

    print("\nSIMILARITY METRICS:")
    print(f"Line-based similarity: {static['similarity']['line_similarity']:.2f} (0-1, higher is more similar)")
    print(f"Character-based similarity: {static['similarity']['char_similarity']:.2f} (0-1, higher is more similar)")

    print("\nCODE STRUCTURE:")
    print(f"AST nodes in correct code: {static['code_structure']['correct']}")
    print(f"AST nodes in generated code: {static['code_structure']['generated']}")
    print(f"Difference: {static['code_structure']['difference']} nodes")

    #print("\nCODE QUALITY:")
    #print(f"Pylint score (correct): {static['code_quality']['correct']:.2f}/10")
    #print(f"Pylint score (generated): {static['code_quality']['generated']:.2f}/10")
    #print(f"Difference: {static['code_quality']['difference']:.2f} points")

    # ML Analysis Section
    print("\n" + "-"*30)
    print("ML-SPECIFIC ANALYSIS")
    print("-"*30)

    ml_analysis = static['ml_analysis']
    
    print("\nIMPORT DIFFERENCES:")
    if ml_analysis['imports']['differences']['missing']:
        print("  Missing imports in generated code:")
        for imp in ml_analysis['imports']['differences']['missing']:
            print(f"    - {imp}")
    if ml_analysis['imports']['differences']['extra']:
        print("  Extra imports in generated code:")
        for imp in ml_analysis['imports']['differences']['extra']:
            print(f"    - {imp}")
    if not ml_analysis['imports']['differences']['missing'] and not ml_analysis['imports']['differences']['extra']:
        print("  No differences in imports")

    print("\nSCIKIT-LEARN COMPONENT DIFFERENCES:")
    if ml_analysis['sklearn_components']['differences']['missing']:
        print("  Missing components in generated code:")
        for comp in ml_analysis['sklearn_components']['differences']['missing']:
            print(f"    - {comp}")
    if ml_analysis['sklearn_components']['differences']['extra']:
        print("  Extra components in generated code:")
        for comp in ml_analysis['sklearn_components']['differences']['extra']:
            print(f"    - {comp}")
    if not ml_analysis['sklearn_components']['differences']['missing'] and not ml_analysis['sklearn_components']['differences']['extra']:
        print("  No differences in scikit-learn components")

    print("\nPREPROCESSING DIFFERENCES:")
    if ml_analysis['preprocessing']['differences']['missing']:
        print("  Missing preprocessing steps in generated code:")
        for step in ml_analysis['preprocessing']['differences']['missing']:
            print(f"    - {step}")
    if ml_analysis['preprocessing']['differences']['extra']:
        print("  Extra preprocessing steps in generated code:")
        for step in ml_analysis['preprocessing']['differences']['extra']:
            print(f"    - {step}")
    if not ml_analysis['preprocessing']['differences']['missing'] and not ml_analysis['preprocessing']['differences']['extra']:
        print("  No differences in preprocessing steps")

    # ML Issues Section
    print("\nDETECTED ML ISSUES:")
    if static['ml_issues']:
        for i, issue in enumerate(static['ml_issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print("  No significant ML-specific issues detected")

    # Dynamic Analysis Section
    if 'dynamic_analysis' in results and results['dynamic_analysis']:
        dynamic = results['dynamic_analysis']
        
        print("\n" + "-"*30)
        print("DYNAMIC ANALYSIS")
        print("-"*30)

        if 'error' in dynamic:
            print(f"\nDynamic analysis error: {dynamic['error']}")
        else:
            print("\nEXECUTION METRICS:")
            print(f"Execution time (correct): {dynamic['execution_time']['correct']:.3f}s")
            print(f"Execution time (generated): {dynamic['execution_time']['generated']:.3f}s")
            print(f"Time difference: {dynamic['execution_time']['difference']:.3f}s")

            print(f"\nMemory usage (correct): {dynamic['memory_usage']['correct']:.2f} MB")
            print(f"Memory usage (generated): {dynamic['memory_usage']['generated']:.2f} MB")
            print(f"Memory difference: {dynamic['memory_usage']['difference']:.2f} MB")

            #print(f"\nSuccess rate (correct): {dynamic['success_rate']['correct']}")
            #print(f"Success rate (generated): {dynamic['success_rate']['generated']}")

            # ML Metrics from execution
            if dynamic['ml_metrics']['correct'] or dynamic['ml_metrics']['generated']:
                print("\nML METRICS FROM EXECUTION:")
                correct_metrics = dynamic['ml_metrics']['correct']
                generated_metrics = dynamic['ml_metrics']['generated']
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in correct_metrics or metric in generated_metrics:
                        correct_val = correct_metrics.get(metric, 'N/A')
                        generated_val = generated_metrics.get(metric, 'N/A')
                        print(f"  {metric.capitalize()}: {correct_val} (correct) vs {generated_val} (generated)")

    # Overall Assessment Section
    #assessment = results['overall_assessment']
    
    #print("\n" + "-"*30)
    #print("OVERALL ASSESSMENT")
    #print("-"*30)

    #print(f"Similarity Score: {assessment['similarity_score']:.2f}/1.0")
    #print(f"Code Quality Score: {assessment['quality_ratio']:.2f}/1.0")
    #print(f"ML Correctness Score: {assessment['ml_correctness_score']:.2f}/1.0")
    #print(f"Execution Score: {assessment['execution_score']:.2f}/1.0")
    #print(f"Overall Score: {assessment['overall_score']:.1f}/10.0")
    #print(f"\nVerdict: {assessment['verdict']}")
    
    print("="*70)
def gather_examples_with_fixed_code(root: Path,model_selection:str) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for script in root.glob("pipelines_old/*/example-0.py"):
        try:
            ai_fixed_path = script.parent / model_selection
            fixed_path = script.parent / "example-0-fixed.py"
            explation_path = script.parent / "example-0-explanation.md"

            if not ai_fixed_path.exists():
                logging.info("Skipping %s, fixed.py not exists.", script.relative_to(root))
                continue

            if not fixed_path.exists():
                logging.info("Skipping %s, example-0-fixed.py not exists.", script.relative_to(root))
                continue

            if not explation_path.exists():
                logging.info("Skipping %s, example-0-explanation.md not exists.", script.relative_to(root))
                continue

            content = script.read_text(encoding="utf-8")
            ai_fixed = ai_fixed_path.read_text(encoding="utf-8")
            fixed =  fixed_path.read_text(encoding="utf-8")
            explanation =  explation_path.read_text(encoding="utf-8")
            

            examples.append(
                {
                    "name": script.parent.name,
                    "path": script.parent,
                    "content": content,
                    "ai_fixed_content": ai_fixed,
                    "fixed_content": fixed,
                    "explanation": explanation
                }
            )
            logging.info("Loaded %s", script.relative_to(root))
        except Exception:
            logging.exception("Unable to read %s", script)
    return examples

if __name__ == "__main__":
    '''if len(sys.argv) != 3:
        print("Usage: python comprehensive_evaluator.py <correct_code_path> <generated_code_path>")
        sys.exit(1)'''

    '''correct_code_path = sys.argv[1]
    generated_code_path = sys.argv[2]

    try:
        results = evaluate_code(correct_code_path, generated_code_path)
        print_evaluation_report(results)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1) '''
    root = Path(__file__).parent.parent.parent / "openhands-automation"
    model={'openhands':'fixed','gemini':'example-0_Gemini'}
    selected_model='openhands'
    examples = gather_examples_with_fixed_code(root,model[selected_model])

    ''' for example in examples:
        correct_code = example['fixed_content']
        generated_code = example['ai_fixed_content']
        #print(example['name'])
        code=generated_code
        code = code.replace('\x00', '')

        results = evaluate_code(correct_code, generated_code)
        print_evaluation_report(results)'''

       
    results_list = []
    results_list_ml = []
    for i, example in enumerate(examples):
        correct_code = example['fixed_content']
        generated_code = example['ai_fixed_content']
        #generated_code = generated_code.replace('\x00', '')
        results = evaluate_code(correct_code, generated_code)
        static = results['static_analysis']
        dynamic = results['dynamic_analysis']

        # Collect general metrics
        row = {
            "Example": example['name'],
            "Line Similarity": static['similarity']['line_similarity'],
            "Char Similarity": static['similarity']['char_similarity'],
            "AST Node Diff": static['code_structure']['difference'],
            "ML Issues Count": len(static['ml_issues']),
            "ML Issues": "; ".join(static['ml_issues'])
        }

        # Collect ML analysis details
        row_2 = {
            "Example": example['name'],
            "Extra imports in Generated code": "\n".join(static['ml_analysis']['imports']['differences']['extra']),
            "Missing imports in Generated code": "\n".join(static['ml_analysis']['imports']['differences']['missing']),
            "Extra sklearn components": "\n".join(static['ml_analysis']['sklearn_components']['differences']['extra']),
            "Missing sklearn components": "\n".join(static['ml_analysis']['sklearn_components']['differences']['missing']),
            "Extra preprocessing steps": "\n".join(static['ml_analysis']['preprocessing']['differences']['extra']),
            "Missing preprocessing steps": "\n".join(static['ml_analysis']['preprocessing']['differences']['missing']),
        }

        # Add dynamic analysis if present
        if dynamic and 'execution_time' in dynamic:
            row["Execution Time Diff"] = dynamic['execution_time']['difference']
        else:
            row["Execution Time Diff"] = None

        if dynamic and 'memory_usage' in dynamic:
            row["Memory Usage Diff"] = dynamic['memory_usage']['difference']
        else:
            row["Memory Usage Diff"] = None

        # Add ML metrics if present
        ml_metrics = dynamic.get('ml_metrics', {})
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            row[f"ML Metric {metric} (correct)"] = ml_metrics.get('correct', {}).get(metric)
            row[f"ML Metric {metric} (generated)"] = ml_metrics.get('generated', {}).get(metric)

        results_list.append(row)
        results_list_ml.append(row_2)


    # Convert to DataFrames after the loop
    df = pd.DataFrame(results_list)
    df_ml = pd.DataFrame(results_list_ml)

    # Save as CSV
    df.to_csv("comprehensive_report_general.csv", index=False)
    df_ml.to_csv("comprehensive_report_ml.csv", index=False)

    # Save as LaTeX table (for LaTeX papers)
    df.to_latex("comprehensive_report_general.tex", index=False)
    df_ml.to_latex("comprehensive_report_ml.tex", index=False)

    dir_path = os.path.join('plots', selected_model)
    os.makedirs(dir_path, exist_ok=True)
    # Bar plot: Line Similarity per Example
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Example', y='Line Similarity', data=df)
    plt.xticks(rotation=90)
    plt.title('Line Similarity per Example')
    plt.tight_layout()
    file_path = os.path.join(dir_path, 'plot_line_similarity.png')
    plt.savefig(file_path)
    plt.close()

    # Bar plot: Char Similarity per Example
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Example', y='Char Similarity', data=df)
    plt.xticks(rotation=90)
    plt.title('Char Similarity per Example')
    plt.tight_layout()
    file_path = os.path.join(dir_path, 'plot_char_similarity.png')
    plt.savefig(file_path)
    plt.close()
    '''
    # Bar plot: ML Issues Count per Example
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Example', y='ML Issues Count', data=df)
    plt.xticks(rotation=90)
    plt.title('ML Issues Count per Example')
    plt.tight_layout()
    plt.savefig('plot_ml_issues_count.png')
    plt.close()'''
    # Turn dict to single numeric value (sum of absolute differences)
    df['AST Node Diff Total'] = df['AST Node Diff'].apply(
        lambda d: sum(abs(v) for v in d.values()) if isinstance(d, dict) else None
    )

    # Pie chart: AST Node Types Distribution
    ast_types_count = {}
    for ast_diff in df['AST Node Diff']:
        if isinstance(ast_diff, dict):
            for ast_type, count in ast_diff.items():
                if ast_type in ast_types_count:
                    ast_types_count[ast_type] += abs(count)
                else:
                    ast_types_count[ast_type] = abs(count)
    
    if ast_types_count:
        plt.figure(figsize=(15, 8))
        labels = list(ast_types_count.keys())
        sizes = list(ast_types_count.values())
        
        '''# Create pie chart
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of AST Node Type Changes')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        plt.savefig('plot_ast_types_pie.png')
        plt.close()'''
        
        # Also create a bar plot for better readability of large numbers
        plt.figure(figsize=(15, 6))
        plt.bar(labels, sizes)
        plt.xticks(rotation=45, ha='right')
        plt.title('AST Node Type Changes Count')
        plt.ylabel('Count')
        plt.tight_layout()
        file_path = os.path.join(dir_path, 'plot_ast_types_bar.png')
        plt.savefig(file_path)
        plt.close()


    