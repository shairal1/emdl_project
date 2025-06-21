"""
Execution Metrics Module
Measures code execution performance and captures runtime behavior
"""
import sys
import subprocess
import time
import psutil
import tempfile
import os
from typing import Dict, Any, Optional
import pprint

def measure_execution_time_and_memory(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code and measure execution time and memory usage.
    
    Args:
        code: Source code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution metrics
    """
    try:
        # Create temporary file for execution
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code)

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute code and measure time
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        execution_time = time.time() - start_time

        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory

        # Clean up
        os.remove(temp_file_path)

        return {
            'execution_time': execution_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': peak_memory,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }

    except subprocess.TimeoutExpired:
        return {
            'execution_time': timeout,
            'memory_used_mb': 0,
            'peak_memory_mb': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Execution timed out',
            'success': False
        }
    except Exception as e:
        return {
            'execution_time': 0,
            'memory_used_mb': 0,
            'peak_memory_mb': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


def extract_ml_metrics_from_output(output: str) -> Dict[str, Any]:
    """
    Extract ML metrics from code execution output.
    
    Args:
        output: Standard output from code execution
        
    Returns:
        Dictionary with extracted ML metrics
    """
    metrics = {}
    
    # Look for common ML metric patterns
    import re
    
    # Accuracy
    accuracy_match = re.search(r'accuracy[:\s]*([0-9]*\.?[0-9]+)', output, re.IGNORECASE)
    if accuracy_match:
        metrics['accuracy'] = float(accuracy_match.group(1))
    
    # Precision
    precision_match = re.search(r'precision[:\s]*([0-9]*\.?[0-9]+)', output, re.IGNORECASE)
    if precision_match:
        metrics['precision'] = float(precision_match.group(1))
    
    # Recall
    recall_match = re.search(r'recall[:\s]*([0-9]*\.?[0-9]+)', output, re.IGNORECASE)
    if recall_match:
        metrics['recall'] = float(recall_match.group(1))
    
    # F1-score
    f1_match = re.search(r'f1[:\s-]*score[:\s]*([0-9]*\.?[0-9]+)', output, re.IGNORECASE)
    if f1_match:
        metrics['f1_score'] = float(f1_match.group(1))
    
    # Model type detection
    if 'RandomForestClassifier' in output:
        metrics['model_type'] = 'RandomForest'
    elif 'LogisticRegression' in output:
        metrics['model_type'] = 'LogisticRegression'
    elif 'SVC' in output or 'SVM' in output:
        metrics['model_type'] = 'SVM'
    elif 'KNeighborsClassifier' in output:
        metrics['model_type'] = 'KNN'
    
    return metrics


def analyze_execution_differences(correct_metrics: Dict[str, Any], 
                                generated_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare execution metrics between correct and generated code.
    
    Args:
        correct_metrics: Metrics from correct code execution
        generated_metrics: Metrics from generated code execution
        
    Returns:
        Dictionary with execution differences
    """
    return {
        'execution_time': {
            'correct': correct_metrics.get('execution_time', 0),
            'generated': generated_metrics.get('execution_time', 0),
            'difference': generated_metrics.get('execution_time', 0) - correct_metrics.get('execution_time', 0)
        },
        'memory_usage': {
            'correct': correct_metrics.get('memory_used_mb', 0),
            'generated': generated_metrics.get('memory_used_mb', 0),
            'difference': generated_metrics.get('memory_used_mb', 0) - correct_metrics.get('memory_used_mb', 0)
        },
        'success_rate': {
            'correct': correct_metrics.get('success', False),
            'generated': generated_metrics.get('success', False)
        },
        'ml_metrics': {
            'correct': extract_ml_metrics_from_output(correct_metrics.get('stdout', '')),
            'generated': extract_ml_metrics_from_output(generated_metrics.get('stdout', ''))
        }
    } 
if __name__ == "__main__":
    correct_code = open("example_code/corrected_code/correct_code.py", "r").read()
    generated_code = open("example_code/generated_code/generated_code.py", "r").read()
    correct_metrics = measure_execution_time_and_memory(correct_code)
    generated_metrics = measure_execution_time_and_memory(generated_code)
    pprint.pprint(analyze_execution_differences(correct_metrics, generated_metrics))