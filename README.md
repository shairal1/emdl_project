# EMDL Project - Machine Learning Code Evaluation System

A comprehensive Python framework for evaluating and comparing machine learning code through both static and dynamic analysis.

## 🎯 Project Overview

This project provides tools to analyze and compare machine learning code implementations, helping to:
- Evaluate code quality and correctness
- Compare different ML implementations
- Detect ML-specific issues and patterns
- Measure execution performance
- Generate comprehensive evaluation reports

## 📁 Project Structure

```
emdl_project/
├── src/                          # Main source code
│   ├── static_analysis/          # Static code analysis
│   │   ├── code_similarity.py    # Code similarity metrics
│   │   ├── code_quality.py       # Code quality analysis (pylint)
│   │   └── ml_analysis.py        # ML-specific static analysis
│   ├── dynamic_analysis/         # Runtime analysis
│   │   └── execution_metrics.py  # Execution time, memory, ML metrics
│   ├── evaluators/               # Comprehensive evaluators
│   │   └── comprehensive_evaluator.py
│   └── utils/                    # Utility functions
│       └── file_utils.py
├── example_code/                 # Example code files
│   ├── corrected_code/           # Reference implementations
│   ├── generated_code/           # Generated/test implementations
│   └── incorrect_code/           # Incorrect implementations
├── dataset/                      # ML datasets
├── analysis/                     # Analysis results
├── utils/                        # Legacy utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd emdl_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pylint psutil numpy pandas scikit-learn
```

### 2. Basic Usage

```bash
# Run comprehensive evaluation
python run_evaluation.py

# Or specify custom file paths
python run_evaluation.py path/to/correct_code.py path/to/generated_code.py
```

### 3. Individual Module Testing

```bash
# Test static analysis modules
python src/static_analysis/code_similarity.py
python src/static_analysis/code_quality.py
python src/static_analysis/ml_analysis.py

# Test dynamic analysis
python src/dynamic_analysis/execution_metrics.py
```

## 🔍 Analysis Features

### Static Analysis

#### Code Similarity Analysis
- **Line-based similarity**: Compares code line by line
- **Character-based similarity**: Compares character sequences
- **AST structure analysis**: Compares abstract syntax trees

```python
from src.static_analysis.code_similarity import calculate_similarity

similarity = calculate_similarity(correct_code, generated_code)
print(f"Similarity: {similarity['char_similarity']:.2f}")
```

#### Code Quality Analysis
- **Pylint integration**: Automated code quality scoring
- **Style and convention checking**
- **Error detection**

```python
from src.static_analysis.code_quality import analyze_code_quality

quality = analyze_code_quality(correct_code, generated_code)
print(f"Quality score: {quality['generated']:.2f}/10")
```

#### ML-Specific Analysis
- **Import statement analysis**: Detects missing/extra imports
- **Scikit-learn component detection**: Identifies ML libraries used
- **Preprocessing step identification**: Detects data transformations
- **Model parameter extraction**: Analyzes model configurations
- **ML issue detection**: Identifies common ML problems

```python
from src.static_analysis.ml_analysis import analyze_ml_components, detect_ml_issues

ml_analysis = analyze_ml_components(correct_code, generated_code)
issues = detect_ml_issues(ml_analysis)
```

### Dynamic Analysis

#### Execution Metrics
- **Execution time measurement**: Tracks runtime performance
- **Memory usage monitoring**: Measures memory consumption
- **Success/failure detection**: Identifies execution errors
- **ML metrics extraction**: Captures accuracy, precision, recall, F1-score

```python
from src.dynamic_analysis.execution_metrics import measure_execution_time_and_memory

metrics = measure_execution_time_and_memory(code)
print(f"Execution time: {metrics['execution_time']:.3f}s")
print(f"Memory used: {metrics['memory_used_mb']:.2f} MB")
```

## 📊 Evaluation Reports

The system generates comprehensive reports including:

### Static Analysis Report
- Similarity metrics (line and character-based)
- Code structure comparison (AST nodes)
- Code quality scores (pylint)
- ML component differences
- Preprocessing step analysis
- Model parameter comparison
- Detected ML issues

### Dynamic Analysis Report
- Execution time comparison
- Memory usage analysis
- Success rate comparison
- ML metrics from execution
- Performance differences

### Overall Assessment
- **Weighted scoring system**: Combines all metrics
- **Verdict classification**: Excellent/Good/Acceptable/Poor/Unacceptable
- **Detailed breakdown**: Individual component scores

## 🛠️ Advanced Usage

### Programmatic Integration

```python
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluators.comprehensive_evaluator import evaluate_code, print_evaluation_report

# Run comprehensive evaluation
results = evaluate_code("correct_code.py", "generated_code.py")

# Access specific results
static_analysis = results['static_analysis']
dynamic_analysis = results['dynamic_analysis']
overall_assessment = results['overall_assessment']

# Print detailed report
print_evaluation_report(results)
```

### Custom Analysis

```python
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Use individual modules
from static_analysis.ml_analysis import extract_sklearn_components
from dynamic_analysis.execution_metrics import extract_ml_metrics_from_output

# Extract specific information
components = extract_sklearn_components(code)
metrics = extract_ml_metrics_from_output(execution_output)
```

## 📋 Requirements

### Python Dependencies
- `pylint`: Code quality analysis
- `psutil`: System and process utilities
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities

### System Requirements
- Python 3.7+
- Sufficient memory for code execution
- Access to execute Python code (for dynamic analysis)

## 🔧 Configuration

### Pylint Configuration
The code quality analysis uses pylint with default settings. You can customize by modifying `src/static_analysis/code_quality.py`.

### Execution Timeout
Dynamic analysis has a default timeout of 30 seconds. Modify the `timeout` parameter in `measure_execution_time_and_memory()`.

### Scoring Weights
Overall assessment weights can be adjusted in `calculate_overall_assessment()`:
- Similarity: 25%
- Code Quality: 20%
- ML Correctness: 30%
- Execution: 25%

## 🧪 Testing

### Test Individual Modules
```bash
# Test static analysis
python src/static_analysis/code_similarity.py
python src/static_analysis/code_quality.py
python src/static_analysis/ml_analysis.py

# Test dynamic analysis
python src/dynamic_analysis/execution_metrics.py
```

### Test with Example Code
```bash
# Use provided example code
python run_evaluation.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/emdl_project
   # Add src to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Pylint Not Found**
   ```bash
   pip install pylint
   ```

3. **Memory Issues**
   - Reduce timeout in dynamic analysis
   - Check available system memory

4. **Code Execution Errors**
   - Ensure code files are valid Python
   - Check for missing dependencies in analyzed code

### Debug Mode
Add debug prints to individual modules or use Python's built-in debugger:
```python
import pdb; pdb.set_trace()
```

## 🤝 Contributing

### Adding New Analysis Types
1. Create new module in appropriate directory (`static_analysis/` or `dynamic_analysis/`)
2. Implement analysis functions
3. Update `comprehensive_evaluator.py` to include new analysis
4. Add tests and documentation

### Code Style
- Follow PEP 8 guidelines
- Add type hints
- Include docstrings
- Test individual modules

## 📄 License

[Add your license information here]

## 👥 Authors

[Add author information here]

## 🙏 Acknowledgments

- Pylint for code quality analysis
- Scikit-learn for ML component detection
- The Python community for various utilities

---

**Note**: This project is designed for educational and research purposes. Always review and validate results before making decisions based on automated analysis.

