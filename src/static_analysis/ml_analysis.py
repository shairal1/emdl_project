"""
ML-Specific Static Analysis Module
Analyzes machine learning code for specific patterns and components
"""
import ast
import re
from typing import Dict, List, Any
import pprint

def extract_imports(code: str) -> List[str]:
    """
    Extract import statements from code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        List of imported modules/classes
    """
    try:
        tree = ast.parse(code)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for name in node.names:
                    imports.append(f"{module}.{name.name}")

        return imports
    except SyntaxError:
        return []


def extract_sklearn_components(code: str) -> List[str]:
    """
    Extract scikit-learn components used in the code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        List of sklearn components used
    """
    # Use regex to find sklearn components
    sklearn_pattern = r'from\s+sklearn\.(\w+)\s+import\s+([^#\n]+)'
    matches = re.findall(sklearn_pattern, code)

    components = []
    for module, imports in matches:
        for imp in imports.split(','):
            imp = imp.strip()
            if imp:
                components.append(f"{module}.{imp}")

    return components


def extract_preprocessing_steps(code: str) -> List[str]:
    """
    Extract preprocessing steps from the code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        List of preprocessing transformations detected
    """
    # Look for data transformations
    transformations = []

    # Text normalization
    if "str.lower()" in code:
        transformations.append("text_lowercasing")
    if "str.strip()" in code:
        transformations.append("text_stripping")
    if "str.replace" in code:
        transformations.append("text_replacement")

    # Spatial aggregation
    if "apply(lambda" in code and "native-country" in code:
        transformations.append("spatial_aggregation")

    # Scaling/normalization
    if "StandardScaler" in code:
        transformations.append("standard_scaling")
    if "Normalizer" in code:
        transformations.append("sample_normalization")
    if "MinMaxScaler" in code:
        transformations.append("minmax_scaling")

    # Imputation
    if "SimpleImputer" in code:
        transformations.append("simple_imputation")

    # Encoding
    if "OneHotEncoder" in code:
        transformations.append("onehot_encoding")

    return transformations


def extract_model_parameters(code: str) -> Dict[str, Any]:
    """
    Extract model parameters from the code.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Dictionary of model parameters
    """
    params = {}

    # Check for random_state
    random_state_match = re.search(r'RandomForestClassifier\(([^)]*)\)', code)
    if random_state_match:
        param_str = random_state_match.group(1)
        if "random_state" in param_str:
            rs_match = re.search(r'random_state=(\d+)', param_str)
            if rs_match:
                params["random_state"] = int(rs_match.group(1))
            else:
                params["random_state"] = True

    return params


def analyze_ml_components(correct_code: str, generated_code: str) -> Dict[str, Any]:
    """
    Analyze ML-specific components in both code snippets.
    
    Args:
        correct_code: The reference/correct code
        generated_code: The code to compare against
        
    Returns:
        Dictionary with ML component analysis
    """
    # Extract ML components
    correct_imports = extract_imports(correct_code)
    generated_imports = extract_imports(generated_code)
    correct_components = extract_sklearn_components(correct_code)
    generated_components = extract_sklearn_components(generated_code)
    correct_preprocessing = extract_preprocessing_steps(correct_code)
    generated_preprocessing = extract_preprocessing_steps(generated_code)
    correct_params = extract_model_parameters(correct_code)
    generated_params = extract_model_parameters(generated_code)

    return {
        'imports': {
            'correct': correct_imports,
            'generated': generated_imports,
            'differences': {
                'missing': [imp for imp in correct_imports if imp not in generated_imports],
                'extra': [imp for imp in generated_imports if imp not in correct_imports]
            }
        },
        'sklearn_components': {
            'correct': correct_components,
            'generated': generated_components,
            'differences': {
                'missing': [comp for comp in correct_components if comp not in generated_components],
                'extra': [comp for comp in generated_components if comp not in correct_components]
            }
        },
        'preprocessing': {
            'correct': correct_preprocessing,
            'generated': generated_preprocessing,
            'differences': {
                'missing': [step for step in correct_preprocessing if step not in generated_preprocessing],
                'extra': [step for step in generated_preprocessing if step not in correct_preprocessing]
            }
        },
        'model_params': {
            'correct': correct_params,
            'generated': generated_params
        }
    }


def detect_ml_issues(ml_analysis: Dict[str, Any]) -> List[str]:
    """
    Detect potential ML-specific issues in the generated code.
    
    Args:
        ml_analysis: Results from analyze_ml_components
        
    Returns:
        List of detected issues
    """
    issues = []

    # Check for StandardScaler vs Normalizer
    if ("preprocessing.StandardScaler" in ml_analysis['sklearn_components']['correct'] and 
        "preprocessing.Normalizer" in ml_analysis['sklearn_components']['generated']):
        issues.append("Generated code uses Normalizer (row-wise normalization) instead of StandardScaler (column-wise standardization)")

    # Check for text normalization differences
    if ("text_lowercasing" in ml_analysis['preprocessing']['generated'] and 
        "text_lowercasing" not in ml_analysis['preprocessing']['correct']):
        issues.append("Generated code lowercases text which may merge distinct categories")

    if ("text_replacement" in ml_analysis['preprocessing']['generated'] and 
        "text_replacement" not in ml_analysis['preprocessing']['correct']):
        issues.append("Generated code replaces characters in text which may alter meaning")

    # Check for spatial aggregation
    if ("spatial_aggregation" in ml_analysis['preprocessing']['generated'] and 
        "spatial_aggregation" not in ml_analysis['preprocessing']['correct']):
        issues.append("Generated code performs spatial aggregation, losing geographic information")

    # Check for random state
    correct_params = ml_analysis['model_params']['correct']
    generated_params = ml_analysis['model_params']['generated']
    if (correct_params.get("random_state") is not None and 
        generated_params.get("random_state") is None):
        issues.append("Generated code doesn't set random_state, making results non-reproducible")
    # TODO: add more issues


    return issues 
if __name__ == "__main__":
    correct_code = open("example_code/corrected_code/correct_code.py", "r").read()
    generated_code = open("example_code/generated_code/generated_code.py", "r").read()
    pprint.pprint(analyze_ml_components(correct_code, generated_code))
    pprint.pprint(detect_ml_issues(analyze_ml_components(correct_code, generated_code)))