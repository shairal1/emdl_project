import os

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # __file__ points to utils.py in Mlflow, so go two levels up to reach project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
