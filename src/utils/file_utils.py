"""
File Utilities Module
Common file operations used across the project
"""
from typing import Optional


def load_code_from_file(file_path: str) -> Optional[str]:
    """
    Load code from a file.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        File contents as string, or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None 