import traceback
from pathlib import Path
import os
import sys
import difflib
from time import sleep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from google import genai
import matplotlib.pyplot as plt
import json
API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc" 
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

def generate_graph():
    with open("output.txt", "r", encoding="utf-8") as f:
        text = f.read()
    prompt = f"""You are given a diagnostic text report that lists problems found in various machine learning pipelines. Each pipeline section contains "Differences:" followed by a list of issues, one per line.

Your task is to:
1. Classify all the listed problems into coherent problem classes (e.g., Data Leakage, Feature Engineering Issues, Code Structure Problems, Reproducibility, etc.).
2. For each class, list the exact problem descriptions (preserve their wording as in the file).
3. Provide the total number of lines (i.e., occurrences) per class.

Output format:
Each line should contain:
[Class Name]: [list of problem descriptions as they appear]: [number of lines changed] — [Total number of lines in this class]

Ensure:
 -remove any ':' in the problem descriptions
- The problem descriptions are **verbatim** from the original text.
- Each problem is listed under only one class.
- The total line count per class is accurate.

Use this input text (between triple backticks):
```{text}```"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    print("Response from LLM:")
    print(response.text)
    
    # Save the response to a file
    with open("problem_classes.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

def process_response(text: str) -> dict:
    """
    Process the response text to extract problem classes and their counts.
    
    Args:
        text: The response text from the LLM.
        
    Returns:
        A dictionary with problem classes as keys and their counts as values.
    """
    lines = text.strip().split('\n')
    problem_classes = {}
    count = 0
    for line in lines:
        if "**" in line:
            class_name = line.replace("**", "").strip().replace(":", "")
            problem_classes[class_name] = {}
        else:
            if line.strip():
                # try:
                print(line.split("â€”")[-1])
                total_count = int(line.split("—")[-1].strip())
                problem_classes[class_name]['total'] = total_count
                problem_description = line.split("—")[0].strip()
                problems = problem_description.split(',')
                for problem in problems:
                    problem = problem.strip()
                    problem_name = problem.split(":")[0].strip()
                    problem_count = int(problem.split(":")[-1].strip())
                    problem_classes[class_name][problem_name] = problem_count
                # except Exception as e:
                #     count += 1
    print(count)
    # dump the problem classes to a JSON file
    with open("problem_classes.json", "w", encoding="utf-8") as f:
        json.dump(problem_classes, f, indent=4)
    print("Problem classes extracted and saved to problem_classes.json")
    return problem_classes

def draw_graph():
    with open("problem_classes.json", 'r') as f:
        problem_classes = json.load(f)
    problem_names = [problem for problem in problem_classes.keys()]
    problem_counts = [problem_classes[problem]['total'] for problem in problem_classes.keys()]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(problem_names, problem_counts, color='#4A90E2', edgecolor='black', linewidth=0.7)

    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Style axes and grid
    plt.xlabel('Problem Category', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Lines', fontsize=12, fontweight='bold')
    plt.title("📊 Number of Lines Changed per Problem Category", fontsize=14, fontweight='bold')

    plt.xticks(rotation=35, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Tight layout and margins
    plt.ylim(0, max(problem_counts) + 30)
    plt.tight_layout()
    plt.show()

with open("problem_classes.txt", 'r', encoding='utf-8') as f:
    response = f.read()

process_response(response)