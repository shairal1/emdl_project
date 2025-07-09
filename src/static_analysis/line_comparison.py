"""
The goal of this script is to compare the number of line changes between two files and it's correlation
with the problem identified in the file.
"""
import traceback
from pathlib import Path
import os
import sys
import difflib
from time import sleep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from google import genai
import matplotlib.pyplot as plt
from src.static_analysis.ml_analysis import analyze_ml_components, detect_ml_issues

fixed_pipeline = "openhands-automation/pipelines/aggregation_errors/fixed.py"
original_pipeline = "openhands-automation/pipelines/aggregation_errors/example-0.py"


API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc" 
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"
with open(fixed_pipeline, 'r') as f:
    fixed_code = f.read()
with open(original_pipeline, 'r') as f:
    original_code = f.read()
# Compare the two files
def list_pipelines(directory: str = "example_pipelines") -> list[str]:
    """Lists all Python files in the specified directory."""
    path = Path(directory)
    pipeline_folders = [f for f in path.iterdir() if f.is_dir()]
    return pipeline_folders
def dynamic_correct_folder(filepath = "openhands-automation/pipelines/"):
    """
    Corrects all Python files in the specified folder.
    """
    pipelines = list_pipelines(filepath)
    folder_names = [os.path.basename(folder) for folder in pipelines]
    counts = []
    for i, pipeline in enumerate(pipelines):
        print(f"Processing pipeline: {pipeline}", i)
        fixed_file = pipeline / "fixed.py"
        original_file = pipeline / "example-0.py"
        with open(fixed_file, "r", encoding="utf-8") as f:
            fixed_code = f.read()
        with open(original_file, "r", encoding="utf-8") as f:
            original_code = f.read()
        count = count_changed_lines(original_code, fixed_code)
        # Try finding problems in the pipeline
        try:
            differences = compare_files_with_LLM(original_code, fixed_code)
            print("Differences found: \n", differences)
            print("Writing differences to output.txt")
            with open("output.txt", "a+", encoding="utf-8") as f:
                f.write(f"Pipeline: {pipeline}\n")
                f.write(f"Number of changed lines: {count}\n")
                f.write("Differences:\n")
                f.write(differences + "\n")
            counts.append(count)
        except Exception as e:
            if "max TPM reached" in str(e):
                print("An error occurred:", e)
                print("max TPM reached, waiting 60 seconds")
                sleep(60)
                differences = compare_files_with_LLM(original_code, fixed_code)
                print("writing differences to output.txt")
                with open("output.txt", "a+", encoding="utf-8") as f:
                    f.write(f"Pipeline: {pipeline}\n")
                    f.write(f"Number of changed lines: {count}\n")
                    f.write("Differences:\n")
                    f.write(differences + "\n")
                counts.append(count)
            else:
                print("An error occurred:", e)
                traceback.print_tb(e.__traceback__)
                return
    with open("output.txt", "a+", encoding="utf-8") as f:
        f.write("Summary of all pipelines:\n")
        for folder, count in zip(folder_names, counts):
            f.write(f"Pipeline: {folder}, Number of changed lines: {count}\n")
    return counts, folder_names


def count_changed_lines(original, fixed):
    diff = difflib.unified_diff(original.splitlines(), fixed.splitlines())
    changed_lines = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
    return len(changed_lines)

def build_prompt(original_code, fixed_code):
    prompt = (
        "You are given two Python files. "
        "The first is the original code containing problems. "
        "The second is the generated code with those problems fixed. "
        "For each problem you identify that was fixed, return a string in the following format:\n"
        "<name of problem>: <number of lines changed to fix this problem>\n"
        "If there are multiple problems, list them all in the same format.\n"
        "Do not include any additional text, explanations, or comments.\n"
        "Only return a string for each problem, nothing else.\n\n"
        "Original code:\n"
        f"{original_code}\n\n"
        "Generated code:\n"
        f"{fixed_code}\n"
    )
    return prompt

def compare_files_with_LLM(original_code, fixed_code):
    prompt = build_prompt(original_code, fixed_code)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text

# dynamic_correct_folder()
print(list_pipelines("openhands-automation/pipelines/"))
counts, folder_names = dynamic_correct_folder("openhands-automation/pipelines/")
plt.figure(figsize=(8, 5))
bars = plt.bar(folder_names, counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
# Add labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height}', 
             ha='center', va='bottom', fontsize=10)

# Add axis labels and title
plt.xlabel('problem')
plt.ylabel('number of lines')
plt.title("Number of lines changed in each pipeline")
plt.ylim(0, max(counts) + 10)  # leave room for labels

plt.tight_layout()
plt.show()