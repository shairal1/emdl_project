import traceback
from google import genai
from time import sleep
from pathlib import Path
import os
API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc"  # Replace with your own API key
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.0-flash"

def try_run_pipeline(code):
    """Tries running the pipeline and returns (None, None) if all is well, or (error, traceback) otherwise."""
    try:
        exec(code, {})
        return None, None
    except Exception as e:
        return str(e), traceback.format_exc()
def ask_gemini_to_find_problems(code: str) -> tuple[str, list[str]]:
    """Sends the code to Gemini to identify ML problems and returns the response text and problem titles."""
    prompt = f"""
    I will give you a Python code snippet. Your task is to identify all the machine learning (ML) problems in the code and describe them in detail.
    Do not provide any code fixes, just list the problems.
    If there are multiple problems, list them all.
    I don't want any empty lines or extra text, just the problems in the specified format.
    Return your output strictly using the following format and nothing else:

    [problem1_name]: description of problem 1  
    [problem2_name]: description of problem 2  
    ...

    Instructions:
    - Use only the format above.
    - Do not add any introductory text or headers.
    - Do not use bullet points, stars, numbering, or markdown.
    - Use exactly one space after each colon.
    - Separate each problem with a new line.

    Repeat this format exactly. Do not alter it.

    Here is the code:
    {code}
    """              
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    problems = response.text.strip().split('\n')

    try:
        titles = [problem[:problem.index(':')].strip() for problem in problems]
    except Exception as e:
        print("Error extracting titles from problems:", e)
        titles = []
        print("Problems:", problems)

    return response.text, titles


def list_pipelines(directory: str = "example_pipelines") -> list[str]:
    """Lists all Python files in the specified directory."""
    path = Path(directory)
    pipeline_folders = [f for f in path.iterdir() if f.is_dir()]
    pipelines = []
    for folder in pipeline_folders:
        #getting all python files in the folder that need to be fixed
        pipelines.extend([str(file) for file in folder.glob("*.py") if ("Gemini" not in file.name) and ("fixed" not in file.name)   ])

    return pipelines
# if this code is run from the command lines it will list all the pipelines in the example_pipelines directory and try to find problems in them using Gemini.
# It will create a file with the same name as the pipeline but with _problems.txt appended to it, containing the problems found by Gemini.
def main():
    # the commented code is used to test pipelines whwith the ml_piped repository we can chanfge it later ot do this
    # pipelines = list_pipelines()
    # for i, pipeline in enumerate(pipelines):
    #     print(f"Processing pipeline: {pipeline}", i)
    #     with open(pipeline, "r", encoding="utf-8") as f:
    #         code = f.read() 
    #     # Try finding problems in the pipeline
    #     try:
    #         text, problems = ask_gemini_to_find_problems(code)
    #     except Exception as e:
    #         print("max TPM reached, waiting 60 seconds")
    #         sleep(60)
    #         text, problems = ask_gemini_to_find_problems(code)
    #     with open(pipeline.replace(".py", "_problems.txt"), "w", encoding="utf-8") as f:
    #         f.write(text)
    #     print(f"Problems found in {pipeline}:")
    #     for problem in problems:
    #         print(f"- {problem}")
    # use for the example code we have.
    pipeline = "LLM_automation/test_pipeline/pipeline.py"
    print(f"looking for pipeline in {os.path.abspath(pipeline)}")
    print(f"Processing pipeline: {pipeline}")
    with open(pipeline, 'r') as f:
        code = f.read()
    try:
        text, problems = ask_gemini_to_find_problems(code)
    except Exception as e:
        if "max TPM reached" in str(e):
            print("Max TPM reached, waiting 60 seconds")
            sleep(60)
            text, problems = ask_gemini_to_find_problems(code)
        else:
            print("An error occurred:", e)
            return
    print(f"Problems found in {pipeline}:")
    print(text)

if __name__ == "__main__":
    main()
