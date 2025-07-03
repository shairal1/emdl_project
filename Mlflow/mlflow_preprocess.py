import os
from google import genai
API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc" 
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

selected_pipelines_dir = os.path.join("Mlflow", "selected_pipelines")
target_files = ["example-0.py", "example-0_fixed_gemini.py"]

def build_prompt(code, folder_name, file_name):
    return f"""
You are an expert ML engineer.

Transform this Python pipeline code so that it uses MLflow for experiment tracking:

- Add `mlflow.autolog()` at the start.
-Start the tracking mlflow.set_tracking_uri("http://127.0.0.1:5000")
  mlflow.set_experiment("Pipeline_track")
- Wrap the training and evaluation into an `mlflow.start_run(run_name=RUN_NAME)` context, where RUN_NAME is meaningful (use '{folder_name}_{file_name}' as run_name).
- Ensure that the metrics: F1 score, accuracy, precision, and recall are logged explicitly with `mlflow.log_metric()`.
- Log the file name as mlflow.log_param("Pipeline", RUN_NAME)
- The code should run as is, with the logging additions and minimal restructuring and dont create a main function
- Use `import mlflow` if not already imported.
- Your code should be able to be executed right away
- Resolve syntax issues or scikit-learn version compatibility issues.
- Clean and restructure code only when necessary to ensure correct execution.
- Return a complete and clean version of the modified Python code — no explanations
- Return only the transformed Python code. Do not include markdown formatting like triple backticks or python. Just the raw code

Here is the original code:

```python
{code}
Return the transformed full Python code only, no explanations.
"""

def process_file(file_path, folder_name, file_name):
    with open(file_path, "r") as f:
        code = f.read()
    prompt = build_prompt(code, folder_name, file_name)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    transformed_code = response.text

    with open(file_path, "w") as f:
        f.write(transformed_code)
    print(f"Processed and updated: {file_path}")


def main():
    for folder_name in os.listdir(selected_pipelines_dir):
        folder_path = os.path.join(selected_pipelines_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in target_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    process_file(file_path, folder_name, file_name)
                else:
                    print(f"File {file_name} not found in {folder_name}")

if __name__ == "__main__":
    main()
