import traceback
import re
from time import sleep
import mlflow
from google import genai
import json
import io
from contextlib import redirect_stdout
import pandas as pd
import os 
import sys
import runpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.LLM_detection.find_errors import ask_gemini_to_find_problems, list_pipelines
# set model temperature to 0 or 0.2
# remove comments
            # stripped = "\n".join(
            #     line for line in content.splitlines() if not line.lstrip().startswith("#")
            # )
#change the API key to your own
API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc" 
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"
def remove_comments(code: str) -> str:
    """Removes comments from the code to avoid giving hints to Gemini."""
    return "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("#")
    )
def try_run_pipeline(code_str: str):
    buf = io.StringIO()
    try:
        # Add the workspace root to sys.path for package imports
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
        # Provide a full global namespace so imports and modules work
        exec_globals = {
            "__file__": os.path.abspath("LLM_automation/test_pipeline/pipeline.py"),
            "__name__": "__main__",
            "__package__": None,
            "__builtins__": __builtins__,
            "os": os,
            "sys": sys,
            "pd": pd,
        }
        # Also add LLM_automation as a module if needed
        import types
        import importlib.util
        # Try to import LLM_automation as a module if it exists
        try:
            spec = importlib.util.find_spec("LLM_automation")
            if spec is not None:
                LLM_automation = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(LLM_automation)
                exec_globals["LLM_automation"] = LLM_automation
        except Exception:
            pass
        with redirect_stdout(buf):
            exec(code_str, exec_globals)
        return None, None, buf.getvalue()
    except Exception as e:
        return str(e), traceback.format_exc(), buf.getvalue()


def try_run_pipeline_file(filepath: str):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            runpy.run_path(filepath, run_name="__main__")
        return None, None, buf.getvalue()
    except Exception as e:
        return str(e), traceback.format_exc(), buf.getvalue()


def ask_gemini_to_fix(code: str, error: str, tb: str, problems: str) -> str:
    """Prompts Gemini to fix the code based on the error and problems."""
    prompt = f"""
This Python ML pipeline has the following problems and throws this error.
Problems: 
{problems}
❌ Error:
{error}

🔍 Traceback:
{tb}

💻 Full code:
```python
{code}
✅ Please provide a corrected, complete version of the code within a single python block and specify with comments what you modified.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text

def ask_gemini_to_improve(code, problems: list[str]) -> str:
    """Sends code and problems to Gemini, returns the raw response."""
    prompt = f"""
This Python ML pipeline has the following problems:
{problems}
Here is the code:
```python
{code}
Don't do hyperparameter tuning, just improve the code.
Please provide a corrected, complete version of the code within a single python block and specify with comments what you modified. don't add any unnecessary comments, just the code.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text
def extract_code(reply: str) -> str | None:
    """Extracts Python code contained in a Markdown ```python block``` if present."""
    match = re.search(r"```python\n(.+?)```", reply, re.S)
    return match.group(1) if match else None 


def _sanitize_label(label: str) -> str:
    # map relational ops → text
    s = label.replace(">=", "ge_") \
             .replace(">",  "gt_") \
             .replace("<=", "le_") \
             .replace("<",  "lt_")
    # spaces → underscores
    s = re.sub(r"\s+", "_", s)
    # drop any remaining disallowed chars (keep alnum, _-./ space)
    s = re.sub(r"[^A-Za-z0-9_\-./ ]", "_", s)
    # collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s
def log_classification_report_artifact(
    report_str: str,
    text_name: str = "classification_report.txt",
    csv_name: str  = None,
    html_name: str = None,
):
    """
    Logs the raw printed report_str as plain text, and optionally as CSV/HTML.
    """
    # 1) raw text
    mlflow.log_text(report_str, text_name)

    # parse once if CSV or HTML requested
    if csv_name or html_name:
        import pandas as pd
        from io import StringIO
        df = pd.read_fwf(StringIO(report_str), index_col=0)

    if csv_name:
        df.to_csv(csv_name)
        mlflow.log_artifact(csv_name)

    if html_name:
        html = df.to_html()
        mlflow.log_text(html, html_name)
def log_classification_report_from_string(
    report_str: str,
    artifact_path: str = "classification_report.json",
    log_individual_metrics: bool = True,
) -> None:
    """
    Parse a sklearn-style text report from `report_str`,
    log the nested dict as a JSON artifact, and optionally
    log each cell as its own MLflow metric with sanitized keys.
    """
    # find header and slice downwards
    lines = report_str.strip().splitlines()
    header_i = next(
        (i for i, L in enumerate(lines)
          if re.search(r"\bprecision\b.*\brecall\b.*\bf1-score\b", L)),
        None
    )
    if header_i is None:
        raise ValueError("Couldn't find classification-report header")

    table = "\n".join(lines[header_i:])
    df = pd.read_fwf(io.StringIO(table), index_col=0)
    df = df.loc[~df.index.isin(df.columns)]            # drop stray header-row
    df = df.apply(pd.to_numeric, errors="coerce")      # to floats

    # log full JSON
    metrics_dict = df.to_dict(orient="index")
    mlflow.log_dict(metrics_dict, artifact_path)

    if log_individual_metrics:
        for raw_label, row in metrics_dict.items():
            safe_label = _sanitize_label(raw_label)
            for metric_name, val in row.items():
                if val is not None and not pd.isna(val):
                    key = f"{safe_label}_{metric_name}"
                    mlflow.log_metric(key, float(val))
def dynamic_correct_folder(filepath = "openhands-automation/pipelines/"):
    """
    Corrects all Python files in the specified folder.
    """
    pipelines = list_pipelines(filepath)
    for i, pipeline in enumerate(pipelines):
        print(f"Processing pipeline: {pipeline}", i)
        with open(pipeline, "r", encoding="utf-8") as f:
            code = f.read() 
        # Try finding problems in the pipeline
        try:
            main(pipeline.replace("example-0.py", ""), pipeline_name="example-0")
        except Exception as e:
            print("An error occurred:", e)
            print("max TPM reached, waiting 60 seconds")
            sleep(60)
            main(filepath.replace("example-0.py", ""), pipeline_name="example-0")
        



def main(filepath: str = "LLM_automation/test_pipeline/", pipeline_name: str = "pipeline"):
    """Main function to run the pipeline and fix it if it fails."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LLM_automation")
    print(filepath)
    with mlflow.start_run(run_name=f"autofix__{filepath[-10:]}"):
        orig_file = filepath + pipeline_name + ".py"
        fixed_file = filepath + pipeline_name + "_Gemini.py"
#--------------Logging files into mlflow--------------------------------
        mlflow.log_artifact(orig_file, "original_pipeline.py")
        print("Original pipeline file logged in mlflow")
        print("Running original pipeline file...")
#--------------Running the original pipeline file------------------------
        error, tb, printed = try_run_pipeline_file(orig_file)
        i = 1
        with open(orig_file, "r", encoding="utf-8") as f:
            code = f.read()
#--------------Fixing the pipeline errors with gemini -------------------
        while error is not None:
            print(f"❌ Pipeline failed with error {i}: {error}")
            print("Traceback:")
            print(tb)   
            print("Requesting fix from Gemini..")

            code = ask_gemini_to_fix(code, error, tb, printed)
            with open(fixed_file, "w", encoding="utf-8") as f:
                f.write(code.replace("```python", "").replace("```", "").strip())
            print("wrote new code to fixed file")
            print("Running fixed pipeline file...")
            error, tb, printed = try_run_pipeline_file(fixed_file)
            i += 1
            if i > 5:
                print("⚠️ Too many iterations without success. Stopping.")
                mlflow.log_text(tb, "traceback.txt")
                mlflow.log_text(error, "final_message.txt")
                break
#---------------Logging the original classification report----------------
        if error is None:
            print("✅ Pipeline fixed successfully after", i, "iterations.")
            print("classification report of non improved code:")
            print(printed)
            mlflow.log_text(printed, "orig_pipeline_output.txt")
            print("Classification report logged in mlflow")
            print("Requesting fixes from Gemini to improve the code further...")
            problems = ask_gemini_to_find_problems(code)
            new_code = ask_gemini_to_improve(code, problems)
#---------------Logging the improved code and classification report----------------
            print("writing improved code to fixed file...")
            with open(fixed_file, "w", encoding="utf-8") as f:
                f.write(new_code.replace("```python", "").replace("```", "").strip())
            print("running improved code...")
            error, tb, printed = try_run_pipeline_file(fixed_file)
            print("classification report of improved code:")
            print(printed)
            mlflow.log_text(printed, "improved_pipeline_output.txt")
            print("Classification report logged in mlflow")
            mlflow.log_text(new_code, "pipeline_fixed.py")
            mlflow.log_artifact(fixed_file, "fixed_pipeline.py")
            print("All done! Check the MLflow UI for details.")
        else:
            print("❌ Pipeline still failed after Gemini's improvements.")
            print("Classification report of fixed code:")
            print(printed)
            mlflow.log_text(printed, "fixed_pipeline_output.txt")
            mlflow.log_text(code, "pipeline_fixed.py")
            mlflow.log_artifact(fixed_file, "fixed_pipeline.py")
            print("All done! Check the MLflow UI for details.")


# def main(filepath: str = "LLM_automation/test_pipeline/pipeline.py"):
#     """Main function to run the pipeline and fix it if it fails."""
#     mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     mlflow.set_experiment("autofix_pipeline")
#     with mlflow.start_run(run_name="autofix_gemini_test"):
#         orig_file = filepath
#         fixed_file = "LLM_automation/test_pipeline/pipeline_fixed.py"

#         # Use runpy to run the pipeline file
#         try:
#             error, tb, printed = try_run_pipeline_file(orig_file)
#         except Exception as e:
#             print("⚠️ Error while trying to run the pipeline:", e)
#             return
#         if error is None:
#             mlflow.log_param("Error in pipeline", "No error")
#             with open(orig_file, "r", encoding="utf-8") as f:
#                 code = f.read()
#             text, problems = ask_gemini_to_find_problems(code)
#             print("Problems found in the pipeline:")
#             for problem in problems:
#                 print(f" - {problem}", end="")
#             print("")
#             new_code = ask_gemini_to_improve(code, problems)
#             with open(fixed_file, "w", encoding="utf-8") as f:
#                 f.write(new_code.replace("```python", "").replace("```", "").strip())
#             print("wrote new code to fixed file")
#             mlflow.log_text(new_code, "pipeline_fixed.py")
#             # print(new_code)
#             error, tb, printed = try_run_pipeline_file(fixed_file)
#             if error is not None:
#                 print("error while trying to run the fixed pipeline:", error, tb)
#                 print("❌ Pipeline still failed after Gemini's improvements.")
#             print(printed)
#             mlflow.log_text(printed, "pipeline_output.txt")
#             print("Classification report logged in mlflow")

#         else:
#             mlflow.log_param("Error in pipeline", error)
#             print("❌ Pipeline failed with error:", error)
#             print("Traceback:")
#             print(tb)
        
#         with open(orig_file, "r", encoding="utf-8") as f:
#             code = f.read()
#         print("❌ Pipeline failed. Requesting fix from Gemini…")
#         current_code = code
#         current_error = error
#         current_tb = tb
#         i = 0
#         # mlflow.log_param("pipeline_file", orig_file)
#         text, problems = ask_gemini_to_find_problems(current_code)
#         # mlflow.log_text(text, "pipeline_problems.txt")
#         # mlflow.log_param("problems", json.dumps(problems))
#         # mlflow.log_text(current_code, "pipeline_original.py")
#         # mlflow.log_metric("fixe_needed", 1)
#         while current_error is not None:
#             fix_reply = ask_gemini_to_fix(current_code, current_error, current_tb, problems)
#             fixed_code = extract_code(fix_reply)
#             if fixed_code is None:
#                 print("⚠️ Couldn't parse the fixed code. Here’s the full Gemini reply:\n")
#                 print(fix_reply)
#                 mlflow.log_text(fix_reply, "gemini_full_response.txt")
#                 return
#             # Write the fixed code to file
#             with open(fixed_file, "w", encoding="utf-8") as f:
#                 f.write(fixed_code)
#             # Run the fixed pipeline file
#             current_error, current_tb, printed = try_run_pipeline_file(fixed_file)
#             i += 1
#             if current_error is None:
#                 print(f"✅ Pipeline fixed successfully after {i} iterations.")
#                 mlflow.log_metric("fix_iterations", i)
#                 mlflow.log_text(fixed_code, "pipeline_fixed.py")
#                 mlflow.log_artifact(fixed_file)
#                 log_classification_report_artifact(printed, "classification_report.txt")
#                 print("Classification report logged as JSON artifact.")
#                 print(f"✅ Fixed code written to {fixed_file}.")
#                 mlflow.log_param("fix_extracted", True)
#                 mlflow.log_text(fix_reply, "gemini_full_response.txt")
#                 i = 0
#                 break
#             else:
#                 print(f"❌ Fix attempt {i} failed with error: {current_error}")
#             if i >= 5:
#                 print("⚠️ Too many iterations without success. Stopping.")
#                 mlflow.log_text(current_tb, "traceback.txt")
#                 mlflow.log_text(current_error, "final_message.txt")
#                 break
#         new_code = ask_gemini_to_improve(current_code, problems)
#         with open(fixed_file, "w", encoding="utf-8") as f:
#             f.write(new_code.replace("```python", "").replace("```", "").strip())
#     print("All done! Check the MLflow UI for details.")



if __name__ == "__main__":
    # main()
    dynamic_correct_folder()