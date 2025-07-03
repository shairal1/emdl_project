import os
from google import genai

API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc"
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.0-flash"

selected_pipelines_dir = os.path.join(os.getcwd(), "Mlflow", "selected_pipelines")
target_file_name = "example-0.py"

def build_prompt(code):
    return f'''
Please fix the following Python code and return only the fixed runnable code:
- In order to fetch the CSV file, keep the same format as in the given code.
-Return only the transformed Python code. Do not include markdown formatting like triple backticks or python. Just the raw code
(No markdown, no explanations, just the pure code):

{code}
'''

def generate_gemini_content(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    # response may be an object with `.text` attribute
    if hasattr(response, "text"):
        return response.text.strip()
    # else fallback to string or first tuple element if tuple
    if isinstance(response, tuple):
        return response[0].strip()
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()

for root, dirs, files in os.walk(selected_pipelines_dir):
    if target_file_name in files:
        file_path = os.path.join(root, target_file_name)
        print(f"Processing {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            original_code = f.read()

        prompt = build_prompt(original_code)
        fixed_code = generate_gemini_content(prompt)

        output_path = os.path.join(root, "example-0_fixed_gemini.py")
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(fixed_code)

        print(f"Saved fixed code to {output_path}")

print("Done processing all example-0.py files.")
