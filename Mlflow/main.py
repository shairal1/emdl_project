import subprocess
import sys
import os

# Define the path to the Mlflow directory
mlflow_dir = os.path.join(os.getcwd(), "Mlflow")

# List of script filenames in order
scripts = [
    "select_pipelines.py",
    "add_utils.py",
    "gemini_fix.py",
    "mlflow_preprocess.py",
    "log_to_mlflow.py"
]

def run_script(script_name):
    script_path = os.path.join(mlflow_dir, script_name)
    print(f"\n--- Running {script_path} ---")
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f" Error while running {script_name}. Exiting.")
        sys.exit(e.returncode)

def main():
    for script in scripts:
        run_script(script)
    print("\n All Mlflow scripts ran successfully.")

if __name__ == "__main__":
    main()
