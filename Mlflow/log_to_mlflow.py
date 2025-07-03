import os
import subprocess

selected_pipelines_dir = os.path.join("Mlflow", "selected_pipelines")
target_files = ["example-0.py", "example-0_fixed_gemini.py"]

def main():
    if not os.path.isdir(selected_pipelines_dir):
        print(f"Directory not found: {selected_pipelines_dir}")
        return

    for folder_name in os.listdir(selected_pipelines_dir):
        folder_path = os.path.join(selected_pipelines_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in target_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    print(f"Executing {file_path} ...")
                    # Run the python script
                    subprocess.run(["python", file_path], check=True)
                else:
                    print(f"File {file_name} not found in {folder_name}")

if __name__ == "__main__":
    main()
