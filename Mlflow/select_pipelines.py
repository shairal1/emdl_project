import os
import shutil

# List of folders you want to take from pipelines
target_folders = [
    "data_leakage"
]

repo_root = os.getcwd()  # assume script runs at repo root

pipelines_dir = os.path.join(repo_root, "openhands-automation", "pipelines")
mlflow_dir = os.path.join(repo_root, "Mlflow")
new_folder_name = "selected_pipelines"
destination_dir = os.path.join(mlflow_dir, new_folder_name)

# Create destination folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

for folder_name in target_folders:
    source_path = os.path.join(pipelines_dir, folder_name)
    if os.path.isdir(source_path):
        dest_path = os.path.join(destination_dir, folder_name)
        # If destination folder exists, remove it to copy fresh
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(source_path, dest_path)
        print(f"Copied {folder_name} to {dest_path}")
    else:
        print(f"Warning: {folder_name} does not exist in {pipelines_dir}")

print("Done copying selected folders.")
