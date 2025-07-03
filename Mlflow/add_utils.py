import os
import shutil

repo_root = os.getcwd()  # or set explicitly
mlflow_dir = os.path.join(repo_root, "Mlflow")
selected_pipelines_dir = os.path.join(mlflow_dir, "selected_pipelines")
utils_source_dir = os.path.join(mlflow_dir, "utils")  # it's a folder

# Check utils folder exists
if not os.path.isdir(utils_source_dir):
    raise FileNotFoundError(f"utils folder not found at expected location: {utils_source_dir}")

# Walk through all subdirectories inside selected_pipelines
for root, dirs, files in os.walk(selected_pipelines_dir):
    utils_dest_dir = os.path.join(root, "utils")

    # If utils folder already exists in dest, remove it to copy fresh
    if os.path.exists(utils_dest_dir):
        shutil.rmtree(utils_dest_dir)

    # Copy the whole utils folder recursively
    shutil.copytree(utils_source_dir, utils_dest_dir)
    print(f"Copied utils folder to {utils_dest_dir}")

print("Done copying utils folder to all subfolders.")
