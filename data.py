from huggingface_hub import hf_hub_download
import os

# Define the repository ID and file path
repo_id = "happy-face/celebrity-face-matching-data"

filename = "TMDB-Face-with-embeddings.csv"

# Define the local directory to save the file
local_dir = "./" # You can change this to any directory

# Ensure the local directory exists
os.makedirs(local_dir, exist_ok=True)

# Download the file
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    local_dir=local_dir
)

print(f"File downloaded to: {os.path.join(local_dir, filename)}")
