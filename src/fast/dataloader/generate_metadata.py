from pathlib import Path
import json
from fast.settings.directory_settings import *


# Function to generate metadata
def generate_metadata(dataset_dir, output_file, group_size=10000):
    files = sorted(dataset_dir.glob("*.mp3"))  # Sort files to ensure order
    metadata = []

    # Open the metadata file in write mode
    with open(output_file, "w") as f:
        for idx, file in enumerate(files, start=1):
            group_number = (idx - 1) // group_size + 1
            metadata_entry = {"file": file.name, "group": group_number}
            f.write(json.dumps(metadata_entry) + "\n")  # Write each entry as a JSON object

if __name__ == "__main__":
    # Generate the metadata file
    output_file = DATASET_MP3_METADATA / "metadata.json"
    generate_metadata(DATASET_MP3_DIR, output_file)
