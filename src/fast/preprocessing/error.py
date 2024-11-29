import pickle


import json
from pathlib import Path
import torch
import json
from pathlib import Path

import json
from pathlib import Path

# Define the min and max values with an additional key for easy access
log_power_spectrogram_minmax = {
    "min_val": float(torch.tensor(-159.9687)),  # Convert tensor to float
    "max_val": float(torch.tensor(135.4550)),  # Convert tensor to float
    "min_max": (float(torch.tensor(-159.9687)), float(torch.tensor(135.4550))),  # Easy access to both
    "metadata": {
        "description": "Min and max values in the dict represent the log power spectrogram, calculated as 20 * log10(power_spectrogram + 1e-8), derived from the magnitude spectrogram of 170k songs (min = 0, max = 2434, rounded to max = 2750)"  }
}

# Define the file path for saving
save_path = Path("src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.json")

# Create directories if they don't exist
save_path.parent.mkdir(parents=True, exist_ok=True)

# Save the dictionary as a JSON file
with open(save_path, "w") as f:
    json.dump(log_power_spectrogram_minmax, f, indent=4)

print(f"Min-max values saved to {save_path}")

# Define the file path for saving
save_path = Path("src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.json")

# Create directories if they don't exist
save_path.parent.mkdir(parents=True, exist_ok=True)

# Save the dictionary as a JSON file
with open(save_path, "w") as f:
    json.dump(log_power_spectrogram_minmax, f, indent=4)

print(f"Min-max values saved to {save_path}")
asd

# Define the file path for saving
save_path = Path("src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.json")

# Create directories if they don't exist
save_path.parent.mkdir(parents=True, exist_ok=True)

# Save the dictionary as a JSON file
with open(save_path, "w") as f:
    json.dump(log_power_spectrogram_minmax, f, indent=4)

print(f"Min-max values saved to {save_path}")



# Load the pickled invalid_list
with open(r'src\fast\preprocessing\dataloader\lists_and_saved_files\min_max_values_log_power_spectrogram.pkl', 'rb') as f:
    invalid_list = pickle.load(f)

# print(invalid_list[:10])  # This will display the contents of the loaded list
# for q in ((invalid_list[:10])):
#     print(q)

print(invalid_list["min_val"])
print(invalid_list["max_val"])
asd
import torch
import pickle
from pathlib import Path

# Define the min and max values as tensors
log_power_spectrogram_minmax = {
    "description": "these are the min and max values calculated from the softest sound and loudest sound's magnitude spectrogram (min 0, max 2434.3056640625) of the whole dataset (170k songs). \
        For inference (new data), and easier numbers we set 2750 to be the absolute largest sound (~1.1 * louder than the loudest sound in this dataset).\
            The values in this dict are the min and max value after using  20 * torch.log10(power_spectrogram + 1e-8), where power_spectrogram = (magnitude spectrogram **2).",
    "min_val": torch.tensor(-159.9687),
    "max_val": torch.tensor(135.4550)
}

# Define the file path for saving
save_path = Path("src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.pkl")

# Create directories if they don't exist
save_path.parent.mkdir(parents=True, exist_ok=True)

# Save the dictionary as a pickle file
with open(save_path, "wb") as f:
    pickle.dump(log_power_spectrogram_minmax, f)

print(f"Min-max values saved to {save_path}")
