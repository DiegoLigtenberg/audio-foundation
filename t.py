import json
import matplotlib.pyplot as plt
import numpy as np

# Path to your JSON file
file_path = r"C:\Github\audio-foundation\rob_plot5_target.json"

# Open and load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Check the type of data and count elements
if isinstance(data, dict):
    element_count = len(data)
    print(f"The JSON contains {element_count} top-level keys.")
elif isinstance(data, list):
    element_count = len(data)
    print(f"The JSON contains {element_count} items in the list.")
else:
    print("Unknown JSON structure.")


with open('rob_plot3_target.json', 'r') as f:
    loaded_rob_plot1 = json.load(f)
rob_plot1 = loaded_rob_plot1


with open('rob_plot5_target.json', 'r') as f:
    loaded_rob_plot = json.load(f)
rob_plot = loaded_rob_plot

# rob_plot = rob_plot1 + rob_plot
# rob_plot = rob_plot[100:]

# Define a moving average function that handles edges properly
def moving_average_with_edges(data, window_size):
    # Calculate padding
    padding = window_size // 2
    
    # Apply edge padding
    padded_data = np.pad(data, (padding, padding), mode='edge') 
    
    # Compute the moving average using 'same' mode to ensure alignment
    smoothed = np.convolve(padded_data, np.ones(window_size) / window_size, mode='same')
    
    # Apply linear interpolation for the first few points to avoid abrupt jumps
    smoothed[:padding] = np.linspace(data[0], smoothed[padding], padding)
    
    return smoothed[:len(data)]

# Constants
batches_per_epoch = 2647  # Number of batches in one epoch
window_size = 1000         # Moving average window size

# Calculate the moving average
smoothed_loss = moving_average_with_edges(rob_plot, window_size)

# Convert batch indices to epoch numbers
epochs = np.arange(len(rob_plot)) / batches_per_epoch

# Plot the original data and the moving average
plt.figure(figsize=(10, 6))
plt.plot(epochs, rob_plot, 'o', label="Original Loss", color='grey', alpha=0.7)  # Real points as grey dots
plt.plot(epochs, smoothed_loss, label=f"Moving Average (window={window_size} batches)", color='blue', linewidth=2)  # Smoothed line in blue
plt.title("Loss per Epoch (Smoothed)")
plt.xlabel("Epoch (2647 batches per epoch)")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()