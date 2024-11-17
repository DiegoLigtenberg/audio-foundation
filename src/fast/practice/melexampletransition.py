import torch
from matplotlib import pyplot as plt

# Define tanh function
def tanh(x: torch.Tensor) -> torch.Tensor:
    output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return (output - output.min()) / (output.max() - output.min())

# Function to create perceptual weights
def create_perceptual_weights_pytorch(nr_steps, min_frequency, max_frequency, bass_cutoff=150, 
                                      min_bass_weight=0.6, max_bass_weight=1.0, min_weight=0.2, 
                                      x_max=200):
    # Generate frequency values
    f = torch.linspace(min_frequency, max_frequency, nr_steps)

    # Calculate switch indices based on frequencies
    switch_idx_0 = int(frequency_0 / (max_frequency - min_frequency) * nr_steps) - 1
    switch_idx_1 = int(frequency_1 / (max_frequency - min_frequency) * nr_steps) - 1

    # Create weight for the first part (bass region)
    weights_0 = tanh(torch.linspace(0.0, 3.0, switch_idx_0))
    weights_0 = weights_0 * (1.0 - min_bass_weight) + min_bass_weight

    # Constant weight between bass_cutoff and frequency_1
    weights_1 = torch.ones(switch_idx_1 - switch_idx_0)

    # Create weight for the third part (high frequency region)
    weights_2 = tanh(torch.linspace(3.0, shift_tanh, nr_steps - switch_idx_1))
    weights_2 = weights_2 * (1.0 - high_min_weight) + high_min_weight

    # Concatenate all weight parts
    weights = torch.concat([weights_0, weights_1, weights_2])

    # Compute Mel scale for the frequencies
    mel = torch.log(f / 700 + 1)  # Logarithmic Mel scale formula

    return f, mel, weights

# Parameters
nr_steps = 2047
min_frequency = 20
max_frequency = 20000
high_min_weight = 0.3
low_min_weight = 0.1
frequency_0 = 100
frequency_1 = 2000
shift_tanh = -18.5

# Get the perceptual weights and Mel scale
frequencies, mel, weights = create_perceptual_weights_pytorch(
    nr_steps, min_frequency, max_frequency, bass_cutoff=150, min_bass_weight=0, 
    max_bass_weight=1.0, min_weight=0.2, x_max=200
)

# Plot Frequency vs Mel
plt.subplot(1, 2, 1)
plt.plot(frequencies[:-1].numpy(), mel.numpy()[:-1], label="Mel scale")  # Ensure both have the same length by removing last element of frequencies
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mel scale (Pitch)")
plt.title("Mel Scale vs Frequency")
# plt.xscale("log")
plt.grid(True)

# Plot Frequency vs Weight
plt.subplot(1, 2, 2)
plt.plot(frequencies[:15000].numpy(), weights[:15000].numpy())  # Frequencies and weights should now have the same length
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.title("Frequency Weighting with Logistic Transition")
# plt.xscale("log")
plt.grid(True)

plt.tight_layout()
plt.show()
