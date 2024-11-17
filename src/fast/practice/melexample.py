"""import torch
import matplotlib.pyplot as plt


# Create perceptual weights based on your provided logic
def create_perceptual_weights_pytorch(nr_steps, min_frequency, max_frequency):
    # Generate frequencies
    f = torch.linspace(min_frequency, max_frequency, nr_steps + 1)

    # Compute Mel scale: log(f / 700 + 1)
    mel = torch.log(f / 700 + 1)

    # Compute weights as the difference of mel scale
    weights = torch.diff(mel)

    # Normalize weights between 0 and 1
    weights /= weights.max()

    # Return frequencies (excluding the last one since diff reduces the length)
    return f, mel, weights


# Parameters
nr_steps = 2048
min_frequency = 20
max_frequency = 20000

# Get the perceptual weights
frequencies, mel, weights = create_perceptual_weights_pytorch(
    nr_steps, min_frequency, max_frequency
)

# Plot Frequency vs Mel
plt.subplot(1, 2, 1)
plt.plot(frequencies.numpy(), mel.numpy(), label="Mel scale")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mel scale (Pitch)")
plt.title("Mel Scale vs Frequency")
# plt.xscale("log")
plt.grid(True)

# Plot Frequency vs Weight
plt.subplot(1, 2, 2)
plt.plot(
    frequencies.numpy()[:-1], weights.numpy()
)  # Exclude last frequency since diff reduces length
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.title("Frequency Weighting")
plt.grid(True)
# plt.xscale("log")

plt.tight_layout()
plt.show()
"""
MIN_WEIGHT = 0.05 # this variable is important if you want to rescale how much higher frequencies are weighted, if not use the above code in docstring
# increase min weight = low frequencies matter less in loss
# decrease min weight = lower frequenies matter more in loss
import torch
from matplotlib import pyplot as plt

# Create perceptual weights based on your provided logic
def create_perceptual_weights_pytorch(nr_steps, min_frequency, max_frequency, min_weight=0.2):
    # Generate frequencies
    f = torch.linspace(min_frequency, max_frequency, nr_steps + 1)

    # Compute Mel scale: log(f / 700 + 1)
    mel = torch.log(f / 700 + 1)

    # Compute weights as the difference of mel scale
    weights = torch.diff(mel)

    # Normalize weights between 0 and 1
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Apply minimum weight adjustment
    weights = weights * (1.0 - min_weight) + min_weight

    # Return frequencies (excluding the last one since diff reduces the length)
    return f, mel, weights


# Parameters
nr_steps = 2048
min_frequency = 20
max_frequency = 20000
min_weight = MIN_WEIGHT

# Get the perceptual weights
frequencies, mel, weights = create_perceptual_weights_pytorch(
    nr_steps, min_frequency, max_frequency, min_weight
)

# Plot Frequency vs Mel
plt.subplot(1, 2, 1)
plt.plot(frequencies.numpy(), mel.numpy(), label="Mel scale")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mel scale (Pitch)")
plt.title("Mel Scale vs Frequency")
plt.xscale("log")
plt.grid(True)

# Plot Frequency vs Weight
plt.subplot(1, 2, 2)
plt.plot(frequencies.numpy()[:-1], weights.numpy())  # Exclude last frequency since diff reduces length
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.title("Frequency Weighting")
plt.grid(True)
plt.xscale("log")

plt.tight_layout()
plt.show()
