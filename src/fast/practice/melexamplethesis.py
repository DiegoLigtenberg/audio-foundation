import numpy as np
import matplotlib.pyplot as plt

# Generate the linear frequency vector
lin_hz_vector = np.array([i * 20000 / 2048 for i in range(2048)])

# Apply the Mel scale transformation
mel_scale_vector = np.array([1 / np.log(2) * np.log(1 + (hz / 1000)) * 1000 for hz in lin_hz_vector])
mel_scale_vector = mel_scale_vector[::-1]  # Reverse the scale (higher frequencies are now at the start)

# Normalize the Mel scale vector
mel_scale_vector = mel_scale_vector / np.max(mel_scale_vector)

# Plot the Mel scale versus frequency
plt.figure(figsize=(12, 6))

# Plot Mel Scale vs Frequency
plt.subplot(1, 2, 1)
plt.plot(lin_hz_vector, mel_scale_vector)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Mel Scale')
plt.title('Mel Scale vs Frequency')
plt.grid(True)

# Visualizing the subset ranges (since it's divided into subsets of 128)
subset_size = 128
mel_ranges = [(i, min(i + subset_size, 2048)) for i in range(0, 2048, subset_size)]

# Plot frequency vs subset ranges
plt.subplot(1, 2, 2)
for start, end in mel_ranges:
    plt.axvspan(lin_hz_vector[start], lin_hz_vector[end - 1], color='orange', alpha=0.3)
plt.plot(lin_hz_vector, mel_scale_vector, label='Mel Scale')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Mel Scale')
plt.title('Subset Ranges Highlighted')
plt.grid(True)

plt.tight_layout()
plt.show()
