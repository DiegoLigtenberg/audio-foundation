import torch
import torchaudio
from torchaudio.transforms import GriffinLim
from matplotlib import pyplot as plt
SAMPLE_RATE = 22050  # Desired sample rate

# Create perceptual weights with a smooth transition between bass and Mel regions
def create_perceptual_weights_pytorch(nr_steps, min_frequency, max_frequency, bass_cutoff=150, min_bass_weight=0.6, max_bass_weight=1.0, min_weight=0.2, x_max=200):
    # Generate frequencies: 'nr_steps' values from min to max frequency
    f = torch.linspace(min_frequency, max_frequency, nr_steps)

    # Apply logistic function for bass weights: Start at 0.6 and follow logistic curve after cutoff
    x0 = 1  # Start at x = 1
    k = 20 / (x_max - 1)  # Steepness factor based on x_max to ensure the logistic function reaches 1.0 at x_max
    bass_weights = min_bass_weight + (max_bass_weight - min_bass_weight) / (1 + torch.exp(-k * (f - x0)))  # Logistic function starting from x = 1 and reaching max_bass_weight

    # Compute Mel scale: log(f / 700 + 1)
    mel = torch.log(f / 700 + 1)
    
    # Calculate the mel weights using the difference of mel values
    mel_weights = torch.diff(mel)  # Resulting mel_weights has length (nr_steps - 1)
    
    # Normalize Mel weights between 0 and 1
    mel_weights = (mel_weights - mel_weights.min()) / (mel_weights.max() - mel_weights.min())
    mel_weights = mel_weights * (1.0 - min_weight) + min_weight
    
    # Ensure bass_weights and mel_weights align by padding them (to be of same length as frequencies)
    bass_weights = bass_weights[:-1]  # Remove last element to match size of mel_weights
    
    # Combine bass and mel weights directly without smooth blending
    weights = bass_weights * (f[:-1] < bass_cutoff).float() + mel_weights * (f[:-1] >= bass_cutoff).float()

    return f, mel, weights  # Return frequencies, mel (with one extra value), and weights



# Parameters
nr_steps = 1025
min_frequency = 20
max_frequency = 20000
bass_cutoff = 150  # Frequency cutoff for bass region
min_bass_weight = 0.2  # Start weight at 0.6
max_bass_weight = 1.0  # End weight at 1.0
min_weight = 0.2  # Base weight for Mel region
x_max = 150  # Maximum frequency value for the logistic function to reach 1.0

# Assuming you have your perceptual weights
frequencies, mel, weights = create_perceptual_weights_pytorch(
    nr_steps, min_frequency, max_frequency, bass_cutoff, min_bass_weight, max_bass_weight, min_weight, x_max
)
# Define threshold for keeping frequency bins (e.g., keep weights above 0.5)
threshold = 0.5
selected_bins = weights >= threshold  # This will give a boolean mask for bins with weight >= threshold

# Compute STFT for the audio
N_FFT = 2046
HOP_LENGTH = 862
audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"
waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True)
# Ensure the waveform is mono by averaging the channels
if waveform.shape[0] > 1:  # Check if stereo or multi-channel
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging channels

window = torch.hann_window(N_FFT)
stft_result = torch.stft(waveform.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=window, return_complex=True)
magnitude_spectrogram = torch.abs(stft_result)  # Get magnitude spectrogram

# Apply the boolean mask to keep only selected frequency bins
filtered_spectrogram = magnitude_spectrogram[selected_bins]
import torch.nn.functional as F
expected_freq_bins  =1024
# Reshape the spectrogram to match the expected 4D input for interpolation
# Add a batch dimension and a channel dimension
filtered_spectrogram = filtered_spectrogram.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, frequency_bins, time_steps)

# Interpolate to match the expected number of frequency bins
filtered_spectrogram_resized = F.interpolate(filtered_spectrogram, size=(expected_freq_bins, filtered_spectrogram.shape[3]), mode='bilinear', align_corners=False)

# Remove the added batch and channel dimensions after interpolation
filtered_spectrogram_resized = filtered_spectrogram_resized.squeeze(0).squeeze(0)



print(filtered_spectrogram.shape)
print(filtered_spectrogram_resized.shape)
# Reconstruct the waveform from the filtered spectrogram
griffin_lim = GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1)
reconstructed_waveform_filtered = griffin_lim(filtered_spectrogram_resized)
reconstructed_waveform_filtered = reconstructed_waveform_filtered.squeeze(0)  # Remove batch dimension (now shape: [time])


torchaudio.save("reconstructed_waveform_high_quality.wav", reconstructed_waveform_filtered.unsqueeze(0), SAMPLE_RATE)
# Plot the original and filtered spectrograms for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrogram.log2().numpy(), aspect="auto", origin="lower")
plt.title("Original Spectrogram")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_waveform_filtered.log2().numpy(), aspect="auto", origin="lower")
plt.title("Filtered Spectrogram (Custom)")
plt.colorbar()

plt.tight_layout()
plt.show()
