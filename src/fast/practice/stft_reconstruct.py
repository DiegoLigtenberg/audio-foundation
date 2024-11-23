import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.transforms import GriffinLim

h = list(range(1290,1400))


# Parameters
N_FFT = 2046  # FFT size
HOP_LENGTH = 862  # Overlap size
SAMPLE_RATE = 22050  # Desired sample rate
CHUNK_DURATION = 10  # Duration in seconds to process

# Path to the audio file
audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"  # Replace with the actual path

# Load the audio file and ensure it's mono
waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True) # can be false too or manual normalize
print(waveform.shape)
if waveform.size(0) > 1:  # Convert to mono if stereo
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to the desired sample rate
if sample_rate != SAMPLE_RATE:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)

# Trim or pad to the specified duration
waveform = waveform[:, :SAMPLE_RATE * CHUNK_DURATION]  # Trim if too long
if waveform.size(1) < SAMPLE_RATE * CHUNK_DURATION:  # Pad if too short
    waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE * CHUNK_DURATION - waveform.size(1)))

# Compute the spectrogram (STFT) with Hann windowing
window = torch.hann_window(N_FFT)
stft_result = torch.stft(waveform.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=window, return_complex=True)

# Compute the magnitude spectrogram (not squared)
magnitude_spectrogram = torch.abs(stft_result)  # This is the magnitude spectrogram (no squaring)

# Apply dynamic range compression (logarithmic scaling)
log_magnitude_spectrogram = torch.log(magnitude_spectrogram + 1e-6)

# Normalize the spectrogram for visualization
log_magnitude_spectrogram = log_magnitude_spectrogram - log_magnitude_spectrogram.max()  # Normalize to the max value

# Griffin-Lim reconstruction with more iterations
griffin_lim = GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH,power=1)

print(magnitude_spectrogram.shape)
# Reconstruct the waveform using Griffin-Lim
reconstructed_waveform = griffin_lim(magnitude_spectrogram)
print(reconstructed_waveform.shape)
# Ensure it's mono (single channel) and remove any extra dimensions
reconstructed_waveform = reconstructed_waveform.squeeze(0)  # Remove batch dimension (now shape: [time])


torchaudio.save("reconstructed_waveform_high_quality.wav", reconstructed_waveform.unsqueeze(0), SAMPLE_RATE)
asd
# # Create a single figure and axis
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot the original waveform
# ax.plot(waveform.numpy().squeeze(), label="Original Waveform", color="blue", alpha=0.7)

# # Plot the reconstructed waveform
# ax.plot(reconstructed_waveform.numpy(), label="Reconstructed Waveform", color="orange", alpha=0.7)

# # Adding title and labels
# ax.set_title("Comparison of Original and Reconstructed Waveforms")
# ax.set_xlabel("Time [samples]")
# ax.set_ylabel("Amplitude")
# ax.legend(loc="upper right")

# Create a single figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the original waveform
ax.plot(waveform.numpy().squeeze(), label="Original Waveform", color="blue", alpha=0.7)

# Plot the reconstructed waveform
ax.plot(reconstructed_waveform.numpy(), label="Reconstructed Waveform", color="orange", alpha=0.7)

# Adding title and labels
ax.set_title("Comparison of Original and Reconstructed Waveforms")
ax.set_xlabel("Time [samples]")
ax.set_ylabel("Amplitude")
ax.legend(loc="upper right")

# Show the plot with blocking to allow zooming and inspection
plt.show(block=True)  # Keeps the plot window open until manually closed