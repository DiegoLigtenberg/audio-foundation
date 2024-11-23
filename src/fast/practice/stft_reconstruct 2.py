import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.transforms import GriffinLim
from torchaudio.transforms import Resample
import torch.nn.functional as F
from fast.settings.directory_settings import *
h = list(range(1290,1400))


# Parameters
N_FFT = 2046  # FFT size
HOP_LENGTH = 861  # Overlap size  (862 gives 512 time)
SAMPLE_RATE = 22050  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process
MONO = True
# Path to the audio file
# audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"  # Replace with the actual path
audio_file_path = DATASET_MP3_DIR / "0015636.mp3"  # Replace with the actual path
# audio_file_path = r"D:\Users\Diego Ligtenberg\Downloads\WORLDS DEEPEST BASS TEST EVER.mp3"

# Load the audio file and ensure it's mono
waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True) # can be false too or manual normalize
# Calculate the total number of samples and the starting index
total_samples = waveform.size(1)  # Total number of samples
start_sample = int(total_samples * 0.25)  # Start at 25% of the total duration

# Slice the waveform to start from 25% to the end
waveform = waveform[:, start_sample:]
# print(sample_rate)
# asd
# Resample to the desired sample rate (e.g., SAMPLE_RATE)
if sample_rate != SAMPLE_RATE:
    resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    waveform = resampler(waveform)

if MONO:
    if waveform.size(0) > 1:  # Convert to mono if stereo
        waveform = waveform.mean(dim=0, keepdim=True)

# Trim or pad to the specified duration
# waveform = waveform[:, :SAMPLE_RATE * CHUNK_DURATION]  # Trim if too long
if waveform.size(1) < SAMPLE_RATE * CHUNK_DURATION:  # Pad if too short
    waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE * CHUNK_DURATION - waveform.size(1)))

# Compute the spectrogram (STFT) with Hann windowing
window = torch.hann_window(N_FFT)
stft_result = torch.stft(waveform.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=window, return_complex=True)


# Compute the magnitude spectrogram (not squared)
magnitude_spectrogram = torch.abs(stft_result)  # This is the magnitude spectrogram (no squaring)
gaussian_noise = torch.normal(mean=0, std=0.2, size=(512, magnitude_spectrogram.size(1)))
# print(gaussian_noise.shape)
# Add noise to every 2nd frequency bin
magnitude_spectrogram[1:200:2, :] += torch.abs(gaussian_noise[:100:])
print(magnitude_spectrogram.mean(),magnitude_spectrogram.min(),magnitude_spectrogram.max())
print(magnitude_spectrogram.shape)
# Griffin-Lim reconstruction with more iterations
griffin_lim = GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH,power=1)
# Reconstruct the waveform using Griffin-Lim
reconstructed_waveform = griffin_lim(magnitude_spectrogram)
# asd

# If MONO is True, convert mono to stereo by adding effects
if MONO:
    # can add chatgpt ideas to make this sound nice stereo
    stereo_waveform = reconstructed_waveform.unsqueeze(0)
    # Save the stereo output
    torchaudio.save("reconstructed_stereo.wav", stereo_waveform, SAMPLE_RATE)

else:
    # If the signal is already stereo, save it directly (no changes needed)
    torchaudio.save("reconstructed_waveform_high_quality.wav", reconstructed_waveform, SAMPLE_RATE)

asd
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