import torch
import torchaudio
import matplotlib.pyplot as plt
import time

# Parameters
N_FFT = 2046  # FFT size
HOP_LENGTH = 862//2  # Overlap size
SAMPLE_RATE = 22050  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process
start_time = time.time()
# Path to the audio file
waveform, sample_rate = None, None
for i in range(10):
    audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"  # Replace with the actual path

    # Load the audio file and ensure it's mono

    if i <= 0: 
        waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True)
    if waveform.size(0) > 1:  # Convert to mono if stereo
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to the desired sample rate
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)

    # Trim or pad to the specified duration
    waveform = waveform[:, :SAMPLE_RATE * CHUNK_DURATION]  # Trim if too long
    if waveform.size(1) < SAMPLE_RATE * CHUNK_DURATION:  # Pad if too short
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE * CHUNK_DURATION - waveform.size(1)))

    # Compute the spectrogram (STFT)
    stft_result = torch.stft(waveform.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, return_complex=True)

    print((stft_result).shape)
    asd
    save_path = "test.pt"
    torch.save(stft_result, save_path)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to save: {elapsed_time:.4f} seconds")
asd


# Compute the power spectrogram (magnitude squared)
power_spectrogram = torch.abs(stft_result) ** 2  # This is the power spectrogram

# Apply dynamic range compression (logarithmic scaling)
# Adding a small value to avoid log(0)
log_power_spectrogram = torch.log(power_spectrogram + 1e-6)

# Normalize the spectrogram for visualization
log_power_spectrogram = log_power_spectrogram - log_power_spectrogram.max()  # Normalize to the max value

# Plot the log compressed power spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(log_power_spectrogram.numpy(), aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(label='Log Power')
plt.xlabel('Time Frames')
plt.ylabel('Frequency Bins')
plt.title('Dynamic Range Compressed Power Spectrogram')
plt.tight_layout()
plt.show()
