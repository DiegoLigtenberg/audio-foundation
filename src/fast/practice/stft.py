import torch
import torchaudio
import matplotlib.pyplot as plt
import time
from fast.settings.directory_settings import *
# Parameters
N_FFT = 2046  # FFT size
HOP_LENGTH = 862  # Overlap size
SAMPLE_RATE = 22050*2  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process
start_time = time.time()
# Path to the audio file
waveform, sample_rate = None, None
for i in range(1):
    audio_file_path = DATASET_MP3_DIR / "0015636.mp3"   # Replace with the actual path

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

    # print((stft_result).shape)
    # asd
    save_path = "test.pt"
    # torch.save(stft_result, save_path)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to save: {elapsed_time:.4f} seconds")
# asd

stft_result[0][0] = 2434.3056640625*1.3 # max value in dataset, imagine there we find 1*3 x louder in real
magnitude_spectrogram = torch.abs(stft_result)
sqrt_spectrogram = torch.sqrt(torch.sqrt(magnitude_spectrogram))
power_spectrogram = magnitude_spectrogram ** 2 
log_power_spectrogram = 20* torch.log10(power_spectrogram + 1e-8)
min_val = -159.9687
max_val = 135.4550
# log_power_spectrogram = (log_power_spectrogram - min_val) / (max_val - min_val)

reversed_power_spectrogram = 10 ** (log_power_spectrogram / 20)
reversed_magnitude_spectrogram = torch.sqrt(reversed_power_spectrogram)



print(log_power_spectrogram.min(),log_power_spectrogram.max(),log_power_spectrogram.mean(),log_power_spectrogram.median())
print(magnitude_spectrogram.min(),magnitude_spectrogram.max(),magnitude_spectrogram.mean(),magnitude_spectrogram.median())
# Normalize the spectrogram for visualization
# log_power_spectrogram = log_power_spectrogram - log_power_spectrogram.max()  # Normalize to the max value

# Plot the log compressed power spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(magnitude_spectrogram.numpy(), aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(label='Log Power')
plt.xlabel('Time Frames')
plt.ylabel('Frequency Bins')
plt.title('Dynamic Range Compressed Power Spectrogram')
plt.tight_layout()
plt.show()
