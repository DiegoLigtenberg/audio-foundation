import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch
# File path
audio_path = 'audio_files/00001.wav'  # Replace with your audio file path

# Load the audio file with the default 'soundfile' backend
waveform, sample_rate = torchaudio.load(audio_path, normalize=True)  # Normalize ensures it's in [-1, 1] range

# Normalize the waveform to 0 dBFS (max amplitude of 1)
# Assuming that 'normalize=True' already puts the audio in [-1, 1] range, no need to divide by 32768.
# But for safety, we can ensure the loudest point is at 1.0 for 0 dBFS scaling.
waveform = waveform / waveform.abs().max()  # This will ensure the peak is normalized to 0 dBFS.

# Check the min and max values after normalization
print("Waveform min:", waveform.min().item())
print("Waveform max:", waveform.max().item())  # Should be in the range [-1.0, 1.0] now.

# Define the duration (3 seconds)
duration = 50  # seconds

# Calculate the number of frames to load
num_frames = int(duration * sample_rate)

# Set the offset to start from the 150th second
offset = int(sample_rate * 1)  # Start from the 150th second

# Slice the waveform to get the 3-second segment
waveform_segment = waveform[:, offset:offset + num_frames]

# Step 2: Create the MelSpectrogram transformation with adjusted parameters
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=4096,           # Window size (e.g., 400 samples)
    hop_length=1050,      # Hop length (e.g., 160 samples)
    n_mels=64,           # Increased Mel bins for more resolution
    f_min=20,            # Minimum frequency to start at 20Hz
    f_max=sample_rate // 2  # Maximum frequency (Nyquist frequency)
)

# Step 3: Apply the MelSpectrogram transformation
mel_spec = mel_spectrogram(waveform_segment)

# Step 4: Convert the Mel spectrogram to dBFS (0 dB corresponds to max amplitude)
# Here we use `top_db=None` to avoid reducing the dynamic range artificially
amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=None)  # Use None to preserve range
mel_spec_db = amplitude_to_db(mel_spec)

# Step 5: Apply Limiting to ensure no values go above 0 dB
# Limiting the spectrogram to a maximum of 0 dB
mel_spec_db = torch.clamp(mel_spec_db, min=-100.0, max=0.0)  # Limit dB values to a practical range

print(mel_spec_db.shape)
# asd
# Step 6: Visualize the Mel Spectrogram
plt.figure(figsize=(10, 4))
# Convert the Mel spectrogram to a numpy array and plot with a better colormap
plt.imshow(mel_spec_db[0].detach().numpy(), cmap='plasma', origin='lower', aspect='auto', extent=[0, mel_spec.size(1), 0, mel_spec.size(0)])
plt.colorbar(format="%+2.0f dB")
plt.title('Mel Spectrogram with Dynamic Range Compression (0 dBFS Limiting)')
plt.xlabel('Time')
plt.ylabel('Mel Frequency bins')
plt.tight_layout()
plt.show()
