import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy.io.wavfile import write  # For saving audio as WAV
import matplotlib.pyplot as plt
from pydub import AudioSegment  # For converting WAV to MP3

# Parameters
N_FFT = 4096*2  # Increased FFT size for higher frequency resolution
HOP_LENGTH = N_FFT // 4  # Overlap size
SAMPLE_RATE = 44100  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process

# Adjust the hop length to a smaller value for higher time resolution
# HOP_LENGTH = N_FFT // 4  # Smaller hop length (more overlap)

# Optional: Adjust the window length to control frequency resolution
WIN_LENGTH = N_FFT  # You can experiment with different values for better tradeoff

# Path to the audio file
audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"  # Replace with the actual path

# Load the audio file and ensure it's mono
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

# Mel Spectrogram with updated parameters (256 Mel bins)
n_mels = 256  # Reduced Mel bins for compatibility with the frequency resolution
window_fn = torch.hann_window

# Mel Spectrogram transformation
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=n_mels,  # Set n_mels to 256
    window_fn=window_fn,
    power=1.0  # Log scale (usually works better for perceptual tasks)
)

mel_spectrogram = mel_transform(waveform)

# Inverse Mel Scale to convert Mel spectrogram back to linear spectrogram
inverse_mel = T.InverseMelScale(n_mels=n_mels, n_stft=N_FFT // 2 + 1, sample_rate=SAMPLE_RATE)
linear_spectrogram = inverse_mel(mel_spectrogram)

# Griffin-Lim reconstruction: Convert the linear spectrogram back to the waveform
griffin_lim = T.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1)
reconstructed_waveform = griffin_lim(linear_spectrogram)

# Ensure it's mono (single channel) and remove any extra dimensions
reconstructed_waveform = reconstructed_waveform.squeeze(0)  # Remove batch dimension (now shape: [time])

# Convert to numpy array for export
reconstructed_waveform_numpy = reconstructed_waveform.detach().cpu().numpy()

# Save the reconstructed waveform as a WAV file
write('reconstructed_audio.wav', SAMPLE_RATE, reconstructed_waveform_numpy)

# Optionally, convert the WAV file to MP3 using pydub (requires ffmpeg installed)
audio_segment = AudioSegment.from_wav('reconstructed_audio.wav')
audio_segment.export('reconstructed_audio.mp3', format='mp3')

# Convert the mel spectrogram to dB scale for visualization
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
print(mel_spectrogram_db.shape)

# Plot the mel spectrogram (optional)
plt.figure(figsize=(10, 6))
plt.imshow(mel_spectrogram_db[0].detach().numpy(), cmap='inferno', origin='lower', aspect='auto', interpolation='none')
plt.title('Mel Spectrogram')
plt.xlabel('Time Frames')
plt.ylabel('Mel Frequency Bins')
plt.colorbar(format='%+2.0f dB')
plt.show()
