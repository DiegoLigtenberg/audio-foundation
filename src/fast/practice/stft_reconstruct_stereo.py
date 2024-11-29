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
HOP_LENGTH = 861  # Overlap size
SAMPLE_RATE = 44100  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process
MONO = True  # Set to True for mono processing

# Path to the audio file
audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"

# Function to preprocess audio
def preprocess_audio(audio_file_path, target_sample_rate, chunk_duration, mono):
    # Load the audio file
    waveform, orig_sample_rate = torchaudio.load(audio_file_path, normalize=True)

    # Trim the start of the waveform to process only a portion
    total_samples = waveform.size(1)
    start_sample = int(total_samples * 0.25)  # Start at 25% of the total duration
    waveform = waveform[:, start_sample:]

    # Resample if necessary
    if orig_sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if required
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim or pad the waveform to the desired chunk duration
    target_length = target_sample_rate * chunk_duration
    waveform = waveform[:, :target_length]  # Trim if too long
    if waveform.size(1) < target_length:  # Pad if too short
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))
    print(waveform.shape)
    return waveform

# Function to compute STFT and perform Griffin-Lim per channel
def process_stft_and_reconstruct(waveform, n_fft, hop_length):
    # Window for STFT
    window = torch.hann_window(n_fft)

    # Handle each channel independently
    reconstructed_channels = []
    for channel in waveform:
        print(waveform.shape)
        # Compute STFT
        stft_result = torch.stft(
            channel, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=n_fft, 
            window=window, 
            return_complex=True
        )

        # Compute magnitude spectrogram
        magnitude_spectrogram = torch.abs(stft_result)

        # Griffin-Lim reconstruction
        griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length, power=1)
        reconstructed_channel = griffin_lim(magnitude_spectrogram)

        # Append reconstructed channel
        reconstructed_channels.append(reconstructed_channel)

    # Stack channels back together
    reconstructed_waveform = torch.stack(reconstructed_channels, dim=0)
    return reconstructed_waveform

# Main execution
waveform = preprocess_audio(audio_file_path, SAMPLE_RATE, CHUNK_DURATION, MONO)
reconstructed_waveform = process_stft_and_reconstruct(waveform, N_FFT, HOP_LENGTH)

# Save the reconstructed audio
output_path = "reconstructed_mono.wav" if MONO else "reconstructed_stereo.wav"
torchaudio.save(output_path, reconstructed_waveform, SAMPLE_RATE)
print(f"Reconstructed audio saved to: {output_path}")
