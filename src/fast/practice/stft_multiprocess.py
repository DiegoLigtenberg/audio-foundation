import torch
import torchaudio
import time
from multiprocessing import Pool
import os 
# Parameters
N_FFT = 2046  # FFT size
HOP_LENGTH = 862  # Overlap size
SAMPLE_RATE = 22050  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process
AUDIO_FILES = ["audio_files/Jo Blankenburg - Meraki Extended.mp3"] * 10  # List of audio files to process (10 copies for demo)
SAVE_PATH = "test.pt"

# num_workers = os.cpu_count()
# print(f"Number of available CPU workers: {num_workers}")

# Function to process each audio file
def process_audio_file(audio_file_path):
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

    # Apply a Hann window to the waveform to reduce spectral leakage
    window = torch.hann_window(N_FFT, device=waveform.device)

    # Compute the spectrogram (STFT) with windowing
    stft_result = torch.stft(waveform.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window=window, return_complex=True)


    # print(f"Processed: {audio_file_path} | Shape: {stft_result.shape}")
    
    # Save the result (consider using different file names to avoid overwriting)
    torch.save(stft_result, f"{SAVE_PATH}_{audio_file_path.split('/')[-1].split('.')[0]}.pt")

    return stft_result.shape

# Main function to run multiprocessing
def process_audio_files_in_parallel(file_paths, num_workers=4,i=0):
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        results = pool.map(process_audio_file, file_paths)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{i} Workers - Processed {len(file_paths)} files in {elapsed_time:.2f} seconds.")
    return results

# This block ensures that multiprocessing works on Windows
if __name__ == '__main__':
    # Example usage with 10 audio files (you can replace this with your actual files)
    for i in [1,2,4,8,16]:
        results = process_audio_files_in_parallel(AUDIO_FILES, num_workers=i,i=i)
