import torch

SAMPLE_RATE = 44100
CHUNK_DURATION = 20  # in seconds
N_FFT = 2046*2
HOP_LENGTH = 1050

# Generate waveform
waveform_length = SAMPLE_RATE * CHUNK_DURATION
waveform = torch.randn(1, waveform_length)

# Apply STFT
stft_result = torch.stft(waveform[0], n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hann_window(N_FFT), return_complex=True)
print(stft_result[-1:].shape[1])

print(torch.ceil(torch.tensor(CHUNK_DURATION * SAMPLE_RATE) /HOP_LENGTH))
asd
# Get spectrogram shape
spectrogram_shape = stft_result.shape
print(stft_result[:,-1].shape)
