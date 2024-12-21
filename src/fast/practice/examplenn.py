import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import GriffinLim
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init

# Parameters
N_FFT = 2046*2  # FFT size
HOP_LENGTH = 1050  # Overlap size
SAMPLE_RATE = 22050*2  # Desired sample rate
CHUNK_DURATION = 30  # Duration in seconds to process
'''
# Path to the audio file
audio_file_path = "audio_files/Jo Blankenburg - Meraki Extended.mp3"  # Replace with the actual path

# Load the audio file and ensure it's mono
waveform, sample_rate = torchaudio.load(audio_file_path, normalize=False)

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

# Griffin-Lim reconstruction will be done after the model predicts the spectrogram
griffin_lim = GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1)

# Prepare the spectrogram for neural network input (flattening it)
input_spectrogram = magnitude_spectrogram.unsqueeze(0)  # Add batch dimension
output_waveform = griffin_lim(magnitude_spectrogram).unsqueeze(0)  # Add batch dimension for output (waveform after Griffin-Lim)


'''
# Define tanh function
def tanh(x: torch.Tensor) -> torch.Tensor:
    output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return (output - output.min()) / (output.max() - output.min())


def create_perceptual_weights_pytorch(nr_steps, min_frequency, max_frequency, bass_cutoff=150, 
                                      min_bass_weight=0.6, max_bass_weight=1.0, min_weight=0.2, 
                                      x_max=200, frequency_0=20, frequency_1=4000, shift_tanh=-3.5):
    # Generate frequency values (scaled to nr_steps)
    f = torch.linspace(min_frequency, max_frequency, nr_steps)

    # Calculate switch indices based on frequencies
    switch_idx_0 = int((bass_cutoff - min_frequency) / (max_frequency - min_frequency) * nr_steps)
    switch_idx_1 = int((frequency_1 - min_frequency) / (max_frequency - min_frequency) * nr_steps)

    # Ensure that switch_idx_0 and switch_idx_1 are within valid bounds
    switch_idx_0 = max(1, switch_idx_0)  # Ensure it's at least 1
    switch_idx_1 = max(switch_idx_0 + 1, switch_idx_1)  # Ensure it's greater than switch_idx_0

    # Create weight for the first part (bass region)
    if switch_idx_0 > 1:  # Avoid empty tensor
        weights_0 = tanh(torch.linspace(0.0, 3.0, switch_idx_0))
        weights_0 = weights_0 * (1.0 - min_bass_weight) + min_bass_weight
    else:
        weights_0 = torch.zeros(switch_idx_0)  # In case of invalid range, use zeros

    # Constant weight between bass_cutoff and frequency_1
    weights_1 = torch.ones(switch_idx_1 - switch_idx_0)

    # Create weight for the third part (high frequency region)
    weights_2 = tanh(torch.linspace(3.0, shift_tanh, nr_steps - switch_idx_1))
    weights_2 = weights_2 * (1.0 - max_bass_weight) + min_weight

    # Concatenate all weight parts
    weights = torch.cat([weights_0, weights_1, weights_2])

    # Compute Mel scale for the frequencies
    mel = torch.log(f / 700 + 1)  # Logarithmic Mel scale formula

    return f,  weights  # Return frequencies, Mel scale, and perceptual weights for plotting


# Parameters
nr_steps = 512 # 2024 or 2048
min_frequency = 20         # min freq thath should be predicted
max_frequency = 10000    #max frequency that should be predicted
high_min_weight = 0.6 # 0.1 (to change for base lvl)
low_min_weight = 0.6 #0.6 or 0.9 (.9 for good bass)
frequency_0 = 20 # 200
frequency_1 = 6000 # 2000
shift_tanh = -3.5 # 4.5 lower is earlier cutoff 

# Get the perceptual weights and Mel scale
frequencies, weights = create_perceptual_weights_pytorch(
    nr_steps, min_frequency, max_frequency, bass_cutoff=150, min_bass_weight=0.0, 
    max_bass_weight=1.0, min_weight=0.2, x_max=200
)
print(weights.shape)

# Plot the perceptual weights with frequency
plt.figure(figsize=(8, 6))
plt.plot(frequencies.numpy(), weights.numpy(), label="Perceptual Weights", color='r')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.title("Perceptual Weights vs Frequency")
# plt.xscale("log")
plt.grid(True)
plt.show()
asd

print(type(weights.shape[0]))
# weights[:int(0.25*weights.shape[0]):] =0
print(weights.shape)
for i,r in enumerate(weights):
    if i%100==00:
        pass
        # print(r)

def perceptual_loss(predicted_spectrogram, target_spectrogram, weights):
    # Ensure spectrograms have the same shape
    assert predicted_spectrogram.shape == target_spectrogram.shape, (
        f"Spectrograms must have the same shape: {predicted_spectrogram.shape} vs {target_spectrogram.shape}"
    )

    # Get dimensions
    # batch_size, channels, freq_bins, time_steps = predicted_spectrogram.shape



    # Perform element-wise multiplication with the weights
    # weighted_target = target_spectrogram * weights  # Apply the same weights to both channels


    # Compute the loss using MSE
    loss = torch.nn.functional.mse_loss(predicted_spectrogram, # Create a zero-like tensor in the shape of weighted_target
                                       target_spectrogram ) * 1000
    return loss




class CNN_Autoencoder(nn.Module):
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE):
        super(CNN_Autoencoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Encoder
        self.encoder = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)  # Input has 2 channels (stereo)

        # Decoder
        self.decoder = nn.ConvTranspose2d(2 + 1, 2, kernel_size=3, stride=1, padding=1)  # Output 2 channels (stereo)
        # self._initialize_weights()
    
    def _initialize_weights(self):
        # Convolutional layers: Xavier (Glorot) initialization
        init.xavier_normal_(self.encoder.weight)
        init.xavier_normal_(self.decoder.weight)

        # Bias initialization: Zero initialization
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def forward(self, x):
        # Encoder: Apply convolution to input spectrogram
        encoded = self.encoder(x)
        
        # Skip connection: add the original input spectrogram (resized to match the encoder output)
        skip_connection = x
        skip_connection = F.interpolate(skip_connection, size=encoded.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate the encoded features and skip connection along the channel dimension
        encoded = torch.cat((encoded, skip_connection), dim=1)
        
        # Decoder: Apply transpose convolution to reconstruct the stereo output
        decoded = self.decoder(encoded)

        return decoded  # Stereo output (2 channels)

# Initialize model, loss function, and optimizer
model = CNN_Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.01)

weights = weights.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 2047)

weights = weights.view(1, 1, 2047, 1)  # Shape becomes (1, 1, 2047, 1)

# Now broadcast it to the shape of input_spectrogram (1, 2, 2047, 1261)
weights = weights.expand(1, 2, 2047, 1261)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    # criterion = nn.MSELoss()
    output_spectrogram = model(input_spectrogram)


    # Now reshape it to the target shape
    weights = weights.reshape(1, 2, 2047, 1261)
    input_spectrogram =  input_spectrogram*weights
    # print(weights)

    # weights = weights.expand(1, 2, 2047, 2047)  # Expand to [1, 2, 2047, 2047], adjust as needed

    input_spectrogram = input_spectrogram*weights
    
    # Compute perceptual loss
    loss = perceptual_loss(output_spectrogram, input_spectrogram, weights)
    # Compute the loss
    # loss = criterion(output_spectrogram, input_spectrogram)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, generate the predicted spectrogram
model.eval()
predicted_spectrogram = model(input_spectrogram).detach()
# Calculate the threshold frequency index (80% of 2047)
# freq_limit = int(0.4 * 2047)

# Zero out the part of the spectrogram above the threshold frequency
# predicted_spectrogram[:, :, freq_limit:, :] = 0
# print(predicted_spectrogram.min(),predicted_spectrogram.max(),predicted_spectrogram.shape) 
# plt.figure(figsize=(10, 6))
# plt.imshow(predicted_spectrogram[0, 0].numpy(), aspect='auto', origin='lower')
# plt.title('Modified Spectrogram (Frequency above freq_limit set to 0)')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()
# Reconstruct the waveform using Griffin-Lim from the predicted spectrogram
predicted_waveform = griffin_lim(predicted_spectrogram.squeeze(0)).unsqueeze(0)

# Save the predicted waveform
torchaudio.save("predicted_waveform_from_spectrogram.wav", predicted_waveform.squeeze(0), sample_rate=SAMPLE_RATE)

# Optionally: Plot the waveform
# plt.figure(figsize=(10, 4))
# plt.plot(predicted_waveform.squeeze(0).numpy())
# plt.title("Predicted Waveform from Spectrogram")
# plt.xlabel("Time (samples)")
# plt.ylabel("Amplitude")
# plt.show()
