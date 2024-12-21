from fast.model.model_channel import *
from fast.preprocessing.dataloader.dataloader import *
from fast.settings.directory_settings import *
from fast.model.model_cnn import *
from matplotlib.pylab import plt

class GeneratorDataset():
    def __init__(self, model_save_path, freq_size, num_channels, seq_len, num_heads, num_decoder_layers, dim_feedforward, device, transforms) -> None:
        self.transforms = transforms
        self.freq_size = freq_size
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.device = device
        
        # Initialize the model with the same architecture
        # self.model = TransformerModel(freq_size, num_channels, seq_len, num_heads, num_decoder_layers, dim_feedforward).to(self.device)
        # # Load the state dictionary into the model
        # self.model.load_state_dict(torch.load(model_save_path,weights_only=True))
        # # Set the model to evaluation mode (important for inference)
        # self.model.eval()

    def initialize_data_dict(self):
        """Helper method to initialize the data dictionary."""
        return {
            "input": {
                "data": {},  # Placeholder for waveform or transformed data
                "metadata": {} # Initialize the model dimensions B x C x F x T=1
            },
            "target": {
                "data": {},  # Placeholder for target data
                "metadata": {},  # Placeholder for target-specific metadata
            }
        }

    def __len__(self):
        # You can implement the length function to control the number of samples in your dataset
        return 1  # Example length, change as needed

    def __getitem__(self, idx):
        data_dict = self.initialize_data_dict()
        # Load your target data
        data_dict["target"]["data"] = self.load_target_data(idx)

        # while True:
            # Apply all transformations
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict  # If successful, return the data dict

            # except Exception as e:            
            #     print(f"Error during transformation: {e}")
            #     return data_dict  # Return the data dict even if transformations fail

    def load_target_data(self, idx):
        """Load the target data for a given index. Replace with actual logic."""
        # Example: Here you would load your target spectrogram slice or whatever target data you have
        return torch.zeros(1, self.num_channels, self.freq_size, self.seq_len)


class AutoregressiveSpectrogramGenerator:
    def __init__(self, model, input_key="log_power_spectrogram_slices", output_key="reconstructed_log_power_spectrogram_song", padding_value=0.0, n_slices=5, max_steps=1000, device=None):
        self.model = model  # The trained model for generating the next token
        self.input_key = input_key
        self.output_key = output_key
        self.padding_value = padding_value
        self.max_steps = max_steps
        self.device = device

    def __call__(self, data_dict):
        nr_model_tokens = 512
        current_sequence = torch.load("src/fast/model/model_weights/spectrogram_input2.pth", weights_only=True)[None, :, :, :nr_model_tokens].to(device)
        current_sequence = current_sequence[:,:,:current_sequence.shape[2]//2,:]
        # print(current_sequence.shape)
        # asd
        # current_sequence[:,:,:,0:50] = 0 # see how a swap in 50 tokens changes complete output (adverserial attack).
        full_sequence = torch.zeros(*current_sequence.shape[:3], nr_model_tokens + self.max_steps).to(device)
        full_sequence[:, :, :, :nr_model_tokens] = current_sequence

        for step in range(self.max_steps):
            current_sequence = full_sequence[:, :, :, step : nr_model_tokens + step]
            with torch.no_grad():  # No need to compute gradients for inference
                output = self.model(current_sequence)
                if step> 50:# == 0:
                    pass
                    # output = output + 0.1 * torch.randn_like(output)
                    # plt.figure(figsize=(10, 4))
                    # plt.imshow(output[0,0,:,:].cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
                    # print(output[0,0,:,:].shape)
                    # print(current_sequence[0,0,-2,:].unsqueeze(1).shape)
                    # asd
                    # plt.imshow(output[0,0,:,:].cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
                    # plt.colorbar(label='Log Power')
                    # plt.xlabel('Time Frames')
                    # plt.ylabel('Frequency Bins')
                    # plt.title('Dynamic Range Compressed Power Spectrogram')
                    # plt.tight_layout()
                    # plt.show()
                   
                    
                    # plt.imshow(current_sequence[0,0,-1,:].unsqueeze(1).cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
                    # plt.colorbar(label='Log Power')
                    # plt.xlabel('Time Frames')   
                    # plt.ylabel('Frequency Bins')
                    # plt.title('Dynamic Range Compressed Power Spectrogram')
                    # plt.tight_layout()
                    # plt.show()

            full_sequence[:, :, :, nr_model_tokens + step] = output[:, :, :, -1]

        data_dict["target"]["metadata"][self.output_key] = full_sequence

        return data_dict



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 512  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_heads = 32//4           #gpt3 -> 96
    num_encoder_layers = 16//4  #gpt3 -> 96
    dim_feedforward = 8192*2   #gpt3 -> 48768
    num_channels = 1 if MONO else 2
    model_save_path = r"src/fast/model/model_weights/model_1target_augment.pth"

    # Initialize the generator model here
    # gen_model = TransformerModel(freq_size//2, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
    # gen_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    # gen_model.eval()

    gen_model = Autoencoder(input_channels=1, num_filters=64).to(device)
    gen_model.load_state_dict(torch.load(model_save_path,weights_only=True))
    gen_model.eval()

    # inp = torch.zeros(1, num_channels, freq_size, seq_len).to(device)
    # outp  = torch.zeros(1, num_channels, freq_size, 1).to(device)

    # prediction = gen_model(inp,outp) 

    # asd

    # Instantiate the transforms with the model
    transforms = [
        AutoregressiveSpectrogramGenerator(input_key="",output_key="log_power_spectrogram",model=gen_model,device=device),
    ]

    # Instantiate the dataset
    gen_dataset = GeneratorDataset(model_save_path, freq_size//2, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward, device, transforms)

    # Accessing the dataset with an index
    idx = 0  # Example index
    data_dict = gen_dataset[idx]

    # extrat normalized model output
    normalized_spectrogram = (data_dict["target"]["metadata"]["log_power_spectrogram"])
    # print(normalized_spectrogram.shape)
    # asd
    zeros_after = torch.zeros((1, 1, normalized_spectrogram.shape[2], normalized_spectrogram.shape[3])).to(device)

    normalized_spectrogram = torch.cat((normalized_spectrogram, zeros_after), dim=2)
    # asd
    # denormalize
    with open("src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.json", 'r') as f:
        data = json.load(f)
    global_min,global_max = data["min_max"]
    log_power_spectrogram = normalized_spectrogram * (global_max - global_min) + global_min

    # print(log_power_spectrogram.shape)
    # asd
    griffin_lim = GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1)
    # Reconstruct the waveform from the log power spectrogram
    power_spectrogram = torch.pow(10, log_power_spectrogram / 20)  # Reverse the log operation

    # Assume magnitude spectrogram (as we have no phase information)
    magnitude_spectrogram = torch.sqrt(power_spectrogram)  # Reverse power spectrogram

    # Apply Griffin-Lim for each channel separately
    waveforms = []
    for channel_magnitude in magnitude_spectrogram:  # Loop through each channel
        # Use Griffin-Lim to reconstruct the waveform for the current channel
        waveform_channel = griffin_lim(channel_magnitude.to('cpu'))
        waveforms.append(waveform_channel)

    # Stack the waveforms for all channels (mono or stereo)
    stacked_waveform = torch.stack(waveforms, dim=0)  # Shape: [channels, time]
    stacked_waveform = stacked_waveform[0,:,:]

    torchaudio.save("reconstructed_waveform_auto_regresive.wav", stacked_waveform, SAMPLE_RATE)  # Use the appropriate sample rate


    print(stacked_waveform.shape)



    # Optionally, loop through dataset
    # for data in gen_dataset:
        # pass
