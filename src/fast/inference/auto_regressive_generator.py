from fast.model.model_channel import *
from fast.preprocessing.dataloader.dataloader import *
from fast.settings.directory_settings import *

class GeneratorDataset():
    def __init__(self, model_save_path, freq_size, num_channels, seq_len, num_heads, num_decoder_layers, dim_feedforward, device, transforms) -> None:
        self.transforms = transforms
        self.freq_size = freq_size
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.device = device
        
        # Initialize the model with the same architecture
        self.model = TransformerModel(freq_size, num_channels, seq_len, num_heads, num_decoder_layers, dim_feedforward).to(self.device)
        # Load the state dictionary into the model
        self.model.load_state_dict(torch.load(model_save_path,weights_only=True))
        # Set the model to evaluation mode (important for inference)
        self.model.eval()

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

        while True:
            try:
                # Apply all transformations
                for transform in self.transforms:
                    data_dict = transform(data_dict)
                return data_dict  # If successful, return the data dict

            except Exception as e:            
                print(f"Error during transformation: {e}")
                return data_dict  # Return the data dict even if transformations fail

    def load_target_data(self, idx):
        """Load the target data for a given index. Replace with actual logic."""
        # Example: Here you would load your target spectrogram slice or whatever target data you have
        return torch.zeros(1, self.num_channels, self.freq_size, self.seq_len)


class AutoregressiveSpectrogramGenerator:
    def __init__(self, model, input_key="log_power_spectrogram_slices", output_key="reconstructed_log_power_spectrogram_song", padding_value=0.0, n_slices=5, max_steps=100,device=None):
        self.model = model  # The trained model for generating the next token
        self.input_key = input_key
        self.output_key = output_key
        self.padding_value = padding_value
        self.max_steps = max_steps
        self.device = device

    def __call__(self, data_dict):
        # Extract input data from the target (as per your GeneratorDataset structure)
        input_data = data_dict["target"]["data"]

        batch_size, c, f, t = input_data.shape
        batch_size_t, c_t, f_t, t_t = input_data.shape
        t_t =1 
    
        # Initialize with empty padded data (same shape as the full spectrogram)
        generated_spectrogram = torch.full((batch_size, c, f, t), self.padding_value, dtype=input_data.dtype).to(self.device)
        generated_spectrogram_target = torch.full((batch_size_t, c_t, f_t, t_t), self.padding_value, dtype=input_data.dtype).to(self.device)
        for step in range(self.max_steps):
            # Predict the next slice
            prediction = self.model(generated_spectrogram)  # Model predicts the next slice
            asd

            # Insert the prediction into the generated_spectrogram at the current timestep
            generated_spectrogram[:, :, :, step] = prediction[:, :, :, 0]  # Insert predicted slice

            # Now, remove the oldest timestep from current_input and append the new prediction
            generated_spectrogram = torch.cat((generated_spectrogram[:, :, :, 1:], prediction), dim=-1)  # Shift left, append new prediction
      

            # if step >= self.n_slices - 1:
            #     break

        # Add the reconstructed spectrogram to the output dictionary
        data_dict["target"]["data"] = generated_spectrogram  # Modify the correct field
        return data_dict



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 1024  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_heads = 32           # gpt3 -> 96
    num_encoder_layers = 16  # gpt3 -> 96
    dim_feedforward = 8192   # gpt3 -> 48768
    num_channels = 1 if MONO else 2
    model_save_path = r"src/fast/model/model_weights/model.pth"

    # Initialize the generator model here
    gen_model = TransformerModel(freq_size, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
    gen_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    gen_model.eval()

    # inp = torch.zeros(1, num_channels, freq_size, seq_len).to(device)
    # outp  = torch.zeros(1, num_channels, freq_size, 1).to(device)

    # prediction = gen_model(inp,outp) 

    # asd

    # Instantiate the transforms with the model
    transforms = [
        AutoregressiveSpectrogramGenerator(model=gen_model,device=device),
    ]

    # Instantiate the dataset
    gen_dataset = GeneratorDataset(model_save_path, freq_size, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward, device, transforms)

    # Accessing the dataset with an index
    idx = 0  # Example index
    data_dict = gen_dataset[idx]

    # Optionally, loop through dataset
    # for data in gen_dataset:
        # pass
