import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_len, input_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        spectrogram = torch.rand(self.seq_len, self.input_size)
        target = torch.rand(1, self.input_size)
        return spectrogram, target
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, freq_size, num_channels, seq_len, num_heads, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()

        # Define freq based on num_channels
        freq_size = freq_size * num_channels

        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(freq_size, max_len=seq_len)
        
        # Channel Attention
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=freq_size,  # Each channel has the same embedding size
            num_heads=num_heads,
            batch_first=True
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=freq_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output Projection
        self.fc_out = nn.Linear(freq_size, freq_size)


    def forward(self, x, tgt):
        # x has shape [batch, channel, frequency, time]
        batch_size, num_channels, frequency, time = x.size()
    
        # Reshape [B x C x F x T] -> [B, T, C*F] for input sequence
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time, num_channels * frequency)  
        # Apply positional encoding to input (x) only
        x = self.positional_encoding(x)  # Apply positional encoding to time (seq_len = time)
        
        # Now the target (tgt) needs to be reshaped as well (since it's the target sequence)
        tgt = tgt.permute(0, 3, 1, 2).reshape(batch_size, 1, num_channels * frequency)  # Reshape tgt
        
        # Apply transformer decoder (predicting a single token)
        out = self.decoder(tgt, x)  # [batch, seq_len=1, input_size]
        
        # Output projection
        out = self.fc_out(out)
        # Check if the output is 3-dimensional (as expected: [B, 1, C*F])

        # Reshape the output back to [batch, channels, frequency, time]
        out = out.reshape(batch_size, num_channels, frequency, 1)  # Reshape to desired output shape
        return out



if __name__ == "__main__":

    # Hyperparameters
    seq_len = 1024  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_samples = 1000
    batch_slice_size = 8          #gpt3 ->4096
    num_heads = 32           #gpt3 -> 96
    num_decoder_layers = 16  #gpt3 -> 96
    dim_feedforward = 8192   #gpt3 -> 48768
    learning_rate = 1e-4
    num_epochs = 2

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = DummyDataset(num_samples=num_samples, seq_len=seq_len, input_size=freq_size)
    dataloader = DataLoader(dataset, batch_size=batch_slice_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = TransformerModel(freq_size, seq_len, num_heads, num_decoder_layers, dim_feedforward).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Count the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    decoder_layer = model.decoder.layers[0]  # Get the first layer of the decoder
    # print(decoder_layer)
    # print(f"Query weight size: {decoder_layer.multihead_attn.in_proj_weight.size()}")
    # print(f"Key weight size: {decoder_layer.multihead_attn.in_proj_weight.size()}")
    # print(f"Value weight size: {decoder_layer.multihead_attn.in_proj_weight.size()}")

    # import sys
    # sys.exit()


    # Training Loop with timing
    start_training = time.time()  # Start timing the entire training process

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_epoch = time.time()  # Start timing the current epoch
        
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for spectrogram, target in tepoch:
                # Move data to GPU
                spectrogram = spectrogram.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                target = torch.zeros_like(target).to(device)  # Initialize the target input with zeros
                output = model(spectrogram, target)
                
                print(spectrogram.shape,target.shape,output.shape)
                asd
                loss = criterion(output[:, -1, :], target.squeeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        end_epoch = time.time()  # End timing the current epoch
        epoch_time = end_epoch - start_epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}, Time: {epoch_time:.2f} seconds")

    end_training = time.time()  # End timing the entire training process
    total_training_time = end_training - start_training
    print(f"Total training time: {total_training_time:.2f} seconds")