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
  
# Positional Encoding
'''
class PositionalEncoding(nn.Module):
    def __init__(self, freq_size, seq_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, freq_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, freq_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / freq_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scale
        self.register_buffer('pe', pe) # positional encoding is not a learnable parameter

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # x + self.alpha * self.pe[:, :x.size(1)]
        '''

class PositionalEncoding(nn.Module):
    def __init__(self, freq_size, seq_len=512):
        super(PositionalEncoding, self).__init__()
        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, freq_size))

    def forward(self, x):
        # Add the learned positional embedding
        return x + self.positional_embedding[:, :x.size(1)]
    
class CausalMask(nn.Module):
    def __init__(self, seq_len):
        super(CausalMask, self).__init__()
        # Precompute the causal mask for the maximum sequence length
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)  # Not a learnable parameter

    def forward(self, seq_len):
        # Return the causal mask sliced to the desired sequence length
        return self.mask[:seq_len, :seq_len]


class TransformerModel(nn.Module):
    def __init__(self, freq_size, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()

        # Define freq based on num_channels
        freq_size = freq_size * num_channels
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(freq_size, seq_len=seq_len)
        
       # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=freq_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Causal Mask
        self.causal_mask = CausalMask(seq_len)

        # Output Projection
        self.fc_out = nn.Linear(freq_size, freq_size)

    def forward(self, x):
        # x has shape [batch, channel, frequency, time]
        batch_size, num_channels, frequency, time = x.size()

        # Reshape [B, C, F, T] -> [B, T, C*F]
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time, num_channels * frequency)

        # Apply positional encoding
        x = self.positional_encoding(x)  # Shape remains [B, T, C*F]

        # Generate the causal mask for the current sequence length
        causal_mask = self.causal_mask(seq_len=time)  # Shape [time, time]

        # Pass through Transformer encoder
        # encoded = self.encoder(x, mask=causal_mask)  # Shape [B, T, C*F]
        encoded = self.encoder(x)
        # Output projection
        out = self.fc_out(encoded)  # Shape [B, T, C*F]
        out = out[:,-1:,:] # pick last prediction of time series

        out = torch.sigmoid(out)  # Shape [B, T, C*F]

        # Reshape back to [B, C, F, T]
        out = out.reshape(batch_size, 1, num_channels, frequency).permute(0, 2, 3, 1) # B, C, F, T
        return out


if __name__ == "__main__":

    # Hyperparameters
    seq_len = 1024  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_samples = 1000
    batch_slice_size = 8          #gpt3 ->4096
    num_heads = 32           #gpt3 -> 96
    num_encoder_layers = 16  #gpt3 -> 96
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
    model = TransformerModel(freq_size, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
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
            for input_tensor_batch, target_tensor_batch in tepoch:
                # Move data to GPU
                input_tensor_batch = input_tensor_batch.to(device)
                target_tensor_batch = target_tensor_batch.to(device)
                optimizer.zero_grad()
                target_tensor_batch = torch.zeros_like(target_tensor_batch).to(device)  # Initialize the target input with zeros
                output = model(input_tensor_batch, target_tensor_batch)
                
                print(input_tensor_batch.shape,target_tensor_batch.shape,output.shape)
                asd
                loss = criterion(output[:, -1, :], target_tensor_batch.squeeze(1))
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