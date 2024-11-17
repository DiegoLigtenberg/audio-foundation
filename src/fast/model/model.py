import torch
import torch.nn as nn

class AudioTransformer(nn.Module):
    def __init__(self, input_dim, seq_length, d_model=512, num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(AudioTransformer, self).__init__()

        # Linear layer to embed 2048-dim frequency frames into `d_model` dimension
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding for sequence position information
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))
        
        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Final linear layer to map back to the original frequency dimension (2048)
        self.fc_out = nn.Linear(d_model, input_dim)

        # Set the max sequence length for masking
        self.seq_length = seq_length

    def forward(self, src, tgt):
        # Embed each frequency frame and add positional encoding
        src = self.input_embedding(src) + self.positional_encoding[:src.size(0), :]
        tgt = self.input_embedding(tgt) + self.positional_encoding[:tgt.size(0), :]

        # Generate the causal mask for the target sequence
        tgt_mask = self.generate_causal_mask(tgt.size(0))

        # Pass through the transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Map to original frequency dimensions
        return self.fc_out(output)

    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

# Example usage
input_dim = 2048  # Frequency bins
seq_length = 128  # Time frames
d_model = 512
num_heads = 8
num_layers = 6

model = AudioTransformer(input_dim=input_dim, seq_length=seq_length, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
src = torch.randn(seq_length, 32, input_dim)  # (sequence_length, batch_size, input_dim)
tgt = torch.randn(seq_length, 32, input_dim)  # (sequence_length, batch_size, input_dim)

output = model(src, tgt)
print(output.shape)  # Expected output shape: (sequence_length, batch_size, input_dim)
