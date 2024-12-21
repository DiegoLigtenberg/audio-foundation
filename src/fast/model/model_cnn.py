import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Residual Block (ResBlock)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        
        # A skip connection is applied after the second convolution
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x)  # Apply the skip connection
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Add the identity to the output (residual connection)
        out += identity
        out = self.relu(out)  # Final ReLU activation
        
        return out

# Autoencoder model with residual blocks in the encoder
class Autoencoder(nn.Module):
    def __init__(self, input_channels=1, num_filters=16):
        super(Autoencoder, self).__init__()
        self.num_filters = num_filters
        self.input_channels = input_channels

        # Encoder with residual blocks
        self.encoder = nn.Sequential(
            ResidualBlock(input_channels, num_filters, stride=2),  # E1: S=512, F=1 -> S=256, F=16
            ResidualBlock(num_filters, num_filters*2, stride=2),  # E2: S=256, F=16 -> S=128, F=32
            ResidualBlock(num_filters*2, num_filters*4, stride=2),  # E3: S=128, F=32 -> S=64, F=64
            ResidualBlock(num_filters*4, num_filters*8, stride=2),  # E4: S=64, F=64 -> S=32, F=128
            ResidualBlock(num_filters*8, num_filters*16, stride=2)  # E5: S=32, F=128 -> S=16, F=256
        )

        # Bottleneck (Latent space)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters*16, num_filters*16, kernel_size=3, stride=1, padding=1),  # Bottleneck: S=16, F=256
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(num_filters*16, num_filters*8, kernel_size=3, stride=1, padding=1),  # D1: S=16, F=256 -> S=16, F=128
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False),  # Upsample F, increase height (S)
            nn.ReflectionPad2d((0, 0, 0, 1)),  # Padding to adjust height correctly
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 2)),  # Downsample width, keep F

            nn.Conv2d(num_filters*8, num_filters*4, kernel_size=3, stride=1, padding=1),  # D2: S=16, F=128 -> S=8, F=64
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False),  # Upsample F, increase height
            nn.ReflectionPad2d((0, 0, 0, 1)),  # Padding to adjust height correctly
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 2)),  # Downsample width, keep F

            nn.Conv2d(num_filters*4, num_filters*2, kernel_size=3, stride=1, padding=1),  # D3: S=8, F=64 -> S=4, F=32
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False),  # Upsample F, increase height
            nn.ReflectionPad2d((0, 0, 0, 1)),  # Padding to adjust height correctly
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 2)),  # Downsample width, keep F

            nn.Conv2d(num_filters*2, num_filters, kernel_size=3, stride=1, padding=1),  # D4: S=4, F=32 -> S=2, F=16
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False),  # Upsample F, increase height
            nn.ReflectionPad2d((0, 0, 0, 1)),  # Padding to adjust height correctly
            nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 2)),  # Downsample width, keep F

            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),  # D5: S=2, F=16 -> S=1, F=1
            nn.ReLU(True),
            nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False),  # Upsample F, increase height
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),  # Downsample width, keep F

            nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1)  # D6: S=1, F=1 -> S=512, F=1 (output with 1 channel)
        )

    def forward(self, x):
        # Encoder path with residual blocks
        x = self.encoder(x)
        
        # Bottleneck (Latent representation)
        x = self.bottleneck(x)
        # print(x.shape)
        
        # Decoder path
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    # Check if CUDA is available and use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    seq_len = 512
    freq_size = 512

    # Model
    model = Autoencoder(input_channels=1, num_filters=64).to(device)  # Move model to GPU

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {count_parameters(model):,} trainable parameters.")

    # Dummy input
    input_tensor = torch.randn(16, 1, seq_len, freq_size).to(device)  # Batch of 16, with shape (1, 512, 512)

    # Forward pass
    output = model(input_tensor)
    print("Output Shape:", output.shape)  # Expected: [16, 1, 1, 512]
