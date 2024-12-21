import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class WeightedLoss(torch.nn.Module):
    def __init__(self, min_frequency, max_frequency, nr_steps, frequency_0, frequency_1, 
                 low_min_weight=0.6, high_min_weight=0.1, shift_tanh=-4.5):
        super(WeightedLoss, self).__init__()
        
        # Store parameters for later use
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.nr_steps = nr_steps
        self.frequency_0 = frequency_0
        self.frequency_1 = frequency_1
        self.low_min_weight = low_min_weight
        self.high_min_weight = high_min_weight
        self.shift_tanh = shift_tanh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute the weights
        self.weights = self.create_perceptual_weights()

    def tanh(self, x: torch.Tensor) -> torch.Tensor:
        output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        return (output - output.min()) / (output.max() - output.min())

    def create_perceptual_weights(self):
        # Generate frequency values (scaled to nr_steps)
        f = torch.linspace(self.min_frequency, self.max_frequency, self.nr_steps)
        
        # Calculate switch indices based on frequencies
        switch_idx_0 = int(self.frequency_0 / (self.max_frequency - self.min_frequency) * self.nr_steps) - 1
        switch_idx_1 = int(self.frequency_1 / (self.max_frequency - self.min_frequency) * self.nr_steps) - 1
        
        # Weight for the first part (bass region)
        weights_0 = self.tanh(torch.linspace(0.0, 3.0, switch_idx_0))
        weights_0 = weights_0 * (1.0 - self.low_min_weight) + self.low_min_weight
        
        # Constant weight between frequency_0 and frequency_1
        weights_1 = torch.ones(switch_idx_1 - switch_idx_0)
        
        # Weight for the third part (high frequency region)
        weights_2 = self.tanh(torch.linspace(3.0, self.shift_tanh, self.nr_steps - switch_idx_1))
        weights_2 = weights_2 * (1.0 - self.high_min_weight) + self.high_min_weight
        
        # Concatenate all weight parts
        weights = torch.cat([weights_0, weights_1, weights_2])
        return weights

    def forward(self, output, slice_batch_target):
        # Reshape weights to match the shape of output and target tensors
        weights = self.weights.view(1, 1, -1, 1).to(device=self.device)  # Shape: [1, 1, 512, 1]
        
        # Apply weights to both output and target
        weighted_output = output * weights
        weighted_target = slice_batch_target * weights
        
        # Calculate the MSE loss (or any other loss)
        loss = F.mse_loss(weighted_output, weighted_target)
        # Calculate the absolute difference
        # abs_diff = torch.abs(weighted_output - weighted_target)

        # Compute the exponential loss
        # loss = torch.mean(torch.exp(abs_diff))
        return loss
    def plot_weights(self):
        # Generate frequency values (scaled to nr_steps)
        f = torch.linspace(self.min_frequency, self.max_frequency, self.nr_steps)
        
        # Plot the weights against frequency
        plt.plot(f, self.weights.numpy())
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Weight")
        plt.ylim(0.0)  # Ensure the weight range starts from 0
        # Uncomment to enable log scale and limits for x-axis
        # plt.xlim(20, 300)
        # plt.xscale("log")
        plt.show()