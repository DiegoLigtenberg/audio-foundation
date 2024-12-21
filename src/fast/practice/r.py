import torch
from matplotlib import pyplot as plt


def tanh(x: torch.Tensor) -> torch.Tensor:
    output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return (output - output.min()) / (output.max() - output.min())


nr_steps = 512
min_frequency = 5
max_frequency = 512
high_min_weight = 0.1
low_min_weight = 0.6

frequency_0 = 5
frequency_1 = 300
shift_tanh = -4.5

f = torch.linspace(min_frequency, max_frequency, nr_steps)

switch_idx_0 = int(frequency_0 / (max_frequency - min_frequency) * nr_steps) - 1
switch_idx_1 = int(frequency_1 / (max_frequency - min_frequency) * nr_steps) - 1

weights_0 = tanh(torch.linspace(0.0, 3.0, switch_idx_0))
weights_0 = weights_0 * (1.0 - low_min_weight) + low_min_weight

weights_1 = torch.ones(switch_idx_1 - switch_idx_0)

weights_2 = tanh(torch.linspace(3.0, shift_tanh, nr_steps - switch_idx_1))
weights_2 = weights_2 * (1.0 - high_min_weight) + high_min_weight

weights = torch.concat([weights_0, weights_1, weights_2])

plt.plot(f, weights)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.ylim(0.0)
# plt.xlim(20, 300)
# plt.xscale("log")
plt.show()