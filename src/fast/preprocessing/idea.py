import torch
import time

'''pseuddocode dataloader'''
# 1) laad full  mp3
# 2) convert mono 
# 3) resample (try to get instantly good sample rate
# 4) stft on full mp3
# 5) sample slices from stft
batch_size = None
model = None
stft_result = None
collate_fn = None
loss_function = None

'''how to slice given a song'''
size = 512
nr_slices = 128

start_time = time()
steps = torch.arange(size)[..., None]
start_idxs = torch.randint(0, stft_result.size(1), (nr_slices,))[None, ...]
slices_idxs = steps + start_idxs
print(f"{time() - start_time:0.2f} sec - slicing idxs")

start_time = time()
stft_result = torch.nn.functional.pad(stft_result, (size, 0))
print(f"{time() - start_time:0.2f} sec - padding start")

start_time = time()
slices = stft_result[:, slices_idxs].contiguous()
print(f"{time() - start_time:0.2f} sec - slicing")

print(f"spectrogram shape: {stft_result.shape}")
print(f"slices shape: {slices.shape}")



''''how a batch looks like'''
batch = {
    "input": torch.tensor([batch_size, 512, 1024]), # shape
    "target": torch.tensor([batch_size, 1, 1024]), # shape
}

output = model(batch["input"])  # shape: [batch_size, 512, 1024] -> [batch_size, 1, 1024]
loss = loss_function(output, batch["target"])


'''collate_fn for batch generation''' 
slices_per_song = 2

uncollated_batch = [
    {"input": [(512, 1024), (512, 1024)], "target": [(1, 1024), (1, 1024)]},
    {"input": [(512, 1024), (512, 1024)], "target": [(1, 1024), (1, 1024)]},
    {"input": [(512, 1024), (512, 1024)], "target": [(1, 1024), (1, 1024)]},
    {"input": [(512, 1024), (512, 1024)], "target": [(1, 1024), (1, 1024)]},
]

batch = collate_fn(batch) = {
    "input": (8, 512, 1024),
    "target": (8, 1, 1024),
}