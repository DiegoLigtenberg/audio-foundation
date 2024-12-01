from fast.model.model_channel import *
from fast.preprocessing.dataloader.dataloader import *
from fast.settings.directory_settings import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = ["audio_files/"]  # Directory containing audio files
    # dirs = [DATASET_MP3_DIR]
    transforms = [
        FileToWaveform(input_key="file_path", output_key="raw_waveform", target_sample_rate=SAMPLE_RATE, mono=MONO),
        WaveformToLogPowerSpectrogram(input_key="raw_waveform", output_key="log_power_spectrogram", n_fft=N_FFT, hop_length=HOP_LENGTH, keep_original=False),
        NormalizeLogPowerSpectrogram(input_key="log_power_spectrogram",output_key="normalized_spectrogram",method="minmax",global_min_max_file=GLOBAL_MIN_MAX_LOG_POWER_SPECTROGRAM,constant=150,keep_original=False),
        LogPowerSpectrogramSliceExtractor(input_key="normalized_spectrogram",output_keys=["input","target"],n_fft=N_FFT, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE, 
                                        slice_duration_sec=CHUNK_DURATION,n_slices=None,keep_original=False), # n _ slices is how many slices we sampel per song, None = dynamic sampling based on duration
    ] 

    output_key = 'input'  # Replace with your actual output key
    target_key = 'target'  # Replace with your actual target key

    # Hyperparameters
    seq_len = 1024  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_samples = 1000
    batch_song_size = 32
    batch_slice_size = 16    #gpt3 ->4096
    num_heads = 32//8           #gpt3 -> 96
    num_encoder_layers = 16//8  #gpt3 -> 96
    dim_feedforward = 8192//8   #gpt3 -> 48768
    learning_rate = 1e-5     #kerpathy constant
    num_epochs = 50000
    num_channels = 1 if MONO else 2

    # assert batch_slice_size, CHUNK_DURATION, batch_slice_size, for durations of 20 seconds batch size should be <= num songs * 2
    min_amt_slices_from_batch_song_size = int(batch_song_size * ((60+CHUNK_DURATION) // CHUNK_DURATION)) # min song duration is 60
    min_amt_slices_from_batch_song_size = 2 ** (min_amt_slices_from_batch_song_size.bit_length() - 1)
    assert batch_slice_size <= min_amt_slices_from_batch_song_size,(
    f"Batch slice size ({batch_slice_size}) exceeds the minimum possible slices ({min_amt_slices_from_batch_song_size})."
)

    dataset = AudioDataset(dir_audio=dirs, extensions=('.mp3',), transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_song_size, # hw many songs we load of n_slices
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, output_key, target_key),  # Pass keys explicitly
    )

    with open('rob_plot.json', 'r') as f:
        loaded_rob_plot = json.load(f)
    rob_plot = loaded_rob_plot
    plt.figure(figsize=(10, 6))
    plt.plot(rob_plot, label="Loss", marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # asd

    # Model, Loss, and Optimizer
    # model = TransformerModel(freq_size, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
    model_save_path = r"src/fast/model/model_weights/model.pth"  # Replace with your desired file path
    model = TransformerModel(freq_size, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    

    class SqrtMAELoss(nn.Module):
        def __init__(self):
            super(SqrtMAELoss, self).__init__()

        def forward(self, predictions, targets):
            # Calculate MAE
            mae = torch.mean(torch.abs(predictions - targets))
            # Return the square root of MAE
            return torch.sqrt(mae)

    criterion = nn.L1Loss()
    # criterion = SqrtMAELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Count the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # for epoch in num_epochs:
    model.train()
    total_loss = 0
    # Example loop over DataLoader
    # rob_plot = []
    for epoch in range(0,num_epochs):
        with TimeBlock() as timeblock:
            # with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
                # for i, batch in enumerate(tepoch):
                for i, batch in enumerate(dataloader):
                    timeblock.time_block()
                    # load s batch_slice_size (s is dynamic based on duration and size of batch_song_size)
                    input_tensor_batch = batch["input"]["data"]["input"]
                    target_tensor_batch = batch["input"]["data"]["target"]
                    # print(input_tensor_batch.shape)

                    # shuffle slices
                    permuted_indices = torch.randperm(input_tensor_batch.shape[0])
                
                    # Calculate the largest multiple of batch_slice_size <= total_slices
                    num_slices_to_use = (input_tensor_batch.shape[0] // batch_slice_size) * batch_slice_size
                    # Trim the tensors to the slices we can actually use
                    input_tensor_batch = input_tensor_batch[:num_slices_to_use]
                    target_tensor_batch = target_tensor_batch[:num_slices_to_use]

                    # the amount of input_tensor_batch slices is based on the amount we have from dynamic duration songs, and the largest multiple of slice_batch_size <= total_slices
                    slices_batches = torch.split(input_tensor_batch, batch_slice_size)
                    slices_batches_target = torch.split(target_tensor_batch,batch_slice_size)

                    total_loss = 0
                    for batch_index, (slice_batch, slice_batch_target) in enumerate(zip(slices_batches, slices_batches_target)):
                        slice_batch = slice_batch.to(device)
                        slice_batch_target = slice_batch_target.to(device)
                        optimizer.zero_grad()
                        output = model(slice_batch)
                        loss = criterion(output, slice_batch_target)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        # print(f'{batch_index} - {loss.item()}')
                        # print(output.shape)

                        # asd
                    rob_plot.append(total_loss / len(slices_batches))
                    print(f'{epoch} - {total_loss / len(slices_batches)}')
                    torch.save(model.state_dict(), model_save_path)
                    # Load from JSON file
                    with open('rob_plot.json', 'w') as f:
                        json.dump(rob_plot, f)

                    # if epoch >=10:
                        # plt.figure(figsize=(10, 6))
                        # plt.plot(rob_plot, label="Loss", marker='o')
                        # plt.title("Loss per Epoch")
                        # plt.xlabel("Epoch")
                        # plt.ylabel("Loss Value")
                        # plt.legend()
                        # plt.grid(True)
                        # plt.show()

                        # print(slice_batch.shape,input_tensor_batch.shape)

                        # tepoch.set_postfix(loss=loss.item())
                    # Clear memory if needed
                    # del spectrogram, target, output  # Explicitly delete the variables




                
                # with tqdm(dataloader, unit="batch") as tepoch:
                #     tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
                #     for spectrogram, target in tepoch:
                        # Move data to GPU


