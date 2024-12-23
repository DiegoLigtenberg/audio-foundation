from fast.model.model_channel import *
from fast.preprocessing.dataloader.dataloader import *
from fast.settings.directory_settings import *
from fast.model.custom_loss import *
from fast.model.model_cnn import *
import matplotlib.pyplot as plt
from multiprocessing import Lock

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torchaudio.set_audio_backend("ffmpeg")
    max_workers = os.cpu_count() if os.cpu_count() is not None else 1
    # dirs = ["audio_files/"]  # Directory containing audio files
    dirs = [DATASET_MP3_DIR]
    transforms = [
        FileToWaveform(input_key="file_path", output_key="raw_waveform", target_sample_rate=SAMPLE_RATE, mono=MONO),
        WaveformToLogPowerSpectrogram(input_key="raw_waveform", output_key="log_power_spectrogram", n_fft=N_FFT, hop_length=HOP_LENGTH, keep_original=False),
        NormalizeLogPowerSpectrogram(input_key="log_power_spectrogram",output_key="normalized_spectrogram",method="minmax",global_min_max_file=GLOBAL_MIN_MAX_LOG_POWER_SPECTROGRAM,constant=150,keep_original=False),
        LogPowerSpectrogramSliceExtractor(input_key="normalized_spectrogram",output_keys=["input","target"],n_fft=N_FFT, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE, 
                                        slice_duration_sec=CHUNK_DURATION,n_slices=None,only_half_frequency=True,random_input_noise=0.01,max_mask_size=0.2,volume_reduction=0.1,mask_probability=0.05,keep_original=False), 
                                        # n _ slices is how many slices we sampel per song, None = dynamic sampling based on duration
    ] 

    output_key = 'input'  # Replace with your actual output key
    target_key = 'target'  # Replace with your actual target key
    collate_fn_with_keys = partial(collate_fn, output_key=output_key, target_key=target_key) # for multiprocessing

    # Hyperparameters
    seq_len = 512  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_samples = 1000
    # was 128 128, think bout lowering to 64 32, rethink what epoch means!
    batch_song_size = 64
    batch_slice_size = 32       #gpt3 ->4096      # 21 is seeing every part of song once on avg -> 3;30 song has 10752 time steps, 512 steps per chunk = 21 slices
    num_heads = 32//8           #gpt3 -> 96
    num_encoder_layers = 16//8  #gpt3 -> 96
    dim_feedforward = 8192//4   #gpt3 -> 48768
    learning_rate = 3e-4     #kerpathy constant
    num_epochs = 5000
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
        collate_fn= collate_fn_with_keys,
        num_workers=8,   #lambda batch: collate_fn(batch, output_key, target_key),  # Pass keys explicitly
        persistent_workers=True, # keep workers alive after epoch
    )

    with open('rob_plot5_target.json', 'r') as f:
        loaded_rob_plot = json.load(f)
    rob_plot = loaded_rob_plot
    plt.figure(figsize=(10, 6))
    plt.plot(rob_plot[50:], label="Loss", marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Model, Loss, and Optimizer
    # model = TransformerModel(freq_size//2, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
    # model = Autoencoder(input_channels=1, num_filters=64).to(device)
    model_save_path = r"src/fast/model/model_weights/model_1target_augment.pth"  # Replace with your desired file path
    model = Autoencoder(input_channels=1, num_filters=64).to(device)
    model.load_state_dict(torch.load(model_save_path,weights_only=True))

    # model = TransformerModel(freq_size//2, num_channels, seq_len, num_heads, num_encoder_layers, dim_feedforward).to(device)
    # model.load_state_dict(torch.load(model_save_path, weights_only=True))
    
    # Example usage
    nr_steps = 512
    min_frequency = 4
    max_frequency = 512
    frequency_0 = 4
    frequency_1 = 350
    shift_tanh = -4.5

    # Initialize the custom loss function
    criterion = WeightedLoss(min_frequency, max_frequency, nr_steps, frequency_0, frequency_1, 
                            low_min_weight=0.6, high_min_weight=0.6, shift_tanh=shift_tanh)


    # Define RMSE loss
    class RMSELoss(nn.Module):
        def __init__(self):
            super(RMSELoss, self).__init__()
            self.mse = nn.MSELoss()  # Use MSE under the hood

        def forward(self, y_pred, y_true):
            return torch.sqrt(self.mse(y_pred, y_true))

    # Instantiate RMSE as the criterion
    criterion = RMSELoss()

    # print(criterion)
    # criterion.plot_weights()
    # asd
    # criterion = nn.L1Loss()
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
    t = 0
    # asd
    for epoch in range(0,num_epochs):
        with TimeBlock() as timeblock:
            # with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
                # for i, batch in enumerate(tepoch):
                for i, batch in enumerate(dataloader):
                    # print(f"{current_time()} - using batch: {i}")

                    timeblock.time_block()
                    # load s batch_slice_size (s is dynamic based on duration and size of batch_song_size)
                    input_tensor_batch = batch["input"]["data"]["input"]
                    target_tensor_batch = batch["input"]["data"]["target"]

                    # shuffle slices
                    permuted_indices = torch.randperm(input_tensor_batch.shape[0])

                    # Apply the permuted indices to shuffle the data
                    input_tensor_batch = input_tensor_batch[permuted_indices]
                    target_tensor_batch = target_tensor_batch[permuted_indices]
                
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
            
                    # if epoch%10 == 0:
                    torch.save(model.state_dict(), model_save_path)
                    rob_plot.append(total_loss / len(slices_batches))
                    # Load from JSON file
                    file_lock = Lock()

                    # When saving the JSON use filelock
                    with file_lock:
                        with open('rob_plot5_target.json', 'w') as f:
                            json.dump(rob_plot, f)
                    t+=1
                    print(f'batch: {t}/{len(dataloader)} epoch: {epoch} batch_loss: {total_loss / len(slices_batches)}')

                    # print(f"{current_time()} - #USED batch: {i}")

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


