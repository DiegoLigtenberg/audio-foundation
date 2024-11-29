from fast.model.model_channel import *
from fast.preprocessing.dataloader.dataloader import *
from fast.settings.directory_settings import *
MONO = True
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = ["audio_files/"]  # Directory containing audio files
    # dirs = [DATASET_MP3_DIR]
    transforms = [
        FileToWaveform(input_key="file_path", output_key="raw_waveform", target_sample_rate=SAMPLE_RATE, mono=MONO),
        WaveformToLogPowerSpectrogram(input_key="raw_waveform", output_key="log_power_spectrogram", n_fft=N_FFT, hop_length=HOP_LENGTH, keep_original=False),
        NormalizeLogPowerSpectrogram(input_key="log_power_spectrogram",output_key="normalized_spectrogram",method="minmax",global_min_max_file=GLOBAL_MIN_MAX_LOG_POWER_SPECTROGRAM,constant=150,keep_original=False),
        LogPowerSpectrogramSliceExtractor(input_key="normalized_spectrogram",output_keys=["input","target"],n_fft=N_FFT, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE, 
                                        slice_duration_sec=CHUNK_DURATION,n_slices=64,keep_original=False), # n _ slices is how many slices we sampel per song
    ] 

    output_key = 'input'  # Replace with your actual output key
    target_key = 'target'  # Replace with your actual target key

    dataset = AudioDataset(dir_audio=dirs, extensions=('.mp3',), transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=16, # hw many songs we load of n_slices
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, output_key, target_key),  # Pass keys explicitly
    )

    # Hyperparameters
    seq_len = 1024  # Time steps # potentially make this 512
    freq_size = 1024  # Frequency bins
    num_samples = 1000
    batch_slice_size = 64    #gpt3 ->4096
    num_heads = 32           #gpt3 -> 96
    num_decoder_layers = 16  #gpt3 -> 96
    dim_feedforward = 8192   #gpt3 -> 48768
    learning_rate = 1e-4
    num_epochs = 2
    num_channels = 1 if MONO else 2

    # Model, Loss, and Optimizer
    model = TransformerModel(freq_size, num_channels, seq_len, num_heads, num_decoder_layers, dim_feedforward).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Count the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # for epoch in num_epochs:
    model.train()
    total_loss = 0
    # Example loop over DataLoader
    with TimeBlock() as timeblock:
        for i, batch in enumerate(dataloader):
            timeblock.time_block()
            spectrogram = batch["input"]["data"]["input"]
            target = batch["input"]["data"]["target"]
            # spectrogram_size = spectrogram.element_size() * spectrogram.numel()
            # target_size = target.element_size() * target.numel()
            # target_size += spectrogram_size


            # print(f"Total size: {target_size / (1024**2)} MB")

            slices_batches = torch.split(spectrogram, batch_slice_size)
            slices_batches_target = torch.split(target,batch_slice_size)
        
            for batch_index, (slice_batch, slice_batch_target) in enumerate(zip(slices_batches, slices_batches_target)):
                slice_batch = slice_batch.to(device)
                slice_batch_target = slice_batch_target.to(device)
                print(slice_batch.shape)
                asd
                optimizer.zero_grad()
                output = model(slice_batch,slice_batch_target)
                loss = criterion(output, slice_batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                print(f'{batch_index} - {loss.item()}')

                # tepoch.set_postfix(loss=loss.item())
            # Clear memory if needed
            # del spectrogram, target, output  # Explicitly delete the variables




             
            # with tqdm(dataloader, unit="batch") as tepoch:
            #     tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            #     for spectrogram, target in tepoch:
                    # Move data to GPU


