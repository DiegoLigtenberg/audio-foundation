import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from fast.settings.directory_settings import *
from fast.settings.audio_settings import *
from fast.helper.time_forloop import TimeBlock
from torchaudio.transforms import GriffinLim
import random
import json

torch.manual_seed(42)  # make grififnlim deterministic # TODO this should run in some global file

class AudioDataset(Dataset):
    def __init__(self, dir_audio: list, extensions=('.mp3',), transforms=None):
        """
        Args:
            dir_audio (list): List of directories containing audio files.
            extensions (tuple): File extensions to filter audio files.
            transforms (list of callable): List of transformations to apply to the data.
        """
        self.candidate_file_paths = [f for d in dir_audio for f in Path(d).rglob('*') if f.suffix in extensions]
        self.file_paths = self.candidate_file_paths[:]  # Start with all files as valid
        self.transforms = transforms if transforms else []
        self.invalid_list = set()  # Use a set to ensure uniqueness

    def initialize_data_dict(self, file_path):
        """Helper method to initialize the data dictionary."""
        return {
            "input": {
                "data": {},  # Placeholder for waveform or transformed data
                "metadata": {"file_path": file_path},  # Include file path metadata
            },
            "target": {
                "data": {},  # Placeholder for target data
                "metadata": {},  # Placeholder for target-specific metadata
            }
        }

    def __len__(self):
        return len(self.file_paths)  # Return the count of valid files

    def __getitem__(self, idx):
        # Get the file path directly from valid file paths
        file_path = str(self.file_paths[idx])
        data_dict = self.initialize_data_dict(file_path)

        # Try loading and processing the file until a valid one is found
        while True:
            try:
                for transform in self.transforms:
                    data_dict = transform(data_dict)
                return data_dict  # If successful, return the data dict

            except InvalidAudioFileException as e:
                print(f"Skipping invalid file: {file_path} - {e}")

                # Add to invalid list if not already present
                if file_path not in self.invalid_list:
                    self.invalid_list.add(file_path)

                # Ensure file is still in file_paths before attempting to remove
                if file_path in self.file_paths:
                    self.file_paths.remove(file_path)

                # If no valid files left, raise an exception
                if len(self.file_paths) == 0:
                    raise RuntimeError("No valid audio files found.")
                
                # Move to the next valid file by updating idx
                idx = (idx + 1) % len(self.file_paths)  # Ensures idx stays within bounds
                file_path = str(self.file_paths[idx])  # Update file_path for the new idx
                data_dict = self.initialize_data_dict(file_path)  # Reinitialize the data dict

class InvalidAudioFileException(Exception):
    """Custom exception to signal an invalid audio file."""
    pass


class FileToWaveform:
    def __init__(self, input_key="file_path", output_key="raw_waveform", target_sample_rate=44100, mono=True, silence_threshold=1e-3):
        self.input_key = input_key
        self.output_key = output_key
        self.target_sample_rate = target_sample_rate
        self.mono = mono
        self.silence_threshold = silence_threshold

    def __call__(self, data_dict):
        file_path = data_dict["input"]["metadata"][self.input_key]

        # Check if the file is valid and can be loaded
        waveform, sample_rate = self.load_audio(file_path)
        if waveform is None:
            raise InvalidAudioFileException(f"Failed to load {file_path}")

        # Check if the duration is between 60 and 300 seconds
        duration = self.get_duration(waveform, sample_rate)
        if duration < 60 or duration > 300:
            raise InvalidAudioFileException(f"Invalid duration for {file_path}: {duration} seconds")

        # Check if the file is not mostly silent
        if self.is_silent(waveform):
            raise InvalidAudioFileException(f"File is mostly silent: {file_path}")

        # Resample and convert to mono if needed
        waveform = self.resample_if_needed(waveform, sample_rate)

        # Convert to mono if required
        if self.mono and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Store the waveform in the input data
        data_dict["input"]["data"][self.output_key] = waveform
        data_dict["input"]["metadata"]["sample_rate"] = self.target_sample_rate if self.target_sample_rate != sample_rate else sample_rate
        data_dict["input"]["metadata"]["duration"] = duration

        return data_dict

    def load_audio(self, file_path):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None

    def get_duration(self, waveform, sample_rate):
        num_samples = waveform.size(-1)
        duration = num_samples / sample_rate
        return duration

    def is_silent(self, waveform):
        rms = torch.sqrt(torch.mean(waveform**2))
        return rms < self.silence_threshold

    def resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform


class WaveformToLogPowerSpectrogram:
    def __init__(self, input_key="raw_waveform", output_key="log_power_spectrogram", n_fft=2046, hop_length=861, keep_original=False):
        """
        Args:
            input_key (str): Key for accessing the input waveform data.
            output_key (str): Key for storing the output log power spectrogram data.
            n_fft (int): FFT size for the spectrogram.
            hop_length (int): Hop length for the spectrogram.
            keep_original (bool): Whether to keep the original waveform in the input data.
        """
        self.input_key = input_key
        self.output_key = output_key
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.keep_original = keep_original
    
    def __call__(self, data_dict):
        """
        Transforms waveform into a log power spectrogram and updates the data dictionary.

        Args:
            data_dict (dict): Input dictionary containing waveform under ["input"]["data"][self.input_key].

        Returns:
            dict: Updated dictionary with the log power spectrogram stored under 
                  ["input"]["data"][self.output_key].
        """
        # Get the waveform from input data using the input_key
        waveform = data_dict["input"]["data"].get(self.input_key)

        # Ensure waveform is 2D (channels, time)
        if waveform.ndimension() == 1:  # Mono waveform
            waveform = waveform.unsqueeze(0)  # Convert to shape (1, time)

        # Compute the log power spectrogram
        log_power_spectrogram = self.compute_log_power_spectrogram(waveform)

        # Store the spectrogram in the dictionary using the output_key
        data_dict["input"]["data"][self.output_key] = log_power_spectrogram

        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]  # Remove the original waveform if not needed

        return data_dict

    def compute_log_power_spectrogram(self, waveform):
        """
        Computes the log power spectrogram from the waveform.

        Args:
            waveform (Tensor): Tensor of shape [channels, time].

        Returns:
            Tensor: Log power spectrogram of shape [channels, frequency_bins, time_frames].
        """
        # Create a Hann window
        window = torch.hann_window(self.n_fft, device=waveform.device)

        # Perform STFT and compute log power spectrogram
        spectrograms = []
        for channel in waveform:  # Loop over each channel (for stereo or multi-channel input)
            # Compute STFT for each channel
            stft_result = torch.stft(
                channel,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True,
            )
            # Compute magnitude spectrogram
            magnitude_spectrogram = torch.abs(stft_result) 
            # Compute power spectrogram
            power_spectrogram = magnitude_spectrogram ** 2 # highlihts dynamic range
            # Compute log power spectrogram (add small constant for numerical stability)
            log_power_spectrogram = 20 * torch.log10(power_spectrogram + 1e-8)
            spectrograms.append(log_power_spectrogram)

        # Stack the spectrograms for all channels (mono or stereo)
        stacked_log_power_spectrogram = torch.stack(spectrograms, dim=0)
        return stacked_log_power_spectrogram

class NormalizeLogPowerSpectrogram:
    def __init__(self, 
                 input_key="log_power_spectrogram", 
                 output_key="normalized_spectrogram", 
                 method="minmax", 
                 global_min_max_file="src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.json",
                 constant=150,                  
                 keep_original=False):
        """
        Args:
            input_key (str): Key for accessing the log power spectrogram data.
            output_key (str): Key for storing the normalized spectrogram data.
            method (str): Normalization method, either "minmax" or "constant".
            constant (float): Constant for the "constant" normalization method.
            keep_original (bool): Whether to retain the original spectrogram in the data dictionary.
            global_min_max_file (str): Path to the JSON file containing the global min/max values.
        """
        assert method in ["minmax", "constant"], "Invalid normalization method. Choose 'minmax' or 'constant'."
        self.input_key = input_key
        self.output_key = output_key
        self.method = method
        self.constant = constant
        self.keep_original = keep_original
        self.global_min_max_file = global_min_max_file

        # Load global min/max values from the provided JSON file
        self.global_min, self.global_max = self.load_global_min_max()

    def __call__(self, data_dict):
        """
        Normalizes the log power spectrogram and updates the data dictionary.

        Args:
            data_dict (dict): Input dictionary containing log power spectrogram under ["input"]["data"][self.input_key].

        Returns:
            dict: Updated dictionary with the normalized spectrogram stored under 
                  ["input"]["data"][self.output_key].
        """
        # Get the log power spectrogram from input data
        log_power_spectrogram = data_dict["input"]["data"].get(self.input_key)
        if log_power_spectrogram is None:
            raise KeyError(f"Key '{self.input_key}' not found in input data.")

        # Normalize the spectrogram
        if self.method == "minmax":
            normalized_spectrogram = self.minmax_normalize(log_power_spectrogram)
        elif self.method == "constant":
            normalized_spectrogram = self.constant_normalize(log_power_spectrogram)

        # Store the normalized spectrogram in the dictionary
        data_dict["input"]["data"][self.output_key] = normalized_spectrogram

        # Remove the original spectrogram if keep_original is False
        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]

        return data_dict

    def load_global_min_max(self):
        """
        Loads the global min and max values from the JSON file.

        Returns:
            tuple: Global min and max values.
        """
        try:
            with open(self.global_min_max_file, 'r') as f:
                data = json.load(f)
                return data["min_max"]
        except Exception as e:
            raise ValueError(f"Error loading global min/max values from file: {e}")

    def minmax_normalize(self, spectrogram):
        """
        Performs Min-Max normalization using global min and max values to scale the spectrogram to [0, 1].

        Args:
            spectrogram (Tensor): Log power spectrogram.

        Returns:
            Tensor: Min-Max normalized spectrogram.
        """
        return (spectrogram - self.global_min) / (self.global_max - self.global_min)

    def constant_normalize(self, spectrogram):
        """
        Normalizes the spectrogram by dividing it by a constant.

        Args:
            spectrogram (Tensor): Log power spectrogram.

        Returns:
            Tensor: Spectrogram divided by the constant.
        """
        return spectrogram / self.constant


class LogPowerSpectrogramSliceExtractor:
    def __init__(self, 
                input_key="log_power_spectrogram", 
                output_keys=["input","target"], 
                target_key="spectrogram_targets", 
                n_fft=2046, 
                hop_length=861, 
                sample_rate=44100,  
                slice_duration_sec=20, 
                padding_value=0.0,
                n_slices=5,
                keep_original=False):  # Added keep_original parameter
        """
        Args:
            n_fft (int): Number of FFT bins.
            hop_length (int): The hop length for STFT.
            sample_rate (int): The sample rate of the waveform.
            slice_duration_sec (int): Duration of each slice in seconds (default 20 seconds).
            input_key (str): Key for accessing the log power spectrogram data.
            output_key (str): Key for storing the extracted spectrogram slices.
            target_key (str): Key for storing the target timestep.
            padding_value (float): Value used for padding when the slice is shorter than the desired length.
            n_slices (int): Number of slices to extract from the spectrogram (default 5).
            keep_original (bool): Whether to keep the original spectrogram in the output (default False).
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate  
        self.slice_duration_sec = slice_duration_sec
        self.input_key = input_key
        self.output_keys = output_keys
        self.target_key = target_key
        self.padding_value = padding_value
        self.n_slices = n_slices
        self.n_slice_param = self.n_slices
        self.keep_original = keep_original  # Store keep_original parameter

        # Calculate the number of time bins for the given slice duration
        self.time_bins_per_slice = self.calculate_time_bins_per_slice()


    def calculate_time_bins_per_slice(self):
        waveform_length = self.sample_rate * self.slice_duration_sec
        waveform = torch.randn(1, waveform_length)
        time_bins = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                            window=torch.hann_window(self.n_fft), return_complex=True).size(-1)
        return time_bins
    
    def calculate_n_slices(self, song_duration_sec):
        """
        Calculate the number of slices based on the song duration.
        Args:
            song_duration_sec (float): The duration of the song in seconds.
        Returns:
            int: The calculated number of slices.
        """
        
        if self.n_slice_param == None:
            max_possible_slices = int(song_duration_sec // self.slice_duration_sec)
            # Scale slices and ensure at least one slice is returned
            return max_possible_slices
        else:
            return self.n_slice_param
        


    def __call__(self, data_dict):
        log_power_spectrogram = data_dict["input"]["data"].get(self.input_key)
        if log_power_spectrogram is None:
            raise KeyError(f"Key '{self.input_key}' not found in input data.")
        
        channels, freq_bins, song_time_bins = log_power_spectrogram.shape

        song_duration = data_dict["input"]["metadata"]["duration"]  # Get song duration in seconds
        if song_duration is None:
            raise KeyError("Song duration ('metadata.duration') not found in the input metadata.")
        
        # Calculate number of slices dynamically based on song duration
        # if self.n_slices == None:
        self.n_slices = self.calculate_n_slices(song_duration)

        # Calculate the number of time bins needed for the slice
        slice_length = self.time_bins_per_slice

        # Front padding: slice_length - 1
        front_padding = slice_length - 1
        
        # Pad the spectrogram in the front
        padded_spectrogram = torch.cat([ 
            torch.full((channels, freq_bins, front_padding), self.padding_value), 
            log_power_spectrogram
        ], dim=-1)

        padded_song_time_bins = padded_spectrogram.size(-1)

        # Calculate the maximum valid starting index for a slice
        max_start_index = padded_song_time_bins - slice_length - 1
        all_start_indices = list(range(0, max_start_index + 1))
        
        # Randomly sample `n_slices` start indices without replacement
        if self.n_slices > len(all_start_indices):
            raise ValueError(f"Cannot extract {self.n_slices} slices, not enough valid start points.")
        
        start_indices = random.sample(all_start_indices, self.n_slices)

        # Extract slices and split into inputs and targets
        input_tensor = torch.empty((self.n_slices, channels, freq_bins, slice_length - 1), dtype=padded_spectrogram.dtype)
        target_tensor = torch.empty((self.n_slices, channels, freq_bins, slice_length - 1), dtype=padded_spectrogram.dtype)
        
        for i, start_index in enumerate(start_indices):
            end_index = start_index + slice_length
            slice = padded_spectrogram[:, :, start_index:end_index]

            # Split into input (all but the last timestep) and target (all but the first timestep)
            input_tensor[i] = slice[:, :, :-1]  # shape [C, F, :T-1] 
            target_tensor[i] = slice[:, :, 1:]  # shape [C, F, 1:T]
            # sanity check print((input_tensor[i][:,:,1]  == target_tensor[i][:,:,0]).all())

        # Add the inputs and targets to the output dictionary
        for key in self.output_keys:
            if key == "input":
                data_dict["input"]["data"][key] = input_tensor
            elif key == "target":
                data_dict["input"]["data"][key] = target_tensor

        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]
        return data_dict

class LogPowerSpectrogramSliceToSong:
    def __init__(self, 
                 input_key="log_power_spectrogram", 
                 output_key="reconstructed_log_power_spectrogram", 
                 padding_value=0.0,
                 n_slices=5,           
                 keep_original=False):
        """
        Args:
            input_key (str): Key for accessing the sliced spectrogram data.
            output_key (str): Key for storing the reconstructed spectrogram.
            slice_duration_sec (int): Duration of each slice in seconds.
            padding_value (float): Padding value used for padding.
            n_slices (int): Number of slices to be used for reconstruction.
            keep_original (bool): Whether to keep the original data in the output.
            slice_length (int): Slice length in time bins.
        """
        self.input_key = input_key
        self.output_key = output_key
        self.padding_value = padding_value
        self.n_slices = n_slices
        self.keep_original = keep_original
        
        # If slice_length is not provided, it should be inferred from the input

    def __call__(self, data_dict):
        # Retrieve the input tensor (sliced spectrogram)
        input_tensor = data_dict["input"]["data"].get(self.input_key)
        slice_length = input_tensor.size(-1) 
        if input_tensor is None:
            raise KeyError(f"Key '{self.input_key}' not found in input data.")
        
        # print(input_tensor.shape)
        
        # Check if the batch size is included in the shape
        if len(input_tensor.shape) == 4:  # No batch dimension, assume batch_size = 1
            input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension at the front
            batch_size = 1
            n_slices, c, f, t = input_tensor.shape[1:]
        else:
            batch_size, n_slices, c, f, t = input_tensor.shape



        # Calculate the full time bins for the song
        full_time_bins = slice_length * self.n_slices
        
        # Create an empty tensor for the reconstructed spectrogram
        reconstructed_spectrogram = torch.full((batch_size, c, f, full_time_bins), self.padding_value, dtype=input_tensor.dtype)
        
        # Place the slices back into their respective positions in the full spectrogram
        for i in range(n_slices):
            start_idx = i * slice_length
            end_idx = start_idx + slice_length
            reconstructed_spectrogram[:, :, :, start_idx:end_idx] = input_tensor[:, i, :, :, :]

        # Add the reconstructed spectrogram to the output dictionary
        data_dict["input"]["data"][self.output_key] = reconstructed_spectrogram

        # Optionally remove the original spectrogram from the dictionary
        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]
        
        return data_dict


class DenormalizeLogPowerSpectrogram:
    def __init__(self, 
                 input_key="normalized_spectrogram", 
                 output_key="denormalized_spectrogram", 
                 method="minmax", 
                 global_min_max_file="src/fast/preprocessing/dataloader/lists_and_saved_files/min_max_values_log_power_spectrogram.json",
                 constant=150, 
                 keep_original=False):
        """
        Args:
            input_key (str): Key for accessing the normalized spectrogram data.
            output_key (str): Key for storing the denormalized spectrogram data.
            method (str): Normalization method used ("minmax" or "constant").
            constant (float): Constant used for the "constant" normalization method.
            keep_original (bool): Whether to retain the original normalized spectrogram in the data dictionary.
            global_min_max_file (str): Path to the JSON file containing the global min/max values.
        """
        assert method in ["minmax", "constant"], "Invalid normalization method. Choose 'minmax' or 'constant'."
        self.input_key = input_key
        self.output_key = output_key
        self.method = method
        self.constant = constant
        self.keep_original = keep_original
        self.global_min_max_file = global_min_max_file

        # Load global min/max values from the provided JSON file
        self.global_min, self.global_max = self.load_global_min_max()

    def __call__(self, data_dict):
        """
        Denormalizes the normalized spectrogram and updates the data dictionary.

        Args:
            data_dict (dict): Input dictionary containing normalized spectrogram under ["input"]["data"][self.input_key].

        Returns:
            dict: Updated dictionary with the denormalized spectrogram stored under 
                  ["input"]["data"][self.output_key].
        """
        # Get the normalized spectrogram from input data
        normalized_spectrogram = data_dict["input"]["data"].get(self.input_key)
        if normalized_spectrogram is None:
            raise KeyError(f"Key '{self.input_key}' not found in input data.")

        # Denormalize the spectrogram
        if self.method == "minmax":
            denormalized_spectrogram = self.minmax_denormalize(normalized_spectrogram)
        elif self.method == "constant":
            denormalized_spectrogram = self.constant_denormalize(normalized_spectrogram)

        # Store the denormalized spectrogram in the dictionary
        data_dict["input"]["data"][self.output_key] = denormalized_spectrogram

        # Remove the original normalized spectrogram if keep_original is False
        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]

        return data_dict

    def load_global_min_max(self):
        """
        Loads the global min and max values from the JSON file.

        Returns:
            tuple: Global min and max values.
        """
        try:
            with open(self.global_min_max_file, 'r') as f:
                data = json.load(f)
                return data["min_max"]
        except Exception as e:
            raise ValueError(f"Error loading global min/max values from file: {e}")

    def minmax_denormalize(self, spectrogram):
        """
        Denormalizes the spectrogram using the global min and max values.

        Args:
            spectrogram (Tensor): Normalized log power spectrogram.

        Returns:
            Tensor: Denormalized spectrogram.
        """
        return spectrogram * (self.global_max - self.global_min) + self.global_min

    def constant_denormalize(self, spectrogram):
        """
        Denormalizes the spectrogram by multiplying it by the constant.

        Args:
            spectrogram (Tensor): Normalized log power spectrogram.

        Returns:
            Tensor: Denormalized spectrogram.
        """
        return spectrogram * self.constant


class LogPowerSpectrogramToWaveform:
    def __init__(self, input_key="log_power_spectrogram", output_key="reconstructed_waveform", n_fft=2046, hop_length=861, keep_original=False):
        """
        Args:
            input_key (str): Key for accessing the input log power spectrogram data.
            output_key (str): Key for storing the output waveform data.
            n_fft (int): FFT size for the spectrogram.
            hop_length (int): Hop length for the spectrogram.
            keep_original (bool): Whether to keep the original log power spectrogram in the input data.
        """
        self.input_key = input_key
        self.output_key = output_key
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.keep_original = keep_original
        self.griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length, power=1)

    def __call__(self, data_dict):
        """
        Transforms log power spectrogram into a reconstructed waveform and updates the data dictionary.

        Args:
            data_dict (dict): Input dictionary containing log power spectrogram under ["input"]["data"][self.input_key].

        Returns:
            dict: Updated dictionary with the reconstructed waveform stored under 
                  ["input"]["data"][self.output_key].
        """
        # Get the log power spectrogram from input data using the input_key
        log_power_spectrogram = data_dict["input"]["data"].get(self.input_key)

        # Ensure log_power_spectrogram is 3D (channels, frequency_bins, time_frames)
        if log_power_spectrogram.ndimension() == 2:  # Single channel (mono)
            log_power_spectrogram = log_power_spectrogram.unsqueeze(0)  # Convert to (1, frequency_bins, time_frames)

        # Reconstruct the waveform from the log power spectrogram
        waveform = self.reconstruct_waveform(log_power_spectrogram)

        # Store the waveform in the dictionary using the output_key
        data_dict["input"]["data"][self.output_key] = waveform
        print(waveform.shape)

        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]  # Remove the original log power spectrogram if not needed

        return data_dict

    def reconstruct_waveform(self, log_power_spectrogram):
        """
        Reconstructs the waveform from the log power spectrogram using Griffin-Lim.

        Args:
            log_power_spectrogram (Tensor): Log power spectrogram of shape [channels, frequency_bins, time_frames].

        Returns:
            Tensor: Reconstructed waveform of shape [channels, time].
        """
        # Convert log power spectrogram back to power spectrogram
        power_spectrogram = torch.pow(10, log_power_spectrogram / 20)  # Reverse the log operation

        # Assume magnitude spectrogram (as we have no phase information)
        magnitude_spectrogram = torch.sqrt(power_spectrogram)  # Reverse power spectrogram

        # Apply Griffin-Lim for each channel separately
        waveforms = []
        for channel_magnitude in magnitude_spectrogram:  # Loop through each channel
            # Use Griffin-Lim to reconstruct the waveform for the current channel
            waveform_channel = self.griffin_lim(channel_magnitude)
            waveforms.append(waveform_channel)
    
        # Stack the waveforms for all channels (mono or stereo)
        stacked_waveform = torch.stack(waveforms, dim=0)  # Shape: [channels, time]
        return stacked_waveform


def collate_fn(batch, output_key, target_key):
    # Extract input and target tensors from the batch
    input_tensors = [sample["input"]["data"][output_key] for sample in batch]
    target_tensors = [sample["input"]["data"][target_key] for sample in batch]
    
    # Stack all input tensors and target tensors to create a batch of shape [batch_size, S, C, F, T]
    input_tensor_batch = torch.cat(input_tensors, dim=0)  # Shape: [batch_size, S, C, F, T]
    target_tensor_batch = torch.cat(target_tensors, dim=0)  # Shape: [batch_size, S, C, 1, T]

    # print(input_tensors.shape,input_tensor_batch.shape)
    # Reshape the batch to flatten the 20 slices into individual examples -> TODO this only needs to happen if you take n_slices is constant
    # input_tensor_batch = input_tensor_batch.view(-1, input_tensor_batch.shape[2], input_tensor_batch.shape[3], input_tensor_batch.shape[4])  # Shape: [batch_size * S, C, F, T]
    # target_tensor_batch = target_tensor_batch.view(-1, target_tensor_batch.shape[2], target_tensor_batch.shape[3], target_tensor_batch.shape[4])  # Shape: [batch_size * S, C, 1, T]

    # Return the batch in the desired format
    return {
        "input": {
            "data": {
                output_key: input_tensor_batch,
                target_key: target_tensor_batch
            }
        }
    }

if __name__ == "__main__":
    # Example DataLoader Usage
    dirs = ["audio_files/"]  # Directory containing audio files
    # dirs = [DATASET_MP3_DIR]
    transforms = [
        FileToWaveform(input_key="file_path", output_key="raw_waveform", target_sample_rate=SAMPLE_RATE, mono=MONO),
        WaveformToLogPowerSpectrogram(input_key="raw_waveform", output_key="log_power_spectrogram", n_fft=N_FFT, hop_length=HOP_LENGTH, keep_original=False),
        NormalizeLogPowerSpectrogram(input_key="log_power_spectrogram",output_key="normalized_spectrogram",method="minmax",global_min_max_file=GLOBAL_MIN_MAX_LOG_POWER_SPECTROGRAM,constant=150,keep_original=False),
        LogPowerSpectrogramSliceExtractor(input_key="normalized_spectrogram",output_keys=["input","target"],n_fft=N_FFT, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE, 
                                        slice_duration_sec=CHUNK_DURATION,n_slices=20,keep_original=False),

        # LogPowerSpectrogramSliceToSong(input_key="input",output_key="reconstructed_log_power_spectrogram", n_slices=20,keep_original=False),
        # DenormalizeLogPowerSpectrogram(input_key="reconstructed_log_power_spectrogram",output_key="denormalized_log_power_spectrogram",method="minmax",global_min_max_file=GLOBAL_MIN_MAX_LOG_POWER_SPECTROGRAM,constant=150,keep_original=False),
        # LogPowerSpectrogramToWaveform(input_key="denormalized_log_power_spectrogram", output_key="reconstructed_waveform", n_fft=N_FFT, hop_length=HOP_LENGTH,keep_original=False)
    ]  # Add all transforms here

    output_key = 'input'  # Replace with your actual output key
    target_key = 'target'  # Replace with your actual target key

    dataset = AudioDataset(dir_audio=dirs, extensions=('.mp3',), transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, output_key, target_key)  # Pass keys explicitly
    )

    min_val = float('inf')  # Start with the largest possible value for min
    max_val = float('-inf')  # Start with the smallest possible value for max

    # Set seed for reproducibility
    # random.seed(42)  # Python random
    # np.random.seed(42)  # NumPy random


    # Example loop over DataLoader
    with TimeBlock() as timeblock:
        for i, batch in enumerate(dataloader):
            timeblock.time_block()
            # batch contains "data" with input and target as needed
            # input_data = batch["input"]["data"]["denormalized_log_power_spectrogram"]
            # print(input_data.shape)
            # asd
            input_data = batch["input"]["data"]
            spectrogram_input = input_data["input"]
            spectrogram_target = input_data["target"]
            
            spectrogram_input = spectrogram_input[0]
            print(spectrogram_input.shape)
            print(spectrogram_target.shape)
            torch.save(spectrogram_input, 'src/fast/model/model_weights/spectrogram_input.pth')
            asd

            
            # asd
            
            # spectrogram = batch["input"]["data"]["target"]
            # print(spectrogram.shape)
            # asd
            
            # spectrogram = batch["input"]["data"]["reconstructed_waveform"]
            # print(spectrogram.shape)
            # # Get the min and max of the spectrogram for the current batch
            # batch_min = spectrogram.min().item()  # .item() to get the scalar value
            # batch_max = spectrogram.max().item()  # .item() to get the scalar value

            # # Update the overall min and max values across all batches
            # min_val = min(min_val, batch_min)
            # max_val = max(max_val, batch_max)
            # # print(min_val,max_val)
            # print(spectrogram.shape)
            # spectrogram = spectrogram.squeeze(0).squeeze(0)
            # print(spectrogram.shape)
            # # print(batch["input"]["metadata"]["file_path"])
            # torchaudio.save("reconstructed_waveform2.wav", spectrogram, SAMPLE_RATE)  # Use the appropriate sample rate

            # asd
            # if i >10000:
                    # break
            # asd
            # print(i,end="\r")
                # print(f'{batch["input"]["metadata"]["file_path"]} {i}-{min_val}-{max_val}')

        # print("Spectrogram shape:", spectrogram.shape)

    # import pickle
    # print(len(dataset.invalid_list))
    # with open('invalid_list.pkl', 'wb') as f:
    #     pickle.dump(dataset.invalid_list, f)
        

    # Print the overall min and max values after the loop
    # print(f"Overall Min Value: {min_val}")
    # print(f"Overall Max Value: {max_val}") 