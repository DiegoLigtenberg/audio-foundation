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
import os
# multiprocessing stuff
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True) # for multiprocessing
from functools import partial

from datetime import datetime
def current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# torch.manual_seed(42)  # make grififnlim deterministic # TODO this should run in some global file
# random.seed(42)

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
                # worker_info = torch.utils.data.get_worker_info()
                # if idx % 32 == 0:
                #     print(f"{current_time()} - loading idx {idx//32:02} - worker {worker_info.id}")
                for transform in self.transforms:
                    data_dict = transform(data_dict)
                # if (idx+1) % 32 == 0:
                #     print(f"{current_time()} - #LOADED idx {(idx-31)//32:02} - worker {worker_info.id}")
                return data_dict  # If successful, return the data dict

            except InvalidAudioFileException as e:
                print(f"Skipping invalid file: {file_path} - {e}",end="\r")

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

        # num_samples = data_dict["input"]["metadata"]["sample_rate"]*10
        # waveform = waveform[:, :num_samples]

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
                only_half_frequency = True, # only process on half 0-10k frequency range obtained from NFFT
                random_input_noise = 0.05, # range from 0 to 0.1 (we pick random so half of value is happening on avereage)
                max_mask_size=0.5, # how large the mask is that we set to 0 of time tokens compared to length of total time duration
                mask_probability=0.5, # percentage probability of masking part of input sequence
                volume_reduction = 0.3,
                n_slices=5,
                keep_original=False):  # Added keep_original parameter
        """
        Args:
            input_key (str): Key for accessing the log power spectrogram data.
            output_keys (list): Keys for storing the processed spectrogram slices (default ["input", "target"]).
            target_key (str): Key for storing the target timestep or related spectrogram data.
            n_fft (int): Number of FFT bins for STFT (default 2046).
            hop_length (int): Hop length for STFT (default 861).
            sample_rate (int): Sampling rate of the waveform (default 44100 Hz).
            slice_duration_sec (float): Duration of each slice in seconds (default 20 seconds).
            padding_value (float): Value for padding when a slice is shorter than the desired length (default 0.0).
            only_half_frequency (bool): Whether to process only the lower half of the frequency range (default True).
            random_input_noise (float): Maximum range of random noise to add to the input; average is half this value (default 0.05).
            max_mask_size (float): Maximum mask size relative to the time tokens (default 0.5).
            mask_probability (float): Probability of masking a part of the input sequence (default 0.5).    
            volume_reduction (float): Probability of reducing volume (gausian) in the song of the input sequence (default 0.3).         
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
        self.only_half_frequency = only_half_frequency
        self.random_input_noise = random_input_noise
        self.max_mask_size = max_mask_size
        self.mask_probability = mask_probability
        self.volume_reduction = volume_reduction
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

        # Pad the spectrogram in the front and only extact 10k of the whole frequence
        if self.only_half_frequency:
            
            freq_bins = freq_bins//2
            padded_spectrogram = torch.cat([ 
                torch.full((channels, freq_bins, front_padding), self.padding_value), 
                log_power_spectrogram[:,:freq_bins,:] # only extract 10k of the whole frequency
            ], dim=-1)
        # Pad the spectrogram in the front 
        else:
            padded_spectrogram = torch.cat([ 
                torch.full((channels, freq_bins, front_padding), self.padding_value), 
                log_power_spectrogram[:,:freq_bins,:] # only extract 10k of the whole frequency
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
        target_tensor = torch.empty((self.n_slices, channels, freq_bins, 1), dtype=padded_spectrogram.dtype)
        
        for i, start_index in enumerate(start_indices):
            end_index = start_index + slice_length
            slice = padded_spectrogram[:, :, start_index:end_index]
            # Split into input (all but the last timestep) and target (all but the first timestep)
            input_tensor[i] = slice[:, :, :-1]  # shape [C, F, :T-1] 
            if self.random_input_noise > 0:
                input_tensor[i] = self.add_random_noise(input_tensor[i],self.random_input_noise)
            if self.mask_probability > 0:
                input_tensor[i] = self.apply_random_mask(tensor=input_tensor[i], max_mask_size=self.max_mask_size, mask_probability=self.mask_probability)
            if self.volume_reduction > 0:
                input_tensor[i] = self.apply_gaussian_volume_reduction(input_tensor[i], volume_reduction_factor=self.volume_reduction)


            
            target_tensor[i] = slice[:, :, -1:] # shape [C, F, 1:T]
            

            # sanity check print((input_tensor[i][:,:,1]  == target_tensor[i][:,:,0]).all())
            # print("Sanity check (target):", (target_tensor[i] == slice[:, :, -1:]).all())
            # asd
        
        # Add the inputs and targets to the output dictionary
        for key in self.output_keys:
            if key == "input":
                data_dict["input"]["data"][key] = input_tensor
            elif key == "target":
                data_dict["input"]["data"][key] = target_tensor

        if not self.keep_original:
            del data_dict["input"]["data"][self.input_key]
        return data_dict
    

    def add_random_noise(self, tensor, noise_scale=0.05):
        """
        Adds random Gaussian noise to the input tensor, but only to non-zero values.
        
        Args:
            tensor (torch.Tensor): The tensor to which noise is added.
            noise_scale (float): The maximum scale of the random noise.
            
        Returns:
            torch.Tensor: The tensor with added noise.
        """
        # Generate a random noise value between 0 and noise_scale
        noise_scale = random.uniform(0, noise_scale)

        # Create a mask of non-zero values in the tensor
        non_zero_mask = tensor != 0

        # Generate Gaussian noise only for non-zero elements
        noise = noise_scale * torch.randn_like(tensor)

        # Apply noise to only the non-zero values
        tensor[non_zero_mask] += noise[non_zero_mask]
        
        return tensor
    
    def apply_random_mask(self, tensor, max_mask_size, mask_probability):
        """
        Randomly masks a portion of the tensor along the time axis with a given probability.

        Args:
            tensor (torch.Tensor): The tensor to which the mask is applied, shape [n_slices, channels, freq_bins, time_length].
            max_mask_size (float): The maximum length (as percentage) of the input_tensor's time dimension.
            mask_probability (float): Probability of applying the mask (value between 0 and 1).
            
        Returns:
            torch.Tensor: The tensor with a random portion masked or unchanged if the mask is not applied.
        """
        # Decide whether to apply the mask based on mask_probability
        if torch.rand(1).item() > mask_probability:
            return tensor
        # Extract the time axis length
        time_axis_length = tensor.shape[-1]  # Corresponds to slice_length - 1
        assert max_mask_size < 0.75, "can't mask more than 75 percent of the spectrogram"
        max_mask_size = int(max_mask_size * time_axis_length)

        # Randomly decide the length of the mask (at least 1, at most max_mask_size or time_axis_length)
        mask_length = torch.randint(1, min(max_mask_size, time_axis_length) + 1, (1,)).item()

        # Randomly decide the starting position of the mask
        mask_start = torch.randint(0, time_axis_length - mask_length + 1, (1,)).item()

        # Calculate the ending position of the mask
        mask_end = mask_start + mask_length
        
        # Apply mask along the last dimension (time axis)
        tensor[:, :, mask_start:mask_end] = 0
        
        return tensor
    
    def apply_gaussian_volume_reduction(self, tensor, volume_reduction_factor=0.3):
        """
        Applies random Gaussian noise-based volume reduction to each element in the tensor.

        Args:
            tensor (torch.Tensor): The input tensor where volume reduction will be applied.
            volume_reduction_factor (float): The scale of the reduction (Gaussian noise standard deviation).

        Returns:
            torch.Tensor: The tensor with volume reduced by a Gaussian noise factor.
        """
        if torch.rand(1).item() > 0.25: # in 25% of cases we reduce volume by up to  30%
            return tensor
        # Generate Gaussian noise for each element in the tensor
        noise = torch.randn_like(tensor) * volume_reduction_factor  # Gaussian noise

        # Apply the noise factor to each element in the tensor (potentially reducing the volume)
        reduced_tensor = tensor * (1 - noise)  # Decrease volume based on negative noise

        return reduced_tensor


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
                                        slice_duration_sec=CHUNK_DURATION,n_slices=None,only_half_frequency=True,random_input_noise=0.01,max_mask_size=0.2,volume_reduction=0.1,mask_probability=0.05,keep_original=False),

        # LogPowerSpectrogramSliceToSong(input_key="input",output_key="reconstructed_log_power_spectrogram", n_slices=20,keep_original=False),
        # DenormalizeLogPowerSpectrogram(input_key="reconstructed_log_power_spectrogram",output_key="denormalized_log_power_spectrogram",method="minmax",global_min_max_file=GLOBAL_MIN_MAX_LOG_POWER_SPECTROGRAM,constant=150,keep_original=False),
        # LogPowerSpectrogramToWaveform(input_key="denormalized_log_power_spectrogram", output_key="reconstructed_waveform", n_fft=N_FFT, hop_length=HOP_LENGTH,keep_original=False)
    ]  # Add all transforms here

    output_key = 'input'  # Replace with your actual output key
    target_key = 'target'  # Replace with your actual target key

    max_workers = os.cpu_count() if os.cpu_count() is not None else 1
    collate_fn_with_keys = partial(collate_fn, output_key=output_key, target_key=target_key) # for multiprocessing
    dataset = AudioDataset(dir_audio=dirs, extensions=('.mp3',), transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_with_keys,  # Pass keys explicitly # used to be lambda batch: collate_fn(batch, output_key, target_key) for single processing
        num_workers=max_workers
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
            # torch.save(spectrogram_input, 'src/fast/model/model_weights/spectrogram_input.pth')
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