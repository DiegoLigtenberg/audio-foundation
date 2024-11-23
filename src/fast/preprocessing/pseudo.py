import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
from fast.settings.directory_settings import *
from fast.helper.time_forloop import TimeLoop
from typing import List, Union, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import torch.nn.functional as F

'''
waveform extractor
augment waveform
spectrogram extractor
padding
n-slice selection + target var setting
normalizing
'''

class AudioDataset:
    def __init__(self, 
                 dir_audio: Union[str, Path, List[str], List[Path]], 
                 dir_metadata: Union[str, Path],
                 pipeline_manager) -> None:
        """
        Initialize the dataset from directories and prepare files for processing. 
        Load metadata for the files automatically.

        Args:
            directories (Union[str, Path, List[str], List[Path]]): 
                A single directory or a list of directories containing audio files.
            metadata_file (Union[str, Path]): Path to the metadata file.
            pipeline_manager: An instance of PipelineManager to handle file processing.
        """
        self.pipeline_manager = pipeline_manager

        # Normalize `directories` to always be a list
        if isinstance(dir_audio, (str, Path)):
            dir_audio = [dir_audio]
        
        # Aggregate all file paths
        self.file_paths: List[str] = []
        for directory in dir_audio:
            directory_path = Path(directory)
            if directory_path.is_dir():
                self.file_paths.extend(sorted(directory_path.glob("*.mp3")))  # Collect and sort *.mp3 files
            else:
                raise ValueError(f"Provided path {directory} is not a valid directory.")
        
        # Convert file paths to strings for consistency
        self.file_paths = [str(path) for path in self.file_paths]
        
        # Load metadata using the internal method
        self.metadata = self._load_metadata(dir_metadata)
    
    def _load_metadata(self, metadata_file: Union[str, Path]) -> defaultdict:
        """
        Internal method to load metadata for audio files from a JSON Lines file.

        Args:
            metadata_file (Union[str, Path]): Path to the metadata file.

        Returns:
            defaultdict: A defaultdict mapping file paths to metadata dictionaries.
        """
        metadata = defaultdict(lambda: {
            "artist": "Unknown Artist",
            "album": "Unknown Album",
            "duration": 0  # Placeholder for missing metadata
        })
        
        metadata_file = Path(metadata_file) / "metadata.json"
        if not metadata_file.is_file():
            raise ValueError(f"Metadata file {metadata_file} does not exist.")

        # Load metadata from the JSON Lines file
        with open(metadata_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                file_name = entry.get("file")  # Assumes metadata has a "file" field
                if file_name:
                    metadata[file_name] = entry  # Map file name to its metadata

        return metadata

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]  # Retrieve file path
        file_name = Path(file_path).name  # Extract file name from path
        metadata = self.metadata[file_name]  # Retrieve metadata for the file
        print(metadata)
        asd
        return self.pipeline_manager.process(file_path, metadata) #metadata)  # Pass file and metadata



class PipelineManager:
    def __init__(self, processors):
        self.processors = processors  # List of processor instances, starting with AudioLoader

    def process(self, file_path, metadata):
        data = file_path
        for processor in self.processors:
            try:
                data = processor.process(data)
            except ProcessingError as e:
                print(f"Error in {e.processor}: {e}")
                # Decide to re-raise, skip, or handle the error
                raise  # Propagate the error
        return data




class ProcessingError(Exception):
    """Custom exception for pipeline processing errors."""
    def __init__(self, message, processor):
        super().__init__(message)
        self.processor = processor

class BaseProcessor(ABC):
    """
    Abstract base class for all processors in the pipeline.
    Enforces a consistent interface for processing data.
    """
    def __init__ (self):
        self.name = type(self).__name__ # initializes name to be classes subtype name

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process the input data and return the result.

        Args:
            data (Any): Input data to be processed.

        Returns:
            Any: Processed output data.
        """
        pass


class AudioLoader(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize any other parameters or configurations if needed

    def process(self, file_path):
        # Create a defaultdict with default values for each key
        data = defaultdict(lambda: None)  # Set default value as None or something else
        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            if waveform.size(1) == 0:  # Check if the audio file is empty
                raise ProcessingError(f"Empty audio file: {file_path}", processor="AudioLoader")
            
            # Populate the dictionary with relevant information
            data["waveform"] = waveform
            data["sample_rate"] = sample_rate
            data["duration"] = waveform.size(1) / sample_rate  # Calculate duration
            
            # You could also add other processed data, e.g., features, etc.
            
        except Exception as e:
            raise ProcessingError(f"Failed to load {file_path}: {e}", processor="AudioLoader")
        
        return data

# asd



# class AudioLoader:
#     def load_audio(self, file_path):
#         # 1. Load the MP3 file
#         # 2. Handle errors or missing files
#         # 3. Return the waveform and sample rate
#         pass


class BasicAudioProcessor:
    def process(self, waveform, sample_rate, target_sample_rate):
        # 1. Check if stereo and convert to mono if needed
        # 2. Resample to the target sample rate
        # 3. Return the processed waveform
        pass


class AdvancedAudioProcessor:
    def augment(self, waveform, sample_rate):
        # 1. Apply advanced audio augmentations
        # 2. Return the augmented waveform
        pass


class SpectrogramGenerator:
    def generate_spectrogram(self, waveform, sample_rate):
        # 1. Perform STFT to generate the spectrogram
        # 2. Return the spectrogram
        pass


class SpectrogramNormalizer:
    def normalize(self, spectrogram):
        # 1. Normalize the spectrogram (zero mean, unit variance)
        # 2. Return the normalized spectrogram
        pass


class SpectrogramSlicer:
    def slice(self, spectrogram, slice_duration, sample_rate):
        # 1. Calculate the number of time frames per slice
        # 2. Extract slices from the spectrogram
        # 3. Return a list of slices
        pass


def collate_fn(batch):
    waveforms = []
    max_length = max([item["waveform"].size(1) for item in batch])  # Find the max length of waveforms in the batch
    
    for item in batch:
        waveform = item["waveform"]
        # Pad the waveform to match the max length (padding zeros)
        padded_waveform = F.pad(waveform, (0, max_length - waveform.size(1)))  # Pad only along the second dimension
        waveforms.append(padded_waveform)
    
    # Stack the padded waveforms along the batch dimension
    batch_waveform = torch.stack(waveforms)
    # Calculate memory usage in bytes
    memory_bytes = batch_waveform.element_size() * batch_waveform.numel()

    # Convert to megabytes for readability
    memory_mb = memory_bytes / (1024 ** 2)
    print(memory_mb)
    
    return batch_waveform


if __name__ == "__main__":

    audio_loader = AudioLoader()
    # resample = ResampleProcessor(target_sample_rate=16000)
    # spectrogram = SpectrogramProcessor()
    # normalize = NormalizeProcessor()
    pipeline_manager = PipelineManager(processors=[audio_loader,]) # processor = [audioloader, spectrogram, normalize ]

    dataset = AudioDataset(dir_audio=[DATASET_MP3_DIR,DATASET_MP3_DIR],
                        dir_metadata = DATASET_MP3_METADATA, 
                        pipeline_manager=pipeline_manager)

    dataloader = DataLoader(
        dataset,
        batch_size=256,  # Number of MP3 files per batch
        collate_fn=collate_fn
    )

    with TimeLoop() as timeloop:
        for batch_idx, batch in enumerate(dataloader):
            timeloop.time_iteration() 
            print(f"Batch {batch_idx + 1}:")
            print(batch.shape)  # Should show (num_slices, channels, slice_width))
                
            # if batch.size(0) == 0:
            #     print("No valid slices in this batch.")

            # asd

    
    pass