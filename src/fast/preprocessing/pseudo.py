import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
from fast.settings.directory_settings import *
from fast.settings.audio_settings import *
from fast.helper.time_forloop import TimeLoop, TimeBlock
from fast.helper.helper_class import defaultdict
from typing import List, Union, Any
from abc import ABC, abstractmethod
import json
import itertools
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
        infinite_defaultdict = lambda: defaultdict(infinite_defaultdict)
        # Define the main defaultdict structure
        self.data_dict = defaultdict(lambda: {
            "metadata": infinite_defaultdict(),  # Infinite nesting for metadata
            "technical_info": infinite_defaultdict()})

        # Normalize `directories` to always be a list
        if isinstance(dir_audio, (str, Path)):
            dir_audio = [dir_audio]
        
        # Aggregate all file paths and initialize the defaultdict
        for directory in dir_audio:
            directory_path = Path(directory)
            if directory_path.is_dir():
                for file_path in sorted(directory_path.glob("*.mp3")):  # Collect and sort *.mp3 files
                    str_path = file_path.as_posix()
                    self.data_dict[str_path]  # Initialize entry for each file
            else:
                raise ValueError(f"Provided path {directory} is not a valid directory.")
        # Load metadata using the internal method
        self.metadata = self._load_metadata(dir_metadata)
    
    def _load_metadata(self, metadata_file: Union[str, Path]) -> None:
        """
        Internal method to load metadata for audio files from a JSON Lines file.

        Args:
            metadata_file (Union[str, Path]): Path to the metadata file.
        """
        metadata_file = Path(metadata_file) / "metadata.json"
        if not metadata_file.is_file():
            raise ValueError(f"Metadata file {metadata_file} does not exist.")

        # Load metadata from the JSON Lines file
        with open(metadata_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                file_name = entry.get("file")  # Assumes metadata has a "file" field
                if file_name in self.data_dict:
                    self.data_dict[file_name]["metadata"]["context_based"]["genre_code"] = entry.get("group")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        '''ideally we want to just pass data_dict containing ["input"] and ["target] towards the model's dataloader'''
        # itertools is efficient way to iterate of all data_dict filenames (the keys)
        file_path = next(itertools.islice(self.data_dict.keys(), idx,idx+1)) # slice takes iterable, start, stop
        self.pipeline_manager.process(file_path, self.data_dict)
        return self.data_dict[file_path] # Process the data element using the pipeline manager


class PipelineManager:
    def __init__(self, processors):
        self.processors = processors  # List of processor instances, starting with AudioLoader

    def process(self, file_path, data_dict):
        for processor in self.processors:
            try:
                processor.process(file_path, data_dict)
            except ProcessingError as e:
                print(f"Error in {e.processor}: {e}")
                # Decide to re-raise, skip, or handle the error
                raise  # Propagate the error


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
    def process(self, file_path: Path) -> Any:
        """
        Process the input data and return the result.

        Args:
            file_path (str): The file path to fetch data from `data_dict`.
            data (Any): The current data being processed, which might be a string (file_path) or intermediate results.

        Returns:
            Any: Processed output data.
        """
        pass


class AudioValidationProcessor(BaseProcessor):
    '''waveforms in torch have [channel, samples] '''
    def __init__(self):
        super().__init__()
        # Initialize any other parameters or configurations if needed

        self.min_duration = 60
        self.max_duration = 300

    def process(self, file_path:str, data_dict: str) -> None:
        """
        Process the file path and load the corresponding audio waveform.

        Args:
            file_path (str): The file path to fetch the audio data from `data_dict`.
            data (str): The current data (file_path).
        """
        try:
            # with TimeBlock() as timeblock:
                # timeblock.time_block()
                waveform, sample_rate = torchaudio.load(file_path)
                data_dict[file_path]["metadata"]["content_based"]["waveform"] = waveform
                data_dict[file_path]["metadata"]["content_based"]["sample_rate"] = sample_rate
        except Exception as e:
            raise ProcessingError(f"Processing error in {self.name}. Failed to load {file_path}: {e}", processor=self.name)        
        return None

class WaveformResampleProcessor(BaseProcessor): 
    '''processor that resamples audio and converts to mono''' #TODO split this into 3 classes, resample, mono. validate
    def __init__(self,target_sample_rate):
        super().__init__() # first call super's init (to set self.name)
        self.min_duration = 60
        self.max_duration = 300
        self.target_sample_rate = target_sample_rate
        required_fields = ["waveform","sample_rate"] # TODO

    def process(self, file_path:str, data_dict: str) -> torch.Tensor:
        '''apply TODO of required fields here'''
        if not data_dict[file_path]["metadata"]["content_based"].contains("waveform"):
            raise ProcessingError(f"Processing error in {self.name}. 'waveform' not found for {file_path}",processor=self.name)
        if not data_dict[file_path]["metadata"]["content_based"].contains("sample_rate"):
            raise ProcessingError(f"Processing error in {self.name}. 'samplerate' not found for {file_path}",processor=self.name)
        
        waveform = data_dict[file_path]["metadata"]["content_based"]["waveform"] # TODO this waveform key could be a variable key hwn initializing this class
        sample_rate = data_dict[file_path]["metadata"]["content_based"]["sample_rate"] # TODO this sample_rate key could be a variable key hwn initializing this class
        self._validate_waveform(waveform,sample_rate)
        waveform = self.resample(waveform,sample_rate,self.target_sample_rate)
        waveform = self.waveform_to_mono(waveform)
        duration_seconds = waveform.size(1) // self.target_sample_rate  

        # if the above functions gave no errors, we can set and save the values inside the dict
        data_dict[file_path]["metadata"]["content_based"]["waveform"] = waveform
        data_dict[file_path]["metadata"]["content_based"]["sample_rate"] = self.target_sample_rate
        data_dict[file_path]["metadata"]["content_based"]["waveform_duration"] = duration_seconds

        return None


    def resample(self, waveform: torch.Tensor, sample_rate: int, target_sample_rate) -> torch.Tensor:
        """
        Resample the waveform to the target sample rate.
        
        Args:
            waveform (torch.Tensor): The waveform to resample.
            sample_rate (int): The original sample rate of the waveform.
            
        Returns:
            torch.Tensor: The resampled waveform.
        """
        if sample_rate != SAMPLE_RATE:  # Only resample if necessary
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        return waveform

    def waveform_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a stereo waveform to mono by averaging the channels.
        
        Args:
            waveform (torch.Tensor): The waveform to convert.
            
        Returns:
            torch.Tensor: The mono waveform.
        """
        if waveform.size(0) > 1:  # Check if stereo
            waveform = waveform.mean(dim=0, keepdim=True)  # Average the stereo channels to mono
        return waveform

    
    def _validate_waveform(self,waveform: torch.Tensor, sample_rate: torch.Tensor):
        # Check if the waveform is empty
        if waveform.size(1) == 0:  # If the waveform has no samples
            raise ProcessingError(f"Processing error in {self.name}. The waveform is empty.", processor=self.name)
    
        # Check if the waveform has at least one channel
        if waveform.size(0) == 0:  # No channels
            raise ProcessingError(f"Processing error in {self.name}. The waveform has no channels.", processor=self.name)
        
        # Check if the duration of the waveform (in seconds) is between min and max duration
        duration_seconds = waveform.size(1) / sample_rate  
        if not (self.min_duration < duration_seconds < self.max_duration):
            raise ProcessingError(f"Processing error in {self.name}. The waveform duration is not between {self.min_duration} and {self.max_duration} seconds.", processor=self.name)


class SpectrogramGenerator(BaseProcessor):
    '''Processor that generates spectrograms from audio waveforms'''

    def __init__(self, mono: bool, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()  # Initialize the base class
        self.mono = mono
        self.n_fft = n_fft  # FFT size for STFT
        self.hop_length = hop_length  # Overlap size for STFT
        self.required_fields = ["waveform"]

    def process(self, file_path: str, data_dict: dict) -> None:
        '''Generates spectrograms based on mono or stereo configuration'''

        # Validate required fields under content-based metadata
        content_based = data_dict[file_path]["metadata"]["content_based"]
        for field in self.required_fields:
            if field not in content_based:
                raise ProcessingError(
                    f"Processing error in {self.name}. '{field}' not found for {file_path}",
                    processor=self.name,
                )

        # Retrieve waveform from content-based metadata
        waveform = content_based["waveform"]

        # Convert to mono if required
        if self.mono and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Generate and set spectrogram(s) directly in content-based metadata
        self._generate_spectrogram(waveform, content_based)

    def _generate_spectrogram(self, waveform: torch.Tensor, content_based: dict) -> None:
        '''Generate spectrogram(s) and set them in content-based metadata'''
        window = torch.hann_window(self.n_fft)  # Window for STFT

        spectrograms = []
        for channel in waveform:
            # Compute STFT
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
            spectrograms.append(magnitude_spectrogram)

        # Set spectrogram(s) in content-based metadata
        if not self.mono and waveform.size(0) == 2:
            # Combine stereo spectrograms
            combined_spectrogram = torch.cat(spectrograms, dim=0)
            content_based["spectrogram"] = combined_spectrogram
        else:
            # Use the first spectrogram for mono or single-channel data
            content_based["spectrogram"] = spectrograms[0]
        print(spectrograms[0].shape)



        # 1. Perform STFT to generate the spectrogram
        # 2. Return the spectrogram



# class AudioLoader:
#     def load_audio(self, file_path):
#         # 1. Load the MP3 file
#         # 2. Handle errors or missing files
#         # 3. Return the waveform and sample rate
#         pass


# class BasicAudioProcessor:
#     def process(self, waveform, sample_rate, target_sample_rate):
#         # 1. Check if stereo and convert to mono if needed
#         # 2. Resample to the target sample rate
#         # 3. Return the processed waveform
#         pass


# class AdvancedAudioProcessor:
#     def augment(self, waveform, sample_rate):
#         # 1. Apply advanced audio augmentations
#         # 2. Return the augmented waveform
#         pass


# class SpectrogramGenerator:
#     def generate_spectrogram(self, waveform, sample_rate):
#         # 1. Perform STFT to generate the spectrogram
#         # 2. Return the spectrogram
#         pass

class SpectrogramPadder:
    def pad(self):
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
    var  = batch[0]["metadata"]["content_based"]["waveform"]

    max_length =  var.size(1) 
    for item in batch:
        item = item["metadata"]["content_based"]["waveform"]
        F.pad(item, (0, max_length - item.size(1)))
        
        # item = item["waveform"]
        # Pad the waveform to match the max length (padding zeros)
        padded_waveform = F.pad(item, (0, max_length - item.size(1)))  # Pad only along the second dimension
        waveforms.append(padded_waveform)
    
    # Stack the padded waveforms along the batch dimension
    batch_waveform = torch.stack(waveforms)
    
    return batch_waveform


if __name__ == "__main__":

    audio_validation_processor = AudioValidationProcessor()
    basic_waveform_processor = WaveformResampleProcessor(target_sample_rate=SAMPLE_RATE)
    spectrogram_generator = SpectrogramGenerator(mono=MONO,n_fft=N_FFT,hop_length=HOP_LENGTH)
    # resample = ResampleProcessor(target_sample_rate=16000)
    # spectrogram = SpectrogramProcessor()
    # normalize = NormalizeProcessor()
    pipeline_manager = PipelineManager(processors=[audio_validation_processor,basic_waveform_processor,spectrogram_generator]) # processor = [audioloader, spectrogram, normalize ]

    dataset = AudioDataset(dir_audio=[DATASET_MP3_DIR,DATASET_MP3_DIR],
                        dir_metadata = DATASET_MP3_METADATA, 
                        pipeline_manager=pipeline_manager)

    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Number of MP3 files per batch
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