import torch
from fast.settings.audio_settings import *

''' 
1- load a file
2- extract segments
3- augment
4- pad the signal (if necessary)
5- extracting log spectrogram from signal
6- normalize spectrogram
7- save the normalized spectrogram

Preprocessing Pipeline
'''

class Loader():
    '''Load is responsible for loading an audio file.'''
    def __init__(self,sample_rate,mono):
        pass


        
# def validate_mp3_files(directory, limit=None):
#     invalid_files = []  # Collect invalid files
#     # Use islice to limit to the first 'limit' files
#     for file in islice(directory.rglob("*.mp3"), limit):
#         # Check if the file has the '.mp3' extension (just to be sure)
#         if file.suffix.lower() != '.mp3':
#             invalid_files.append((file, "Not an MP3 file"))
            

    def load_from_path(self):
        pass

    def reset_input_counter(self):
        pass


class Padder:
    '''responsible to apply zero padding to an array - works for stereo'''

    def __init__(self,mode="constant"):
        self.mode = mode

        
    def left_pad(self,array,num_missing_items):
        pass

    def right_pad(self,array,num_missing_items):
        pass


class LogSpectroGramExtractor():
    def __init__(self,n_fft):
        self.n_fft = n_fft
        self.has_hop = False # makes sure that we only calculate hop length once      

    def extract_stft(self,signal):
        pass

    def plot_spectrogram(self,spectrogram):
        '''power scale (^2) for better visualisation (more contrast) -> amp-to-db '''
        pass

class MinMaxNormalizer:
    def __init__(self,min_val,max_val):
        pass


class PreprocessingPipeline:
    def __init__(self,chunk_duration):
        pass




