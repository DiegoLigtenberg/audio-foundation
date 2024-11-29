
import torchaudio
from itertools import islice

N_FFT = 2046  # FFT size
HOP_LENGTH = 861  # Overlap size
SAMPLE_RATE = 44100  # Desired sample rate
CHUNK_DURATION = 20  # Duration in seconds to process
MONO = False

# SPECTROGRAM_SAVE_DIR = "train_spectrogram"
# MIN_MAX_VALUES_SAVE_DIR = "F:/Thesis/test"
# TEMP_INFERENCE_SAVE_DIR = "temp_inference"       # do not change
# print(DATASET_MP3_DIR)


# def validate_mp3_files(directory, limit=None):
#     invalid_files = []  # Collect invalid files
#     # Use islice to limit to the first 'limit' files
#     for file in islice(directory.rglob("*.mp3"), limit):
#         # Check if the file has the '.mp3' extension (just to be sure)
#         if file.suffix.lower() != '.mp3':
#             invalid_files.append((file, "Not an MP3 file"))
            
#     return invalid_files
# # Run validation
# invalid_files = validate_mp3_files(DATASET_MP3_DIR)

# # Print results
# if invalid_files:
#     print("Invalid or corrupted MP3 files:")
#     for file, error in invalid_files:
#         print(f" - {file}: {error}")
# else:
#     print("All MP3 files are valid!")