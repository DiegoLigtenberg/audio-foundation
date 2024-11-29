import os
import csv
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from yt_dlp import YoutubeDL
import multiprocessing
import re
from tqdm import tqdm
import keyboard 
import threading
from datetime import datetime
import glob
import sys
from fast.settings.directory_settings import *

'''
we want outcome sample of everything to be 44.1khz (44100), this happens after the mp3 compression of 192 bitrate, just so every song is eventually consistent.
'''
FFMPEG_YDL_DIR = r"C:\yt-dlp\ffmpeg-2024-11-06-git-4047b887fc-full_build\bin\ffmpeg.exe'"

# Configuration options for yt-dlp
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': DATASET_MP3_DIR / '%(id)s.%(ext)s',
    'ffmpeg_location': FFMPEG_YDL_DIR,
    'ignoreerrors': 'only_download',
    'quiet': True,
    'no_warnings': True,  # Suppress warnings
    'logtostderr': False,  # Do not log to stderr

    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        # 'preferredquality': '192', # number of bits used to represent samplerate and bit depth (to compression)  https://www.waveroom.com/blog/bit-rate-vs-sample-rate-vs-bit-depth/
        'nopostoverwrites': False,
    }],
    'postprocessor_args': [
        '-ar', '44100', 
        '-ac', '2',
    ],
    'concurrent_fragment_downloads': 20,  # Download 4 parts concurrently
}

# Load the current start number from a file
def load_start_number(file_path):
    """Load the saved start number from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    return 1  # Default to 1 if the file doesn't exist

# Save the current start number to a file
def save_start_number(file_path, start_number):
    """Save the current start number to a file."""
    with open(file_path, 'w') as file:
        file.write(str(start_number))


# Listen for the Alt + F8 key combination
def listen_for_exit():
    """Listen for Alt + F8 key press to stop the program."""
    while True:
        if keyboard.is_pressed('alt+f8'):  # Check for Alt + F8 key press
            print("Exit key pressed. Terminating program.")
            os._exit(0)  # Immediately exit the program
        time.sleep(0.1)  # Sleep for a short time to avoid CPU overload


# Get the URL index from the CSV file that corresponds to the start_number
def get_csv_file_to_start_from(folder_path, start_number):
    """Determine which CSV file to start processing from and return its index in the sorted folder list."""
    sorted_files = sorted(os.listdir(folder_path))
    total_urls_processed = 0
    
    for index, _csv in enumerate(sorted_files):
        if _csv.endswith('.csv'):
            csv_path = os.path.join(folder_path, _csv)
            with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
                url_count = sum(1 for row in file)
                total_urls_processed += url_count
                if total_urls_processed >= start_number:
                    return index, 0  # Return the index and offset 0 for start_number
    return None, 0  # In case no file is found, return None

# Function to create the log file name based on the current time (YYMMDDHHMM)
def get_log_filename():
    return datetime.now().strftime("%y%m%d%H%M") + "_log.txt"

# Function to write to the log safely in a multiprocessing context
def write_to_log(log_file_path,log_queue):
    with open(log_file_path, 'a', newline='', encoding='utf-8') as log_file:
        while True:
            log_entry = log_queue.get()  # Get log entries from the queue
            if log_entry == 'DONE':
                break  # Stop when 'DONE' is received
            log_file.write(log_entry)
            log_file.flush()  # Ensure it's written immediately

# Cleanup function for temp folder
def cleanup_temp_folder(output_dir):
    """Cleans up files in the dataset_mp3 directory that do not match the 7-digit filename pattern."""
    pattern = re.compile(r"^\d{7}\.mp3$")  # Matches filenames like "0000001.mp3"
    
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path) and not pattern.match(filename):
            try:
                os.remove(file_path)
                # print(f"Removed {file_path}")
            except Exception as e:
                # print(f"Error removing {file_path}: {e}",end="\r")
                pass

def cleanup_old_log_files(log_dir, max_files=5):
    # List all .txt files in the directory and sort them by modification time (oldest first)
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.txt")), key=os.path.getmtime)
    
    # If the number of log files exceeds `max_files`, delete the oldest ones
    while len(log_files) > max_files:
        # Delete the oldest log file
        os.remove(log_files[0])
        log_files.pop(0)
       

def download_single_video(url, output_dir, condition_counts, processed_urls, start_number, lock, start_number_file, increment_counter):
    """Downloads a single video and returns the final MP3 filename."""
    try:
        # Check if URL was already processed
        if url in processed_urls:
            # print(f"Skipping {url}: Already processed.", end="\r")
            with lock:
                condition_counts['skipping_already_processed'] += 1
            return None
        
        # Save the updated start_number immediately after each URL is processed
        save_start_number(start_number_file, start_number.value)

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False) # function: extract_info -> surpressed the error for private video, deleted video -> search for : self.report_error(str(e), e.format_traceback()) 
            if not info_dict:
                # print(f"Skipping {url}: Failed to retrieve video information.", end="\r")
                with lock:
                    condition_counts['skipping_video_info'] += 1
                return None

            duration = info_dict.get('duration', 0)
            if not (60 < duration < 300):  # Check if the video duration is between 60 and 300 seconds
                # print(f"Skipping {url}: Duration {duration} seconds is out of range.", end="\r")
                with lock:
                    condition_counts['skipping_duration'] += 1
                return None

            # print(f"Downloading {url} (Duration: {duration} seconds)", end="\r")
            ydl.download([url])

            video_id = info_dict.get('id', 'unknown')
            temp_filename = os.path.join(output_dir, f"{video_id}.mp3")

            # Generate a sequential filename using the shared `start_number`
            with lock:
                file_number = str(start_number.value).zfill(7)  # Ensure file name is 7 digits
                final_filename = os.path.join(output_dir, f"{file_number}.mp3")
                start_number.value += 1  # Increment safely
                condition_counts['file_saved'] += 1

            if os.path.exists(temp_filename):  # Check the correct path
                shutil.move(temp_filename, final_filename)
                processed_urls.add(url)

                return final_filename
            else:
                # print(f"Error: MP3 file not found for {url}", end="\r")
                with lock:
                    condition_counts['file_save_error'] += 1
    except Exception as e:
        # print(f"Error with URL {url}: {e}", end="\r")
        with lock:
            condition_counts['download_error'] += 1
    return None


# Function to download videos in batches
def download_video_batch(batch_urls, output_dir, condition_counts, processed_urls, start_number, lock, start_number_file, increment_counter):
    """Download and save videos in batches."""
    temp_filenames = []
    for url in batch_urls:
        temp_filename = download_single_video(url, output_dir, condition_counts, processed_urls, start_number, lock, start_number_file, increment_counter)
        if temp_filename:
            temp_filenames.append(temp_filename)
    return temp_filenames

# Main download function that runs in parallel
def download_short_videos_parallel(urls, output_dir, condition_counts, batch_size=1, start_number=1, start_number_file= DATASET_MP3_CONFIG / "start_number.txt"):
    """Download videos in parallel with batch processing."""
    url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    processed_urls = set()

    with ProcessPoolExecutor() as executor, multiprocessing.Manager() as manager:
        lock = manager.Lock()
        shared_start_number = manager.Value('i', start_number)  # Shared start number
        increment_counter = manager.Value('i', 0)  # Track the number of processed items
        futures = []

        for batch in url_batches:
            futures.append(
                executor.submit(download_video_batch, batch, output_dir, condition_counts, processed_urls, shared_start_number, lock, start_number_file, increment_counter)
            )

        for future in as_completed(futures):
            try:
                temp_filenames = future.result()
                for temp_filename in temp_filenames:
                    # print(f"Saved: {temp_filename}", end="\r")
                    pass
                with lock:
                    increment_counter.value += len(temp_filenames)  # Increment the counter after each batch
            except Exception as e:
                pass
                # print(f"Error in batch processing: {e}", end="\r")

# Main entry point of the program
def main():
    folder_path = DATASET_MP3_URLS
    output_dir = DATASET_MP3_DIR  # Changed directory to mp3_file_dir
    os.makedirs(output_dir, exist_ok=True)
    
    start_number_file = DATASET_MP3_CONFIG / 'start_number.txt'  # File to store the current start number
    start_number_dir = os.path.dirname(start_number_file)
    os.makedirs(start_number_dir, exist_ok=True)
    start_number = load_start_number(start_number_file)

    # logging
    log_file_path = os.path.join(DATASET_MP3_LOGS, get_log_filename())
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Create folder if it doesn't exist
    cleanup_old_log_files(os.path.dirname(log_file_path))

    # Create a queue for process-safe logging
    log_queue = multiprocessing.Queue()

    # Create a separate process for writing to the log file
    log_writer_process = multiprocessing.Process(target=write_to_log, args=(log_file_path,log_queue,))
    log_writer_process.start()

    with multiprocessing.Manager() as manager:
        condition_counts = manager.dict({
            'skipping_duration': 0,
            'file_saved': 0,
            'download_error': 0,
            'file_copied': 0,
            'skipping_video_info': 0,
            'file_save_error': 0,
            'skipping_already_processed': 0,
        })

        # Start the key listener in a separate thread so it doesn't block the main program
        listener_thread = threading.Thread(target=listen_for_exit, daemon=True)
        listener_thread.start()

        csv_file_to_start_from, start_from_line = get_csv_file_to_start_from(folder_path, start_number)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files_from_load = csv_files[csv_file_to_start_from::]
        print("start from genre:\t", csv_file_to_start_from, "-", csv_files_from_load[0])

        if csv_files_from_load:
            # Use tqdm for a progress bar over the list of CSV files
            with open(log_file_path, 'a', newline='', encoding='utf-8') as log_file:
                # Log the start of the process (in the main process)
                log_queue.put(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")



            with tqdm(csv_files_from_load, desc="Processing CSV Files", unit="file", leave=True) as pbar:
                for i,_csv in enumerate(pbar):
                    cleanup_temp_folder(output_dir)
                    start_number = load_start_number(start_number_file)
                    # each 50 genres we show the progressbar using \n gimmick
                    if i%50 == 0: pbar.unit = "file\n"
                    else: pbar.unit = "file"

                    csv_path = os.path.join(folder_path, _csv)
                    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        # URLs contain each row of URLs in _csv genre
                        URLs = [row['url'] for row in reader]

                    start_time = time.time()
                    download_short_videos_parallel(URLs, output_dir, condition_counts, batch_size=1, start_number=start_number, start_number_file=start_number_file)
                    pbar.update(1)  # Update the progress bar manually after each file
                 


                    end_time = time.time()
                    elapsed_time = end_time - start_time
                      # Update the progress bar manually
                    with open(log_file_path, 'a', newline='', encoding='utf-8') as log_file:
                        log_file.write(f"{_csv} - Time taken: {elapsed_time:.2f} seconds. {condition_counts}\n")

                    cleanup_temp_folder(output_dir)

        else:
            print("No CSV files found in the folder.")
            return


if __name__ == '__main__':
    '''
    temp_dataloader_configs has start_number.txt
    - set this to 1 if you want to restart
    - dont adjust it if you want to continue
    alt + f8 to quit program
    '''
    main()
    
