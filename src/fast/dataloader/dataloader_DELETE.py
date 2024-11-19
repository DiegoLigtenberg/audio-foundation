from yt_dlp import YoutubeDL
import os
import csv
import time

# urls can be a batch up up to 99999 songs, then we continue with new
folder_path = 'database/urls'
# Directory for storing processed audio files
output_dir = 'database/temp_mp3_error_analysis/'
os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read the first CSV file from the list
if csv_files:
    first_csv = csv_files[1]
    csv_path = os.path.join(folder_path, first_csv)

    # Open the CSV file and extract the 'url' column
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Use DictReader to get rows as dictionaries
        URLs = [row['url'] for row in reader]  # Extract 'url' column into a list
        # print(URLs)  # Print the list of URLs to check
else:
    print("No CSV files found in the folder.")

# print(csv_path)
# asd
# List of URLs to download from (you can add more URLs here for batch processing)
URLS = URLs[100:300]

print(len(URLS))
# print(URLS)
# URLS = ["https://www.youtube.com/watch?v=CKM4Fap4LC0&pp=ygUQYWNpZCBjcnVuayBtdXNpYw%3D%3D","https://www.youtube.com/watch?v=CKM4Fap4LC0&pp=ygUQYWNpZCBjcnVuayBtdXNpYw%3D%3D","https://www.youtube.com/watch?v=MmwtnaLxvAY&pp=ygUQYWNpZCBjcnVuayBtdXNpYw%3D%3D"]
# fine the download options with higher quality audio settings
# asd
# fine the download options with higher quality audio settings
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }],
    'postprocessor_args': [
        '-ar', '44100',
        '-ac', '1',
    ],
    'quiet': True,
    'ignoreerrors': True,
    'outtmpl': 'temp.%(ext)s',  # Temporary file for renaming
    'no-keep-fragments': True,
    'resize-buffer': True,
}


# Function to download videos and save with incremental filenames
def download_short_videos(urls, output_dir):
    skip_counts = {
        "no_info_dict": 0,
        "not_in_duration_range": 0,
        "error_with_url": 0,
    }

    current_number = 1 #len(os.listdir(output_dir)) + 1  # Start numbering based on existing files
    with YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                # Get video info
                
                info_dict = ydl.extract_info(url, download=False)
                # asd

                # Check if info_dict is valid
                if not info_dict:
                    print(f"Skipping {url}: Failed to retrieve video information.")
                    skip_counts["no_info_dict"] += 1
                    continue

                # Get duration safely
                duration = info_dict.get('duration', 0)  # Duration in seconds

                if 60 < duration < 300:  # Check if the video is between 1 and 5 minutes
                    print(f"Downloading {url} (Duration: {duration} seconds)", end="\r")
                    ydl.download([url])  # Download the video to the temp location

                    # Generate filename (zero-padded to 7 digits)
                    file_number = str(current_number).zfill(7)
                    final_filename = os.path.join(output_dir, f"{file_number}.mp3")

                    # Check if the final filename exists and remove it if it does
                    if os.path.exists(final_filename):
                        os.remove(final_filename)  # Delete the existing file if it exists

                    # Rename and move the temp file
                    temp_file = "temp.mp3"
                    if os.path.exists(temp_file):
                        os.rename(temp_file, final_filename)  # This will now overwrite the file
                        print(f"Saved: {final_filename}", end="\r")
                    else:
                        print(f"Temporary file not found for {url}")
                    
                    current_number += 1
                else:
                    print(f"Skipping {url} (Duration: {duration} seconds)",end="\r")
                    skip_counts["not_in_duration_range"] += 1
            except Exception as e:
                print(f"Error with URL {url}: {e}",end="\r")
                skip_counts["error_with_url"] += 1

    print(skip_counts)

# Record the start time
start_time = time.time()

URLS = URLs[100:200]
URLS = ["https://www.youtube.com/watch?v=0veWDx5beDs&pp=ygUPYWNpZC13YXZlIG11c2lj,Dark Minimal Techno NON STOP Radio Mix"]
# Call the function to download videos
download_short_videos(URLS, output_dir)

asd
URLS = URLs[200:300]

output_dir = 'database/temp_mp3_error_analysis2/'
os.makedirs(output_dir, exist_ok=True)
download_short_videos(URLS, output_dir)

# Record end time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")