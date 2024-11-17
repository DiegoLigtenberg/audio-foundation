from yt_dlp import YoutubeDL

# urls can be a batch up up to 99999 songs, then we continue with new

# List of URLs to download from (you can add more URLs here for batch processing)
URLS = [
    'https://www.youtube.com/watch?v=INVALID_URL',  # Invalid URL (for testing error handling)
    'https://www.youtube.com/watch?v=QYh6mYIJG2Y&ab_channel=ArianaGrandeVevo'  # Valid URL
]

# Define the download options with higher quality audio settings
ydl_opts = {
    'format': 'bestaudio/best',  # Download the best audio quality available
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',  # Simpler audio extraction
        'preferredcodec': 'wav',  # Convert audio to WAV format (lossless)
        'preferredquality': '320',  # Set quality to 320 kbps for better audio fidelity
    }],
    'postprocessor_args': [
        '-ar', '44100',  # Set sample rate to 44.1 kHz (standard for most AI audio tasks)
        '-ac', '1',       # Use mono audio for simplicity in speech-related AI tasks
    ],
    'quiet': True,  # Set to True if you want to suppress output messages
    'ignoreerrors': True,  # Ignore errors from invalid URLs
    'outtmpl': 'audio_files/%(autonumber)s.%(ext)s',  # Save audio with auto-incrementing number, e.g., 1.wav, 2.wav
}

# Download all URLs without needing to manually loop over them
with YoutubeDL(ydl_opts) as ydl:
    ydl.download(URLS)  # Download all URLs
