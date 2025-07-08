import yt_dlp
import whisper
import numpy as np
import subprocess
import tempfile
import time

CHUNK_DURATION = 10  # seconds
MODEL_SIZE = "base"  # or "small", "medium", "large"

# Converts the YouTube URL into an audio URL of the last 10 seconds of the stream
def get_audio_url(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if info is None:
                raise ValueError("Invalid YouTube URL")
            
            # Loops through all formats and returns the first one that has an audio codec
            for f in info.get('formats', []):
                if f.get('acodec') != 'none':
                    return f['url']
            
            raise ValueError("No audio stream found in available formats")
    except Exception as e:
        raise Exception("Failed to get audio URL")


def ffmpeg_stream(audio_url):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', audio_url,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', '16000',
        '-loglevel', 'error',
        '-'
    ]

    try:
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        return process
    except Exception as e:
        raise Exception("Failed to start ffmpeg stream")

def main():
    youtube_url = input("Enter YouTube Live URL: ").strip()
    # stream_and_transcribe(youtube_url)
    audio_url = get_audio_url(youtube_url)
    print(f"[INFO] Streaming from: {audio_url}")

if __name__ == "__main__":
    main()
