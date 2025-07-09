import subprocess
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg

# Configuration
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZhKJ8C1Yc_4"
MODEL_SIZE = "small"  # "tiny", "base", "small", "medium"
USE_GPU = False  # Set to True if you have CUDA
SEGMENT_LENGTH = 10  # Process 10-second chunks

# Initialize model with error handling
try:
    model = WhisperModel(
        MODEL_SIZE,
        device="cuda" if USE_GPU else "cpu",
        compute_type="float32"  # Force float32 to avoid warning
    )
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    exit(1)

def get_stream_url():
    """Get audio stream URL with fallback options"""
    try:
        cmd = [
            "yt-dlp",
            "-g",
            "-f", "bestaudio[ext=webm]/bestaudio",
            "--no-check-certificates",
            YOUTUBE_URL
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting stream URL: {e.stderr}")
        return None

def process_stream():
    stream_url = get_stream_url()
    if not stream_url:
        print("Failed to get stream URL")
        return

    # FFmpeg pipeline configuration
    ffmpeg_args = {
        'ac': 1,            # Mono audio
        'ar': '16000',      # 16kHz sample rate
        'fflags': '+discardcorrupt+genpts',
    }

    try:
        process = (
            ffmpeg
            .input(stream_url, **{'protocol_whitelist': 'file,http,https,tcp,tls'})
            .output('pipe:', format='s16le', **ffmpeg_args)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode('utf8', 'replace')}")
        return

    sample_rate = 16000
    bytes_per_sample = 2  # 16-bit audio
    chunk_size = sample_rate * SEGMENT_LENGTH * bytes_per_sample
    buffer = b''
    
    try:
        while True:
            # Read raw audio data
            raw_data = process.stdout.read(4096)
            if not raw_data:
                break
                
            buffer += raw_data
            
            # Process when we have enough audio
            while len(buffer) >= chunk_size:
                segment = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                
                # Convert to numpy array
                audio_np = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe
                try:
                    segments, info = model.transcribe(
                        audio_np,
                        beam_size=5,
                        language="en",
                        vad_filter=True  # Enable voice activity detection
                    )
                    
                    for segment in segments:
                        print(f"[{segment.start:.2f}s] {segment.text}")
                        
                except Exception as e:
                    print(f"Transcription error: {str(e)}")

    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        process.terminate()

if __name__ == "__main__":
    process_stream()