import subprocess
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg
import time
from datetime import datetime

# Configuration - KEEP THESE THE SAME AS YOUR WORKING VERSION
YOUTUBE_URL = "https://www.youtube.com/watch?v=isJ7_4bf8KQ"
MODEL_SIZE = "small"
USE_GPU = False
SEGMENT_LENGTH = 10

# ONLY ADDITION: Debug logging
def debug_log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Initialize model - EXACTLY AS YOUR WORKING VERSION
try:
    model = WhisperModel(
        MODEL_SIZE,
        device="cuda" if USE_GPU else "cpu",
        compute_type="float32"
    )
    debug_log("Model loaded successfully")
except Exception as e:
    debug_log(f"Failed to load model: {str(e)}")
    exit(1)

# get_stream_url - ENHANCED FOR LIVE STREAMS
def get_stream_url():
    try:
        # First try to get the best audio format for live streams
        cmd = [
            "yt-dlp",
            "-g",
            "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=mp3]/bestaudio",
            "--no-check-certificates",
            "--live-from-start",  # Important for live streams
            YOUTUBE_URL
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream_url = result.stdout.strip()
        
        if stream_url:
            debug_log("Successfully got stream URL")
            return stream_url
        else:
            debug_log("No stream URL returned, trying fallback...")
            
            # Fallback: try without live-from-start
            cmd_fallback = [
                "yt-dlp",
                "-g",
                "-f", "bestaudio",
                "--no-check-certificates",
                YOUTUBE_URL
            ]
            result_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)
            fallback_url = result_fallback.stdout.strip()
            
            if fallback_url:
                debug_log("Successfully got fallback stream URL")
                return fallback_url
            else:
                debug_log("No fallback URL available")
                return None
                
    except subprocess.CalledProcessError as e:
        debug_log(f"Error getting stream URL: {e.stderr}")
        return None

# process_stream - CORE LOGIC REMAINS THE SAME
def process_stream():
    stream_url = get_stream_url()
    if not stream_url:
        debug_log("Failed to get stream URL")
        return

    debug_log(f"Stream URL: {stream_url}")
    
    # Check if this is likely a live stream (HLS)
    is_live_stream = 'm3u8' in stream_url or 'manifest' in stream_url or 'playlist' in stream_url
    debug_log(f"Detected live stream: {is_live_stream}")

    # FFmpeg pipeline - ENHANCED FOR LIVE STREAMS
    try:
        if is_live_stream:
            # Enhanced configuration for live streams
            process = (
                ffmpeg
                .input(stream_url, 
                       f='m3u8',  # Explicitly specify HLS format
                       protocol_whitelist='file,http,https,tcp,tls',
                       user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                       headers='Referer: https://www.youtube.com/',
                       live_start_index=0,  # Start from beginning of live stream
                       fflags='+discardcorrupt+genpts')  # Handle corrupt packets
                .output('pipe:', 
                       format='s16le',
                       acodec='pcm_s16le',
                       ac=1,
                       ar=16000,
                       loglevel='error')
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
        else:
            # Standard configuration for regular videos
            process = (
                ffmpeg
                .input(stream_url, **{'protocol_whitelist': 'file,http,https,tcp,tls'})
                .output('pipe:', format='s16le', ac=1, ar='16000')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
        
        debug_log("FFmpeg started successfully")
    except ffmpeg.Error as e:
        debug_log(f"FFmpeg error: {e.stderr.decode('utf8', 'replace')}")
        return

    # Audio processing - SAME AS YOUR WORKING VERSION
    sample_rate = 16000
    bytes_per_sample = 2
    chunk_size = sample_rate * SEGMENT_LENGTH * bytes_per_sample
    buffer = b''
    bytes_received = 0  # ONLY ADDITION: Track received bytes
    chunks_processed = 0
    
    try:
        debug_log("Starting transcription...")
        while True:
            # Read audio - SAME AS YOUR WORKING VERSION
            raw_data = process.stdout.read(4096)
            if not raw_data:
                debug_log("No more audio data received")
                break
                
            bytes_received += len(raw_data)  # ONLY ADDITION: Track bytes
            buffer += raw_data
            
            # Process chunks - SAME AS YOUR WORKING VERSION
            while len(buffer) >= chunk_size:
                segment = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                chunks_processed += 1
                
                audio_np = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                
                try:
                    segments, info = model.transcribe(
                        audio_np,
                        beam_size=5,
                        language="en",
                        vad_filter=True
                    )
                    for segment in segments:
                        print(f"[{segment.start:.2f}s] {segment.text}")
                        
                except Exception as e:
                    debug_log(f"Transcription error: {str(e)}")
            
            # ONLY ADDITION: Periodic status
            if bytes_received % (1024 * 100) == 0:  # Log every ~100KB
                debug_log(f"Received {bytes_received/1024:.1f}KB, processed {chunks_processed} chunks...")

    except KeyboardInterrupt:
        debug_log("\nStopping transcription...")
    finally:
        process.terminate()
        debug_log(f"Finished. Total received: {bytes_received/1024:.1f}KB, processed {chunks_processed} chunks")

if __name__ == "__main__":
    process_stream()