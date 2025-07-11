import subprocess
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg
import time
import logging
import os
import librosa
from scipy import signal 

# Configuration
YOUTUBE_URL = "https://www.youtube.com/watch?v=lsx5ErH3k5o"
MODEL_SIZE = "base"  # Good balance of speed and accuracy
SEGMENT_LENGTH = 5  # Process 5-second chunks
READ_TIMEOUT = 10  # Seconds to wait for data
MAX_RETRIES = 3  # Max connection attempts
BUFFER_SIZE = 4096  # Read buffer size

# Logging setup
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def get_hls_url():
    """Get stream URL with live stream support"""
    # For live streams, try these formats in order
    formats_to_try = [
        "best[ext=m4a]",  # Best audio quality
        "bestaudio[ext=m4a]",  # Best audio in m4a
        "best[height<=720]",  # Fallback to video with audio
        "bestaudio",  # Any best audio
        "best",  # Final fallback
    ]
    
    for fmt in formats_to_try:
        try:
            cmd = [
                "yt-dlp",
                "-g",
                "-f", fmt,
                "--no-check-certificates",
                "--force-ipv4",
                YOUTUBE_URL
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=15, check=True)  # Longer timeout for live
            
            url = result.stdout.strip()
            if url:
                logger.info(f"Success with format: {fmt}")
                return url.split('\n')[-1]
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Format {fmt} failed: {e.stderr[:200]}...")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout getting URL for format {fmt}")
    
    logger.error("All format attempts failed")
    return None

def is_live_stream(url):
    """Check if URL is a live stream"""
    try:
        cmd = ["yt-dlp", "--print", "is_live", url]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=10, check=True)
        return result.stdout.strip().lower() == "true"
    except:
        return False

def setup_stream(url, is_live=False):
    """Configure FFmpeg for streaming with live stream optimizations"""
    try:
        input_options = {
            'headers': 'Referer: https://www.youtube.com\r\nUser-Agent: Mozilla/5.0',
            'fflags': '+discardcorrupt+genpts+nobuffer',
            'flags': 'low_delay',
            'timeout': '10000000',
        }
        
        if is_live:
            # Additional options for live streams
            input_options.update({
                'reconnect': '1',
                'reconnect_at_eof': '1',
                'reconnect_streamed': '1',
                'reconnect_delay_max': '30',
                'live_start_index': '0',
            })
        
        return (
            ffmpeg
            .input(url, **input_options)
            .output(
                'pipe:',
                format='s16le',
                ac=1,
                ar=16000,
                acodec='pcm_s16le',
                loglevel='error',
                map='a:0'
            )
            .global_args('-nostdin')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg setup failed: {e.stderr.decode('utf8', 'replace')}")
        return None

def process_audio_stream(process, model):
    """Core audio processing with HLS optimizations"""
    sample_rate = 16000
    chunk_size = sample_rate * SEGMENT_LENGTH * 2  # 16-bit samples
    buffer = b''
    last_data_time = time.time()
    
    while True:
        # Check for timeout
        if time.time() - last_data_time > READ_TIMEOUT:
            logger.warning(f"No data for {READ_TIMEOUT}s - restarting")
            return False
            
        # Read data
        try:
            data = os.read(process.stdout.fileno(), BUFFER_SIZE)
            if not data:
                time.sleep(0.1)
                continue
                
            last_data_time = time.time()
            buffer += data
            
            # Process complete chunks
            while len(buffer) >= chunk_size:
                segment = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                
                # Convert to numpy array
                audio = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for silent chunks
                rms = np.sqrt(np.mean(audio**2))
                if rms < 0.001:
                    logger.warning(f"Silent chunk detected (RMS: {rms:.6f})")
                
                # Transcribe
                segments, _ = model.transcribe(audio, beam_size=3)
                for seg in segments:
                    print(f"[{seg.start:.1f}s] {seg.text}")
                    
        except (BlockingIOError, ValueError, OSError) as e:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return False
            
    return True

def transcribe_hls():
    """Main transcription workflow for HLS"""
    logger.info("Initializing Whisper model...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="float32")
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        logger.info(f"Attempt {retry_count + 1}/{MAX_RETRIES}")
        
        # Get HLS URL
        stream_url = get_hls_url()
        if not stream_url:
            retry_count += 1
            time.sleep(5)
            continue
        
        logger.info(f"Stream URL obtained: {stream_url[:80]}...")
        
        # Start FFmpeg
        process = setup_stream(stream_url)
        if not process:
            retry_count += 1
            time.sleep(5)
            continue
        
        # Process stream
        try:
            logger.info("Starting HLS transcription...")
            if not process_audio_stream(process, model):
                logger.warning("Stream processing ended unexpectedly")
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
        finally:
            logger.info("Cleaning up FFmpeg process...")
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                pass
        
        retry_count += 1
        logger.info(f"Restarting in 5 seconds...")
        time.sleep(5)

    logger.error("Max retries reached. Exiting.")

if __name__ == "__main__":
    try:
        transcribe_hls()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")