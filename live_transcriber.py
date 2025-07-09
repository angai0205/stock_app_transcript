import subprocess
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg
import time
from datetime import datetime

# Configuration - KEEP THESE THE SAME AS YOUR WORKING VERSION
YOUTUBE_URL = "https://www.youtube.com/watch?v=MfDZN_gqy0Q"
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
    
    # Check if this is likely a live stream (HLS or DASH)
    is_live_stream = any(keyword in stream_url.lower() for keyword in ['m3u8', 'manifest', 'playlist', 'dash'])
    debug_log(f"Detected live stream: {is_live_stream}")

    # FFmpeg pipeline - ENHANCED FOR LIVE STREAMS
    try:
        if is_live_stream:
            # Enhanced configuration for live streams (both HLS and DASH)
            input_args = {
                'protocol_whitelist': 'file,http,https,tcp,tls',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'headers': 'Referer: https://www.youtube.com/',
                'fflags': '+discardcorrupt+genpts'
            }
            
            # Add format-specific options
            if 'm3u8' in stream_url.lower():
                input_args['f'] = 'm3u8'
                input_args['live_start_index'] = '0'
                debug_log("Using HLS configuration")
            elif 'dash' in stream_url.lower():
                input_args['f'] = 'dash'
                debug_log("Using DASH configuration")
            else:
                debug_log("Using generic live stream configuration")
            
            try:
                process = (
                    ffmpeg
                    .input(stream_url, **input_args)
                    .output('pipe:', 
                           format='s16le',
                           acodec='pcm_s16le',
                           ac=1,
                           ar=16000,
                           loglevel='error')
                    .overwrite_output()
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                debug_log("Enhanced FFmpeg configuration started successfully")
            except ffmpeg.Error as e:
                debug_log(f"Enhanced configuration failed: {e.stderr.decode('utf8', 'replace')}")
                debug_log("Trying fallback configuration...")
                
                # Fallback: simpler configuration
                process = (
                    ffmpeg
                    .input(stream_url, 
                           protocol_whitelist='file,http,https,tcp,tls',
                           user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
                    .output('pipe:', 
                           format='s16le',
                           acodec='pcm_s16le',
                           ac=1,
                           ar=16000,
                           loglevel='error')
                    .overwrite_output()
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                debug_log("Fallback FFmpeg configuration started successfully")
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

    # Audio processing - ENHANCED FOR LIVE STREAMS
    sample_rate = 16000
    bytes_per_sample = 2
    chunk_size = sample_rate * SEGMENT_LENGTH * bytes_per_sample
    buffer = b''
    bytes_received = 0
    chunks_processed = 0
    
    debug_log(f"Chunk size: {chunk_size} bytes ({SEGMENT_LENGTH}s of audio)")
    
    try:
        debug_log("Starting transcription...")
        last_data_time = time.time()
        consecutive_empty_reads = 0
        max_wait_time = 10.0  # Wait up to 10 seconds for data
        
        while True:
            # Read audio - ENHANCED FOR LIVE STREAMS
            raw_data = process.stdout.read(8192)  # Increased buffer size for live streams
            
            if not raw_data:
                consecutive_empty_reads += 1
                current_time = time.time()
                
                # Check if we've been waiting too long
                if current_time - last_data_time > max_wait_time:
                    debug_log(f"No audio data for {max_wait_time} seconds, checking FFmpeg status...")
                    
                    # Check if FFmpeg process is still running
                    if process.poll() is not None:
                        debug_log(f"FFmpeg process has exited with code {process.returncode}")
                        # Read any remaining stderr output
                        stderr_output = process.stderr.read().decode('utf8', 'replace')
                        if stderr_output:
                            debug_log(f"FFmpeg stderr: {stderr_output}")
                        break
                    else:
                        debug_log("FFmpeg process is still running but not providing data")
                        # Try to read stderr to see if there are any error messages
                        try:
                            stderr_output = process.stderr.read1(1024).decode('utf8', 'replace')
                            if stderr_output:
                                debug_log(f"FFmpeg stderr: {stderr_output}")
                        except:
                            pass
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                continue
            
            # Reset counters when we get data
            consecutive_empty_reads = 0
            last_data_time = time.time()
            bytes_received += len(raw_data)
            buffer += raw_data
            
            # Only log every 10th read to reduce spam
            if bytes_received % (8192 * 10) == 0:
                debug_log(f"Received {len(raw_data)} bytes, total: {bytes_received}, buffer: {len(buffer)}")
            
            # Process chunks - ENHANCED DEBUGGING
            while len(buffer) >= chunk_size:
                segment = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                chunks_processed += 1
                
                debug_log(f"Processing chunk {chunks_processed} ({len(segment)} bytes)")
                
                # Convert to numpy array
                audio_np = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check if audio has actual content (not just silence)
                audio_rms = np.sqrt(np.mean(audio_np**2))
                if audio_rms < 0.001:  # Very low volume
                    debug_log(f"Chunk {chunks_processed} appears to be silence (RMS: {audio_rms:.6f})")
                
                try:
                    segments, info = model.transcribe(
                        audio_np,
                        beam_size=5,
                        language="en",
                        vad_filter=True
                    )
                    
                    if segments:
                        for segment in segments:
                            print(f"[{segment.start:.2f}s] {segment.text}")
                    else:
                        debug_log(f"Chunk {chunks_processed}: No speech detected")
                        
                except Exception as e:
                    debug_log(f"Transcription error in chunk {chunks_processed}: {str(e)}")
            
            # Periodic status - ENHANCED
            if bytes_received % (1024 * 100) == 0:  # Log every ~100KB
                debug_log(f"Received {bytes_received/1024:.1f}KB, processed {chunks_processed} chunks, buffer: {len(buffer)} bytes")

    except KeyboardInterrupt:
        debug_log("\nStopping transcription...")
    finally:
        process.terminate()
        debug_log(f"Finished. Total received: {bytes_received/1024:.1f}KB, processed {chunks_processed} chunks")
        
        # Process any remaining buffer
        if len(buffer) > 0:
            debug_log(f"Processing final {len(buffer)} bytes of buffer...")
            try:
                audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                segments, info = model.transcribe(
                    audio_np,
                    beam_size=5,
                    language="en",
                    vad_filter=True
                )
                for segment in segments:
                    print(f"[{segment.start:.2f}s] {segment.text}")
            except Exception as e:
                debug_log(f"Error processing final buffer: {str(e)}")

if __name__ == "__main__":
    process_stream()