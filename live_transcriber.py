import subprocess
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg
import time
import logging
import os
import librosa
from googletrans import Translator, LANGUAGES
import threading
from queue import Queue

# Configuration
YOUTUBE_URL = "https://www.youtube.com/watch?v=iEpJwprxDdk"
MODEL_SIZE = "base"  
SEGMENT_LENGTH = 5 
READ_TIMEOUT = 10  
MAX_RETRIES = 3  
BUFFER_SIZE = 4096  

# Audio Configuration
ENABLE_SPEECH_DETECTION = True
SILENCE_THRESHOLD = 0.005  # RMS threshold for silence
SPEECH_CONFIDENCE_THRESHOLD = 0.3
MIN_SPEECH_DURATION = 1.0
SKIP_MUSIC_DETECTION = True

# Language Config
TARGET_LANGUAGE = "zh-CN"
ENABLE_TRANSLATION = True
TRANSLATION_TIMEOUT = 5

translator = Translator()

# Logging setup - Option 2: Separate log file
def setup_logging_to_file():
    """Setup logging to write to a separate log file"""
    log_file = "transcriber.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w')  # Write to file only
        ]
    )
    print(f"Logs will be written to: {log_file}")
    print(f"To view logs in another terminal, run: python view_logs.py")
    return logging.getLogger()

# Choose your logging method here:
# logger = setup_logging_to_pipe()  # Option 1: Named pipe
logger = setup_logging_to_file()    # Option 2: Log file

def translate_text(text, target_lang=TARGET_LANGUAGE, source_lang='auto'):
    if not text or not text.strip():
        return text

    try:
        result = translator.translate(text, dest=target_lang, src=source_lang)    

        if result.src != target_lang:
            logger.debug(f"Translated from {result.src} to {target_lang}: {text[:50]}...")
        
        return result.text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  

def process_transcription_with_translation(segments, target_lang=TARGET_LANGUAGE):
    """Process transcription segments with translation"""
    for seg in segments:
        if seg.text.strip():
            original_text = seg.text.strip()
            
            if ENABLE_TRANSLATION:
                translated_text = translate_text(original_text, target_lang)
                
                # Show both original and translated if different
                if translated_text != original_text:
                    print(f"[{seg.start:.1f}s] {original_text}")
                    print(f"[{seg.start:.1f}s] → {translated_text}")
                else:
                    print(f"[{seg.start:.1f}s] {original_text}")
            else:
                print(f"[{seg.start:.1f}s] {original_text}")

def detect_speech_activity(audio, sample_rate=16000):
    """Detect if audio contains speech vs music/silence"""
    
    # Calculate energy-based features
    rms = np.sqrt(np.mean(audio**2))
    
    # Skip very quiet segments
    if rms < 0.005:  # Silence threshold
        return False, "silence"
    
    # Calculate spectral features
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    
    # Spectral centroid (brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate))
    
    # Spectral rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate))
    
    # Zero crossing rate (speech vs music indicator)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # MFCC features (voice characteristics)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # Simple heuristic classification
    # Speech typically has:
    # - Moderate spectral centroid (1000-4000 Hz)
    # - Higher zero crossing rate than music
    # - Specific MFCC patterns
    
    is_speech = (
        1000 < spectral_centroid < 4000 and  # Speech frequency range
        zcr > 0.05 and  # Sufficient zero crossings
        rms > 0.01  # Sufficient energy
    )
    
    if not is_speech:
        if zcr < 0.03:
            return False, "music"
        else:
            return False, "noise"
    
    return True, "speech"

def get_hls_url():
    """Get stream URL with live stream support"""
    # For live streams, try these formats in order
    formats_to_try = [
        "best[height<=720]",    # Fallback to video with audio
        "bestaudio",            # Any best audio
        "best",                 # Final fallback
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
    consecutive_non_speech = 0  
    total_segments_processed = 0
    total_segments_transcribed = 0
    
    logger.info(f"Speech detection: {'enabled' if ENABLE_SPEECH_DETECTION else 'disabled'}")
    logger.info(f"Translation: {'enabled' if ENABLE_TRANSLATION else 'disabled'} (target: {TARGET_LANGUAGE})")
    
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
                total_segments_processed += 1

                # Convert to numpy array
                audio = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for silent chunks
                if ENABLE_SPEECH_DETECTION:
                    is_speech, audio_type = detect_speech_activity(audio)
                    
                    if not is_speech:
                        consecutive_non_speech += 1
                        logger.debug(f"Skipping {audio_type} segment (consecutive: {consecutive_non_speech})")
                        
                        # Print status every 10 non-speech segments
                        if consecutive_non_speech % 10 == 0:
                            logger.info(f"Skipping {audio_type} - {consecutive_non_speech * SEGMENT_LENGTH}s of non-speech")
                        
                        continue

                    
                    # Reset counter when speech is detected
                    if consecutive_non_speech > 0:
                        logger.info(f"Speech detected after {consecutive_non_speech * SEGMENT_LENGTH}s of {audio_type}")
                        consecutive_non_speech = 0
                else:
                    consecutive_non_speech = 0 

                # Transcribe
                try:
                    segments, info = model.transcribe(audio, beam_size=3)
                    total_segments_transcribed += 1

                    for seg in segments:
                        if seg.text.strip():
                            original_text = seg.text.strip()
                            if not original_text or original_text == ".":
                                continue  # Skip silence markers or garbage
                            
                            if ENABLE_TRANSLATION:
                                # Option 1: Google Translate
                                translated_text = translate_text(original_text, TARGET_LANGUAGE)
                                
                                # Option 2: OpenAI (uncomment to use)
                                # translated_text = translate_with_openai(original_text, TARGET_LANGUAGE)
                                
                                # Show original and translation if different
                                if translated_text != original_text:
                                    # detected_lang = info.language if hasattr(info, 'language') else 'unknown'
                                    # print(f"[{seg.start:.1f}s] ({detected_lang}) {original_text}")
                                    print(f"[{seg.start:.1f}s] → {translated_text}")
                                else:
                                    print(f"[{seg.start:.1f}s] {original_text}")
                            else:
                                print(f"[{seg.start:.1f}s] {original_text}")

                except Exception as e:
                    logger.error(f"Transcription error: {e}")

                if total_segments_processed % 50 == 0:
                    transcription_rate = (total_segments_transcribed / total_segments_processed) * 100
                    logger.info(f"Stats: {total_segments_processed} processed, {total_segments_transcribed} transcribed ({transcription_rate:.1f}%)")
                    
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