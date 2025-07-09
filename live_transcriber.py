import yt_dlp
import whisper
import numpy as np
import subprocess
import time
from typing import Optional
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriberConfig:
    chunk_duration: int = 10  # seconds per chunk
    model_size: str = "base"  # Whisper model size
    sample_rate: int = 16000  # audio sample rate
    language: str = "en"      # transcription language

class YouTubeTranscriber:
    def __init__(self, config: TranscriberConfig):
        self.config = config
        self.model = whisper.load_model(config.model_size)
        self.ffmpeg_process: Optional[subprocess.Popen] = None

    # Converts the YouTube URL into an audio URL of the last 10 seconds of the stream
    def _get_audio_url(self, youtube_url: str) -> str:
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

    # Turns the audio URL into a subprocess object for whisper to process
    def _ffmpeg_stream(self, audio_url: str) -> subprocess.Popen:
        ffmpeg_cmd = [
            'C:\\Tools\\ffmpeg\\bin\\ffmpeg.exe'
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
            return subprocess.Popen(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL, 
                bufsize=10**8
            )
        except Exception as e:
            raise Exception("Failed to start ffmpeg stream")

    def _process_audio(self, chunk: bytes) -> str:
        audio_np = np.frombuffer(chunk, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        if len(audio_np) < 1000:
            return ""
        
        result = self.model.transcribe(
            audio_np,
            language=self.config.language,
            fp16=False,
            verbose=False
        )
        
        return result['text'][0].strip() if result['text'] else ""

    def transcribe_stream(self, youtube_url: str):
        try:
            # Step 1: Get audio stream url
            audio_url = self._get_audio_url(youtube_url)

            # Step 2: Start FFmpeg stream
            self.ffmpeg_process = self._ffmpeg_stream(audio_url)

            # Step 3: 
            chuck_size = self.config.sample_rate * self.config.chunk_duration * 2

            # Step 4: Process the stream in small chunks
            while True: 
                chunk = self.ffmpeg_process.stdout.read(chuck_size)

                if not chunk:
                    break

                start_time = time.time()
                text = self._process_audio(chunk)
                processing_time = time.time() - start_time

                if text: 
                    logger.info(f"Transcription ({processing_time:.2f}s): {text}")
                else: 
                    logger.warning("Empty transciprtion result")
                    
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                logger.info("FFmpeg process terminated")

if __name__ == "__main__":
    config = TranscriberConfig(
        chunk_duration=5, 
        model_size="small", 
        sample_rate=16000, 
        language="en"
    )
    transcriber = YouTubeTranscriber(config)

    youtube_url = input("Enter YouTube Live URL: ").strip()

    audio_url = transcriber._get_audio_url(youtube_url)
    # print(f"[INFO] Streaming from: {audio_url}")
    transcriber.transcribe_stream(youtube_url)
        
