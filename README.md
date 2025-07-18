# Live YouTube Transcription with Faster-Whisper

A script that transcribes **live YouTube streams** in real-time using [faster-whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition. Optional support for **translation**, **silence skipping**, and **output syncing** for better alignment with the video stream.

## Features

- 🔊 Real-time transcription from YouTube Live streams
- 🌍 Automatic translation to target language
- 🤫 Skips silence/music using audio feature detection
- 🕒 Adds delay to sync transcript with video playback
- 🔁 Auto-reconnects on stream failure
- 📝 Logs written to transcriber.log

## Installtion

```bash
git clone https://github.com/angai0205/stock_app_transcript.git
cd stock_app_transcript
```

### Use virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Install dependencies

```bash
pip install -r requirements.txt

# Or manually

pip install faster-whisper ffmpeg-python numpy librosa "googletrans==4.0.0-rc1" yt-dlp
```

## Install FFmpeg

### On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

### On MacOS (using Homebrew):

```bash
brew install ffmpeg
```

### On Windows:
- Download from: https://ffmpeg.org/download.html
- Add ffmpeg to your system PATH.

## Optional: Speed Boost (for GPUs):

```bash
pip install faster-whisper[torch]
```

#### And change this line within the script

```bash
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
```

## Running the script

#### Set the target YouTube Live stream URL:

```bash
YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_LIVE_STREAM_ID"
```

#### Setting the target language for the translation:

```bash
TARGET_LANGUAGE = "en"

# View the list of available languages at: https://cloud.google.com/translate/docs/languages
```

```bash
# Ensuring that python 3.8+ is being used

python transcriber.py   
```

## Logging

```bash
# Run in seperate terminal to see logging information
python view_logs.py
```


