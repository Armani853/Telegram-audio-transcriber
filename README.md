# Voice Assistant

Telegram bot that accepts voice messages or `.ogg` audio files and transcribes
Russian speech through Groq Whisper. Long audio files are handled by splitting
them into safe chunks with `ffmpeg`.

## Setup

1. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Create `.env` in this directory:

   ```env
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   GROQ_API_KEY=your_groq_api_key
   # Optional, useful if api.telegram.org is blocked on your network.
   # TELEGRAM_PROXY_URL=http://127.0.0.1:8080
   # Optional. If omitted, GROQ_PROXY_URL uses TELEGRAM_PROXY_URL.
   # GROQ_PROXY_URL=http://127.0.0.1:8080
   # Optional. Required for Telegram files larger than 20 MB.
   # TELEGRAM_API_BASE=http://localhost:8081
   ```

3. Install `ffmpeg` if you want to transcribe long audio files:

   ```powershell
   winget install --id Gyan.FFmpeg -e
   ```

4. Check local configuration without sending files to external services:

   ```powershell
   python test_whisper.py --check
   ```

5. Check that Telegram Bot API is reachable from this computer:

   ```powershell
   python test_whisper.py --diagnose
   ```

6. Check both Telegram and Groq:

   ```powershell
   python test_whisper.py --diagnose-full
   ```

7. Run the Telegram bot:

   ```powershell
   python test_whisper.py
   ```

## Long Telegram voice messages

The standard cloud Telegram Bot API cannot download files larger than 20 MB.
To accept 1-1.5 hour voice messages, run the official Telegram Bot API server
near this bot and set `TELEGRAM_API_BASE`.

Example local server command:

```powershell
telegram-bot-api --api-id=YOUR_API_ID --api-hash=YOUR_API_HASH --local --http-port=8081
```

Then add this to `.env`:

```env
TELEGRAM_API_BASE=http://localhost:8081
```

Long files are automatically split into chunks before Groq Whisper. The default
chunk settings are safe for Groq's 25 MB direct upload limit:

```env
# Optional tuning
MAX_GROQ_CHUNK_BYTES=20971520
MAX_TELEGRAM_INPUT_BYTES=2147483648
AUDIO_CHUNK_SECONDS=900
AUDIO_CHUNK_OVERLAP_SECONDS=5
AUDIO_CHUNK_BITRATE=64k
```

## Local transcription test

To transcribe the included `voice.ogg` file through Groq:

```powershell
python test_whisper.py --transcribe voice.ogg
```

This command sends the selected audio file to Groq.
