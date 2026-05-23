# Voice Assistant

Telegram bot that accepts voice messages or `.ogg` audio files and transcribes
Russian speech through Groq Whisper.

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
   ```

3. Check local configuration without sending files to external services:

   ```powershell
   python test_whisper.py --check
   ```

4. Check that Telegram Bot API is reachable from this computer:

   ```powershell
   python test_whisper.py --diagnose
   ```

5. Check both Telegram and Groq:

   ```powershell
   python test_whisper.py --diagnose-full
   ```

6. Run the Telegram bot:

   ```powershell
   python test_whisper.py
   ```

## Local transcription test

To transcribe the included `voice.ogg` file through Groq:

```powershell
python test_whisper.py --transcribe voice.ogg
```

This command sends the selected audio file to Groq.
