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
   # Optional. Public URL for /long upload links.
   # PUBLIC_UPLOAD_BASE_URL=https://your-public-domain
   # Optional upload server settings.
   # UPLOAD_HOST=0.0.0.0
   # UPLOAD_PORT=8080
   # UPLOAD_CHUNK_BYTES=8388608
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

### Recommended: /long upload button

If you cannot get Telegram `api_id` and `api_hash`, use the built-in upload
page. It bypasses the Telegram Bot API 20 MB download limit because the file is
uploaded directly to this bot's HTTP server.

1. Set a public URL for the upload server:

   ```env
   PUBLIC_UPLOAD_BASE_URL=https://your-public-domain
   ```

   For Amvera, use the public app URL. For a local computer on the same Wi-Fi,
   you can use the computer LAN address, for example:

   ```env
   PUBLIC_UPLOAD_BASE_URL=http://192.168.1.72:8080
   ```

   The phone must be able to open that URL. If Windows Firewall blocks it,
   allow Python or port `8080` on the local network.

2. Run the bot as usual:

   ```powershell
   python test_whisper.py
   ```

3. In Telegram, send:

   ```text
   /long
   ```

   The bot sends an inline button that opens the upload page. If you send a
   Telegram file larger than 20 MB directly to the bot, it automatically sends
   the same upload button.

4. Open the one-time link, upload `.ogg`, `.opus`, `.mp3`, `.m4a`, `.wav`, or
   `.webm`, then wait for the transcription in Telegram. The browser upload
   uses chunks and can resume already uploaded chunks while the one-time link is
   still alive.

5. During processing, the bot edits one status message with the current stage.
   Short transcriptions are sent as Telegram text. Long transcriptions are sent
   as a text preview plus a full `.txt` file.

Upload settings:

```env
MAX_UPLOAD_BYTES=2147483648
UPLOAD_CHUNK_BYTES=8388608
UPLOAD_TOKEN_TTL_SECONDS=86400
```

The bot also exposes:

```text
GET /health
GET /upload/<upload_id>
POST /upload/<upload_id>
POST /upload/<upload_id>/init
PUT /upload/<upload_id>/chunk/<index>
POST /upload/<upload_id>/complete
```

The standard cloud Telegram Bot API still cannot download Telegram files larger
than 20 MB. The upload button is the practical workaround when you do not have
Telegram `api_id` and `api_hash`.

### Optional: local Telegram Bot API

The standard cloud Telegram Bot API cannot download files larger than 20 MB.
To accept 1-1.5 hour voice messages sent directly as Telegram messages, run the
official Telegram Bot API server near this bot and set `TELEGRAM_API_BASE`.

Use a native Windows `telegram-bot-api.exe` for the first working setup. In
`--local` mode Telegram returns local file paths, so the Python bot must run on
the same machine and see the same filesystem as the Telegram Bot API server.
Docker adds extra path-mapping work and is not the recommended first setup here.

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
