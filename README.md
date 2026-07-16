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
   # Optional. Use only if YouTube asks the hosting IP to confirm it is not a bot.
   # YTDLP_COOKIES_FILE=/path/to/youtube-cookies.txt
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

## Media library

The bot also includes a local media-library mode for lectures, voice files,
videos, podcasts, and meeting recordings.

In Telegram:

```text
/library
```

This opens `/app`, where you can upload `.ogg`, `.opus`, `.mp3`, `.m4a`, `.wav`,
`.webm`, `.mp4`, `.mov`, or `.mkv`. The media-library upload also uses chunked
resumable upload. After a file is accepted, a background worker stores the
original file, extracts and normalizes audio with `ffmpeg`, transcribes it with
Groq Whisper, creates `.txt`, `.srt`, and `.vtt` outputs, indexes the transcript,
and generates chapters, summary, and action items.

Search from Telegram:

```text
/search нормализация базы данных
```

Search results include the media title, an approximate timestamp, a transcript
snippet, and a link that opens the media page at that timestamp.

Local media-library settings:

```env
# Optional
# On Windows defaults to %LOCALAPPDATA%\VoiceAssistant
MEDIA_DATA_DIR=C:\Users\you\AppData\Local\VoiceAssistant
MEDIA_DB_PATH=media_library.sqlite3
MEDIA_STORAGE_DIR=storage
MEDIA_MAX_ACTIVE_JOBS=2
MEDIA_UPLOAD_SESSION_TTL_SECONDS=86400
MEDIA_MAX_DURATION_SECONDS=10800
ENABLE_GROQ_LLM_POSTPROCESSING=1
GROQ_LLM_MODEL=llama-3.1-8b-instant
```

The first implementation is intentionally local-first: SQLite + filesystem
object storage + an in-process worker. The database and `storage/` directory are
ignored by git. This keeps the current simple local bot workflow intact while
leaving a clear path to Redis/PostgreSQL/MinIO later.

Media-library features included in the local version:

- persistent SQLite tables for users, media items, files, jobs, chunks, search,
  chapters, summaries, tasks, and upload sessions;
- local object storage for original media, extracted audio, TXT, SRT, VTT, and
  metadata JSON;
- chunked resumable upload with TTL cleanup;
- background in-process worker with job statuses and SSE progress;
- Telegram progress message editing for media-library uploads;
- video audio extraction and loudness normalization through `ffmpeg`;
- SHA-256 deduplication with derived transcript/search data reuse;
- Groq Whisper transcription plus optional Groq LLM chapters, summary, key
  points, and action items;
- web player with clickable transcript/chapter timestamps;
- Telegram `/search` and web search over transcript chunks;
- task export as `tasks.txt`.

Production upgrade path:

- replace SQLite with PostgreSQL for concurrent multi-user hosting;
- replace local filesystem storage with MinIO/S3;
- replace the in-process queue with Redis + RQ/Celery workers;
- keep the same media/job/object-storage boundaries already present in the
  local implementation.

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

## YouTube links

Send a regular `youtube.com` or `youtu.be` video link to the bot. Videos up to
12 hours are accepted by default. The bot first uses the video's original manual
or automatic captions; if none exist, it downloads the audio and asks the
configured speech-to-text provider for segment timestamps.

YouTube results are always sent as a UTF-8 `.txt` document, including for short
videos. The first section is a timestamped table of contents; the second is the
complete transcript. Native YouTube chapters are preferred, timestamp rows in
the description are the next fallback, and videos without either receive an
automatic transcript-derived outline covering the full duration. Tiny caption
fragments are grouped into readable phrases. Every row keeps the timestamp on
the left and the text on the right:

```text
СОДЕРЖАНИЕ ВИДЕО

0:03     │ Introduction
2:15     │ Main topic

РАСШИФРОВКА ВИДЕО

0:03     │ first caption line
0:07     │ next caption line
```

Current YouTube extraction needs yt-dlp's EJS component and a JavaScript
runtime. Both are installed by `requirements.txt` (`yt-dlp[default]` plus
Deno). If YouTube challenges a hosting IP even with those installed, export
Netscape-format cookies and set `YTDLP_COOKIES_FILE` to their server path.
When `YTDLP_PROXY_URL` is not set, the bot reuses `TELEGRAM_PROXY_URL` for
YouTube; this avoids a blocked direct IP while preserving a separate override.

Optional settings:

```env
YOUTUBE_MAX_DURATION_SECONDS=43200
YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS=7200
YTDLP_JS_RUNTIME=deno
YTDLP_PROXY_URL=http://proxy-host:proxy-port
YTDLP_COOKIES_FILE=/path/to/youtube-cookies.txt
YTDLP_EXTRACTOR_ARGS=youtube:player_client=default,-android_sdkless
```

The bot retries transient extractor and media-fragment failures. It also
distinguishes temporary HTTP failures from videos that YouTube itself marks as
private, region-restricted, or copyright-blocked. A blocked video cannot be
transcribed because YouTube exposes neither its media nor its captions.

## YouTube downloads

The main keyboard has a separate `📥 Скачать видео с YouTube` flow. After the
next URL, the bot reads the real YouTube formats and offers only exact available
360p, 720p, 1080p, and audio options. Video is downloaded as MP4 with the exact
selected height; audio is normalized to MP3. The completed media is verified
with ffprobe before a user receives a download link.

Files smaller than the configured Telegram threshold are additionally sent in
chat. Larger files use an unguessable HTTP download URL, avoiding the cloud Bot
API's upload limit. The landing page displays title, exact quality, size,
duration, and expiry. Results are cached by YouTube video ID and quality, so a
repeat request is returned immediately. Expired files and database records are
deleted automatically.

```env
YOUTUBE_DOWNLOAD_TTL_SECONDS=259200
YOUTUBE_DOWNLOAD_REQUEST_TTL_SECONDS=1800
YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS=21600
YOUTUBE_DOWNLOAD_CONCURRENT_FRAGMENTS=8
YOUTUBE_DOWNLOAD_MAX_CONCURRENT=2
YOUTUBE_TELEGRAM_DIRECT_LIMIT_BYTES=51380224
YOUTUBE_METADATA_CACHE_TTL_SECONDS=600
YOUTUBE_TRANSCRIPT_CACHE_TTL_SECONDS=21600
YOUTUBE_TRANSCRIPT_MAX_CONCURRENT_BUILDS=2
YOUTUBE_RESULT_MAX_CONCURRENT_SENDS=4
HTTP_LISTEN_BACKLOG=2048
# YOUTUBE_DOWNLOAD_STORAGE_DIR=/persistent/youtube_downloads
# YOUTUBE_DOWNLOAD_DB_PATH=/persistent/youtube_downloads.sqlite3
```

### Одновременные пользователи

- Одинаковые варианты YouTube-ссылки (`watch`, `youtu.be`, `shorts`, `live` и ссылки с tracking-параметрами) приводятся к одному video ID.
- Сотни одновременных запросов одного видео разделяют один запрос метаданных, одну загрузку и одну сборку расшифровки. Каждый чат получает отдельный ответ, но тяжёлая работа не дублируется.
- Готовая расшифровка кэшируется на 6 часов, метаданные — на 10 минут. Значения настраиваются переменными выше.
- Разные тяжёлые видео обрабатываются ограниченным числом параллельных сборок; остальные безопасно ожидают, не забирая одновременно всю память и CPU.
- Отправка TXT в Telegram ограничена четырьмя параллельными документами и повторяется при временной сетевой ошибке.
- Горячая выдача готовых файлов не обновляет SQLite на каждом byte-range запросе; активные записи держатся в памяти, а TTL продлевается периодически.

Один процесс обеспечивает корректность и backpressure, но не превращает один сервер и один API-ключ в бесконечную вычислительную мощность. Для тысяч одновременно разных длинных видео нужны несколько stateless-экземпляров бота, общая постоянная очередь (например, Redis/RabbitMQ), отдельные STT-воркеры, S3-совместимое хранилище/CDN и оплаченные лимиты провайдеров. Без этого качество сохраняется, но время ожидания разных тяжёлых задач растёт вместе с очередью.

### Слабый интернет и обрыв связи

- Скачивание с YouTube выполняется на сервере и не зависит от того, открыт ли Telegram на телефоне.
- Готовый результат сначала сохраняется в SQLite, затем доставляется в Telegram. При временном сбое Telegram отправка повторяется автоматически и переживает перезапуск процесса бота.
- Файловый HTTP-сервер поддерживает `Range`/`206 Partial Content`, поэтому браузер или менеджер загрузок может продолжить большой файл с уже полученного байта.
- Срок хранения по умолчанию — 72 часа. Каждое открытие ссылки или продолжение загрузки снова продлевает его на полный срок.
- Если конкретное качество не удалось подготовить, бот оставляет кнопки повторной попытки, меньшего качества и «Только аудио».
- Полностью передать новый файл на устройство без какого-либо интернет-соединения физически невозможно. Сервер закончит и сохранит работу офлайн от клиента; устройство получит сообщение и сможет продолжить скачивание после восстановления сети.
- Для доступа за пределами локальной LAN значение `PUBLIC_UPLOAD_BASE_URL` обязано быть публичным HTTPS-адресом. Для тысяч пользователей файлы следует вынести в S3-совместимое хранилище/CDN с поддержкой Range, а бот оставить управляющим сервисом.

For production, `PUBLIC_UPLOAD_BASE_URL` must be a public HTTPS address pointing
to the embedded HTTP service. Download only media that the user owns or is
authorized to save.

The web UI is responsive for laptop, Android phone, iPhone, and iPad widths.
Download responses provide both an ASCII fallback filename and an RFC 5987
UTF-8 filename, the correct media MIME type, byte ranges for pause/resume, and
mobile-safe security headers. Uploads are chunked and resumable after a dropped
connection. Automated cross-device tests exercise desktop Chrome, Android
Chrome, iPhone Safari, and iPad Safari user-agent profiles, including parallel
first/middle/last byte ranges and interrupted upload recovery.

A private address such as `http://192.168.x.x` works only while the device is on
the same local network. Phones on cellular data and users outside that network
require a public HTTPS `PUBLIC_UPLOAD_BASE_URL`; client-side code cannot make a
private LAN address globally reachable.

Small YouTube results under `YOUTUBE_TELEGRAM_DIRECT_LIMIT_BYTES` are sent
directly through Telegram and therefore work globally without the HTTP file
server. The bot first attempts a streamable Telegram video and automatically
retries as a document when Telegram does not accept the video's codec. All
download actions use the same neutral `⬇️ Скачать готовый файл` label.

For large files, configure a stable public hostname. With a Cloudflare named
tunnel, publish `https://files.example.com` to `http://127.0.0.1:8080`, keep
`cloudflared` running as a service, and set:

```env
PUBLIC_UPLOAD_BASE_URL=https://files.example.com
```

An account-less `trycloudflare.com` Quick Tunnel is suitable only for a short
test: its hostname changes after restart, it has no uptime guarantee, and it
must not be used as the production download address. S3/R2 plus a CDN is the
preferred architecture when large files will be downloaded by many users.

## Speech-to-text provider switch

The active provider is selected with one environment variable. Groq remains
the default, while OpenAI and Deepgram adapters are ready as production
alternatives:

```env
# Current high-accuracy configuration
STT_PROVIDER=groq
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_WHISPER_FALLBACK_MODEL=whisper-large-v3-turbo
STT_MAX_CONCURRENT_JOBS=1

# Reserve 1: higher-quality OpenAI transcription
# STT_PROVIDER=openai
# OPENAI_API_KEY=...
# OPENAI_STT_MODEL=gpt-4o-transcribe
# OPENAI_STT_TIMESTAMP_MODEL=gpt-4o-transcribe-diarize

# Reserve 2: high-throughput pre-recorded transcription
# STT_PROVIDER=deepgram
# DEEPGRAM_API_KEY=...
# DEEPGRAM_STT_MODEL=nova-3
```

Configured reserve model IDs are also kept in `STT_BACKUP_MODELS` in
`test_whisper.py`. Changing `STT_PROVIDER` and restarting the bot switches the
implementation; no handler or pipeline changes are required. After moving to a
paid high-throughput plan, `STT_MAX_CONCURRENT_JOBS` raises local concurrency
without a code edit (keep it below the provider's own concurrency/rate limit).
If Groq returns a valid but empty result, the bot automatically retries with
language auto-detection and then `GROQ_WHISPER_FALLBACK_MODEL`.

The free Groq quota is suitable for development, not thousands of active
users. Production deployment should use a paid provider plan, persistent job
queue, multiple workers, usage monitoring, retry/backoff, and alerts before
quota exhaustion. The in-process semaphore intentionally handles only one
transcription at a time and must be replaced or distributed before a public
high-volume launch.
