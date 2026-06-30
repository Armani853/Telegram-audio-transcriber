import asyncio
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
import logging
from logging.handlers import RotatingFileHandler
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from urllib.parse import urlparse
from pathlib import Path
from typing import Awaitable, Callable, Optional

from aiohttp import web
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.filters import CommandStart, Command
from aiogram.types import FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
    AsyncGroq,
)
import httpx


# ============================================================
# Configuration
# ============================================================

load_dotenv()


def read_positive_int_env(name: str, default: int) -> int:
    """
    Read a positive integer from environment variables.
    """
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")

    return value


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip()
GROQ_PROXY_URL = os.getenv("GROQ_PROXY_URL", TELEGRAM_PROXY_URL).strip()
TELEGRAM_API_BASE = os.getenv("TELEGRAM_API_BASE", "").strip()
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg").strip() or "ffmpeg"
FFPROBE_BINARY = os.getenv("FFPROBE_BINARY", "ffprobe").strip() or "ffprobe"
AUDIO_CHUNK_BITRATE = os.getenv("AUDIO_CHUNK_BITRATE", "64k").strip() or "64k"
PUBLIC_UPLOAD_BASE_URL = os.getenv("PUBLIC_UPLOAD_BASE_URL", "").strip()
UPLOAD_HOST = os.getenv("UPLOAD_HOST", "0.0.0.0").strip() or "0.0.0.0"
UPLOAD_PORT = read_positive_int_env(
    "PORT",
    read_positive_int_env("UPLOAD_PORT", 8080),
)

# Groq Whisper model
GROQ_WHISPER_MODEL = "whisper-large-v3"

# Forced transcription language
TRANSCRIPTION_LANGUAGE = "ru"

# Telegram message length limit is 4096; keep margin for safety.
TELEGRAM_SAFE_CHUNK_SIZE = 3500
CLOUD_TELEGRAM_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024
LONG_TRANSCRIPTION_TEXT_THRESHOLD = TELEGRAM_SAFE_CHUNK_SIZE * 2

# Keep Groq uploads below the free-tier 25 MB direct upload limit.
MAX_GROQ_CHUNK_BYTES = read_positive_int_env(
    "MAX_GROQ_CHUNK_BYTES",
    20 * 1024 * 1024,
)

# Local Telegram Bot API can download large files; keep a practical safety cap.
MAX_TELEGRAM_INPUT_BYTES = read_positive_int_env(
    "MAX_TELEGRAM_INPUT_BYTES",
    2 * 1024 * 1024 * 1024,
)
MAX_UPLOAD_BYTES = read_positive_int_env(
    "MAX_UPLOAD_BYTES",
    MAX_TELEGRAM_INPUT_BYTES,
)
UPLOAD_CHUNK_BYTES = read_positive_int_env(
    "UPLOAD_CHUNK_BYTES",
    8 * 1024 * 1024,
)
UPLOAD_TOKEN_TTL_SECONDS = read_positive_int_env(
    "UPLOAD_TOKEN_TTL_SECONDS",
    24 * 60 * 60,
)
AUDIO_CHUNK_SECONDS = read_positive_int_env("AUDIO_CHUNK_SECONDS", 15 * 60)
AUDIO_CHUNK_OVERLAP_SECONDS = read_positive_int_env("AUDIO_CHUNK_OVERLAP_SECONDS", 5)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = Path(__file__).with_name("bot_error.log")
SUPPORTED_UPLOAD_SUFFIXES = {".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm"}

router = Router()
groq_transcription_semaphore = asyncio.Semaphore(1)
ProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass
class UploadSession:
    chat_id: int
    created_at: float
    used: bool = False
    chunk_dir: Optional[Path] = None
    file_name: str = ""
    file_size: int = 0
    total_chunks: int = 0
    received_chunks: set[int] = field(default_factory=set)


upload_sessions: dict[str, UploadSession] = {}


# ============================================================
# Utility functions
# ============================================================

def validate_config() -> None:
    """
    Validate required tokens before bot startup.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. Add it to .env or environment variables."
        )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env or environment variables."
        )

    if AUDIO_CHUNK_OVERLAP_SECONDS >= AUDIO_CHUNK_SECONDS:
        raise RuntimeError(
            "AUDIO_CHUNK_OVERLAP_SECONDS must be smaller than AUDIO_CHUNK_SECONDS."
        )

    if UPLOAD_CHUNK_BYTES > MAX_UPLOAD_BYTES:
        raise RuntimeError("UPLOAD_CHUNK_BYTES must be smaller than MAX_UPLOAD_BYTES.")


def get_groq_client() -> AsyncGroq:
    """
    Create the Groq client only after configuration is validated.
    """
    validate_config()
    http_client_kwargs = {
        "trust_env": False,
        "timeout": httpx.Timeout(600.0, connect=30.0),
    }
    groq_proxy_url = get_effective_proxy_url(GROQ_PROXY_URL)
    if groq_proxy_url:
        http_client_kwargs["proxy"] = groq_proxy_url

    return AsyncGroq(
        api_key=GROQ_API_KEY,
        http_client=httpx.AsyncClient(**http_client_kwargs),
    )


def create_bot() -> Bot:
    """
    Create Telegram bot, optionally using TELEGRAM_PROXY_URL and local Bot API.
    """
    validate_config()
    telegram_proxy_url = get_effective_proxy_url(TELEGRAM_PROXY_URL)

    session_kwargs = {}
    if telegram_proxy_url:
        session_kwargs["proxy"] = telegram_proxy_url
    if TELEGRAM_API_BASE:
        session_kwargs["api"] = TelegramAPIServer.from_base(
            TELEGRAM_API_BASE,
            is_local=True,
        )

    if session_kwargs:
        return Bot(token=TELEGRAM_BOT_TOKEN, session=AiohttpSession(**session_kwargs))

    return Bot(token=TELEGRAM_BOT_TOKEN)


def get_effective_proxy_url(proxy_url: str) -> str:
    """
    Ignore local Windows proxy values when the app runs on Linux hosting.
    """
    proxy_url = proxy_url.strip()
    if not proxy_url:
        return ""

    parsed = urlparse(proxy_url)
    host = (parsed.hostname or "").lower()
    is_local_proxy = host in {"127.0.0.1", "localhost", "::1"}

    if is_local_proxy and not sys.platform.startswith("win"):
        logging.warning("Ignoring local proxy %s on non-Windows host.", proxy_url)
        return ""

    return proxy_url


def create_temp_directory(prefix: str) -> Path:
    """
    Create a temporary directory with predictable write permissions.
    """
    base_dir = Path(tempfile.gettempdir())
    for _ in range(10):
        candidate = base_dir / f"{prefix}{uuid.uuid4().hex}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue

    raise RuntimeError("Could not create a temporary directory.")


def configure_logging() -> None:
    """
    Configure console logging once.
    """
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)


async def check_telegram_connection() -> bool:
    """
    Check that the bot can reach Telegram Bot API and the token is valid.
    """
    bot = create_bot()
    try:
        me = await bot.get_me()
        print(f"Telegram: OK (@{me.username})")
        return True
    except TelegramNetworkError as exc:
        print("Telegram: FAILED")
        print(f"Reason: {exc}")
        print(
            "Fix: enable VPN or set TELEGRAM_PROXY_URL in .env, "
            "then run this check again."
        )
        return False
    except Exception as exc:
        print("Telegram: FAILED")
        print(f"Reason: {type(exc).__name__}: {exc}")
        return False
    finally:
        await bot.session.close()


def is_ogg_file_name(file_name: Optional[str]) -> bool:
    """
    Check if a file name looks like an .ogg file.
    """
    if not file_name:
        return False
    return file_name.lower().endswith(".ogg")


def is_supported_audio_message(message: Message) -> bool:
    """
    The bot accepts:
    1. Telegram voice messages.
    2. Audio files with .ogg filename or audio/ogg MIME type.
    3. Document files with .ogg filename or audio/ogg MIME type.

    This covers normal voice messages and forwarded/uploaded .ogg files.
    """
    if message.voice:
        return True

    if message.audio:
        mime_type = message.audio.mime_type or ""
        file_name = message.audio.file_name or ""
        return mime_type in {"audio/ogg", "audio/oga", "audio/opus"} or is_ogg_file_name(file_name)

    if message.document:
        mime_type = message.document.mime_type or ""
        file_name = message.document.file_name or ""
        return mime_type in {"audio/ogg", "audio/oga", "audio/opus"} or is_ogg_file_name(file_name)

    return False


def extract_file_id_and_size(message: Message) -> tuple[str, Optional[int], str]:
    """
    Extract Telegram file_id, file_size, and a safe local filename.
    """
    if message.voice:
        return (
            message.voice.file_id,
            message.voice.file_size,
            f"voice_{message.voice.file_unique_id}.ogg",
        )

    if message.audio:
        file_name = message.audio.file_name or f"audio_{message.audio.file_unique_id}.ogg"
        return message.audio.file_id, message.audio.file_size, file_name

    if message.document:
        file_name = message.document.file_name or f"document_{message.document.file_unique_id}.ogg"
        return message.document.file_id, message.document.file_size, file_name

    raise ValueError("No supported audio file found in the message.")


async def send_long_message(message: Message, text: str) -> None:
    """
    Send long transcription text in chunks because Telegram messages
    cannot be longer than 4096 characters.
    """
    text = text.strip()

    if not text:
        await message.answer("Не удалось распознать речь: результат пустой.")
        return

    for start in range(0, len(text), TELEGRAM_SAFE_CHUNK_SIZE):
        chunk = text[start:start + TELEGRAM_SAFE_CHUNK_SIZE]
        await message.answer(chunk)


async def send_long_text(bot: Bot, chat_id: int, text: str) -> None:
    """
    Send long text in chunks because Telegram messages cannot exceed 4096 chars.
    """
    text = text.strip()

    if not text:
        await bot.send_message(chat_id=chat_id, text="Не удалось распознать речь: результат пустой.")
        return

    for start in range(0, len(text), TELEGRAM_SAFE_CHUNK_SIZE):
        chunk = text[start:start + TELEGRAM_SAFE_CHUNK_SIZE]
        await bot.send_message(chat_id=chat_id, text=chunk)


async def send_transcription_result(bot: Bot, chat_id: int, text: str) -> None:
    """
    Send short transcriptions as messages and long ones as preview plus .txt.
    """
    text = text.strip()

    if not text:
        await bot.send_message(chat_id=chat_id, text="Не удалось распознать речь: результат пустой.")
        return

    if len(text) <= LONG_TRANSCRIPTION_TEXT_THRESHOLD:
        await send_long_text(bot, chat_id, text)
        return

    await bot.send_message(
        chat_id=chat_id,
        text="Расшифровка длинная. Ниже пришлю начало, а полный текст отправлю .txt файлом.",
    )

    preview = text[:LONG_TRANSCRIPTION_TEXT_THRESHOLD].rstrip()
    await send_long_text(bot, chat_id, f"{preview}\n\n[Полный текст в файле ниже]")

    file_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"transcription_{file_stamp}.txt"
    temp_path = Path(tempfile.gettempdir()) / f"transcription_{uuid.uuid4().hex}.txt"

    try:
        temp_path.write_text(text, encoding="utf-8")
        await bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(str(temp_path), filename=file_name),
            caption="Полная расшифровка в текстовом файле.",
        )
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as cleanup_error:
            logging.warning("Could not delete transcription txt file %s: %s", temp_path, cleanup_error)


async def safe_edit_message(message: Message, text: str) -> None:
    """
    Best-effort status update. Telegram can reject edits for old/deleted messages.
    """
    try:
        await message.edit_text(text)
    except Exception as exc:
        logging.warning("Could not edit status message: %s", exc)


async def safe_delete_message(message: Message) -> None:
    """
    Best-effort status cleanup. Failure must not block sending transcription.
    """
    try:
        await message.delete()
    except Exception as exc:
        logging.warning("Could not delete status message: %s", exc)


def cleanup_upload_chunk_dir(session: UploadSession) -> None:
    """
    Remove temporary chunk files for an upload session.
    """
    if session.chunk_dir is None:
        return

    try:
        shutil.rmtree(session.chunk_dir, ignore_errors=True)
    except Exception as exc:
        logging.warning("Could not delete upload chunk dir %s: %s", session.chunk_dir, exc)
    finally:
        session.chunk_dir = None
        session.received_chunks.clear()
        session.file_name = ""
        session.file_size = 0
        session.total_chunks = 0


def cleanup_upload_sessions() -> None:
    """
    Remove expired one-time upload sessions.
    """
    now = time.time()
    expired_ids = [
        upload_id
        for upload_id, session in upload_sessions.items()
        if now - session.created_at > UPLOAD_TOKEN_TTL_SECONDS
    ]
    for upload_id in expired_ids:
        session = upload_sessions.pop(upload_id, None)
        if session is not None:
            cleanup_upload_chunk_dir(session)


def create_upload_session(chat_id: int) -> str:
    """
    Create a one-time upload session bound to a Telegram chat.
    """
    cleanup_upload_sessions()
    upload_id = uuid.uuid4().hex
    upload_sessions[upload_id] = UploadSession(chat_id=chat_id, created_at=time.time())
    return upload_id


def get_public_upload_base_url() -> str:
    """
    Return the public base URL used in Telegram messages.
    """
    if PUBLIC_UPLOAD_BASE_URL:
        return PUBLIC_UPLOAD_BASE_URL.rstrip("/")
    host = "localhost" if UPLOAD_HOST in {"0.0.0.0", "::"} else UPLOAD_HOST
    return f"http://{host}:{UPLOAD_PORT}"


def build_upload_url(upload_id: str) -> str:
    """
    Build a user-facing upload URL.
    """
    return f"{get_public_upload_base_url()}/upload/{upload_id}"


def build_upload_keyboard(upload_url: str) -> InlineKeyboardMarkup:
    """
    Build a one-button keyboard that opens the upload page.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Загрузить длинное аудио",
                    url=upload_url,
                )
            ]
        ]
    )


def create_upload_prompt(chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """
    Create a one-time upload URL and Telegram inline keyboard.
    """
    upload_id = create_upload_session(chat_id)
    upload_url = build_upload_url(upload_id)
    text = (
        "Файл больше 20 МБ, поэтому обычный Telegram Bot API не позволяет мне "
        "скачать его автоматически.\n\n"
        "Нажми кнопку ниже и выбери аудиофайл. После загрузки я сам пришлю "
        "расшифровку сюда."
    )
    return text, build_upload_keyboard(upload_url)


def render_upload_page(upload_id: str, error: str = "") -> web.Response:
    """
    Render a mobile-friendly upload page with resumable chunked upload.
    """
    escaped_upload_id = escape(upload_id)
    error_html = ""
    if error:
        error_html = f'<div class="notice error">{escape(error)}</div>'

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Загрузка аудио</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f6fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #5f6b7a;
      --line: #d9e0ea;
      --accent: #1f7a5b;
      --accent-strong: #155f46;
      --danger: #b42318;
      --danger-bg: #fff1f0;
      --ok-bg: #ecfdf3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, sans-serif;
    }}
    main {{
      width: min(560px, calc(100vw - 28px));
      margin: 28px auto;
      padding: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 24px; line-height: 1.2; }}
    p {{ margin: 0 0 16px; line-height: 1.45; color: var(--muted); }}
    .formats {{ margin: 14px 0; padding: 12px; border: 1px solid var(--line); border-radius: 8px; color: var(--muted); font-size: 14px; }}
    .picker {{
      display: block;
      width: 100%;
      margin: 16px 0 10px;
      padding: 16px;
      border: 1px dashed #9aa8b8;
      border-radius: 8px;
      text-align: center;
      cursor: pointer;
      background: #fbfcfe;
      color: var(--text);
      font-weight: 700;
    }}
    input[type="file"] {{ display: none; }}
    .file-info {{ min-height: 22px; margin: 10px 0 14px; color: var(--muted); font-size: 14px; overflow-wrap: anywhere; }}
    button {{
      width: 100%;
      min-height: 46px;
      border: 0;
      border-radius: 8px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    #uploadButton {{ background: var(--accent); color: white; }}
    #uploadButton:disabled {{ opacity: .55; cursor: not-allowed; }}
    #cancelButton {{ margin-top: 10px; background: #eef2f6; color: var(--text); }}
    .progress-wrap {{ display: none; margin: 16px 0 10px; }}
    .progress-bar {{ width: 100%; height: 12px; overflow: hidden; background: #e8edf3; border-radius: 999px; }}
    .progress-fill {{ width: 0%; height: 100%; background: var(--accent); transition: width .15s ease; }}
    .progress-text {{ margin-top: 8px; color: var(--muted); font-size: 14px; }}
    .notice {{ margin-top: 14px; padding: 12px; border-radius: 8px; line-height: 1.4; }}
    .error {{ background: var(--danger-bg); color: var(--danger); }}
    .success {{ background: var(--ok-bg); color: var(--accent-strong); }}
    .hint {{ margin-top: 14px; font-size: 13px; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Загрузка длинного аудио</h1>
    <p>Выбери аудиофайл. После загрузки страницу можно закрыть, результат придет в Telegram.</p>
    {error_html}
    <div class="formats">
      Форматы: .ogg, .opus, .mp3, .m4a, .wav, .webm<br>
      Максимальный размер: {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ
    </div>
    <form id="uploadForm" action="/upload/{escaped_upload_id}" method="post" enctype="multipart/form-data">
      <label class="picker" for="audioFile">Выбрать аудиофайл</label>
      <input id="audioFile" name="audio_file" type="file" accept=".ogg,.opus,.mp3,.m4a,.wav,.webm,audio/*" required>
      <div id="fileInfo" class="file-info">Файл еще не выбран.</div>
      <button id="uploadButton" type="submit" disabled>Загрузить</button>
      <button id="cancelButton" type="button" hidden>Отменить загрузку</button>
    </form>
    <div id="progressWrap" class="progress-wrap">
      <div class="progress-bar"><div id="progressFill" class="progress-fill"></div></div>
      <div id="progressText" class="progress-text">0%</div>
    </div>
    <div id="status"></div>
    <div class="hint">Ссылка одноразовая и привязана к твоему чату.</div>
  </main>
  <script>
    const uploadId = "{escaped_upload_id}";
    const maxUploadBytes = {MAX_UPLOAD_BYTES};
    const supportedExtensions = [".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm"];
    const fileInput = document.getElementById("audioFile");
    const fileInfo = document.getElementById("fileInfo");
    const form = document.getElementById("uploadForm");
    const uploadButton = document.getElementById("uploadButton");
    const cancelButton = document.getElementById("cancelButton");
    const progressWrap = document.getElementById("progressWrap");
    const progressFill = document.getElementById("progressFill");
    const progressText = document.getElementById("progressText");
    const statusBox = document.getElementById("status");
    let cancelled = false;
    let activeRequest = null;

    function formatBytes(bytes) {{
      if (!bytes) return "0 Б";
      const units = ["Б", "КБ", "МБ", "ГБ"];
      let value = bytes;
      let unit = 0;
      while (value >= 1024 && unit < units.length - 1) {{
        value = value / 1024;
        unit += 1;
      }}
      return `${{value.toFixed(value >= 10 || unit === 0 ? 0 : 1)}} ${{units[unit]}}`;
    }}

    function extensionOf(name) {{
      const dot = name.lastIndexOf(".");
      return dot === -1 ? "" : name.slice(dot).toLowerCase();
    }}

    function showStatus(message, kind) {{
      statusBox.innerHTML = "";
      const notice = document.createElement("div");
      notice.className = `notice ${{kind}}`;
      notice.textContent = message;
      statusBox.appendChild(notice);
    }}

    function setProgress(percent, label) {{
      const rounded = Math.max(0, Math.min(100, Math.round(percent)));
      progressWrap.style.display = "block";
      progressFill.style.width = `${{rounded}}%`;
      progressText.textContent = label || `${{rounded}}%`;
    }}

    function requestJson(url, options) {{
      return fetch(url, options).then(async (response) => {{
        const data = await response.json().catch(() => ({{}}));
        if (!response.ok) {{
          throw new Error(data.error || "Сервер не принял запрос.");
        }}
        return data;
      }});
    }}

    function putChunk(url, blob) {{
      return new Promise((resolve, reject) => {{
        const xhr = new XMLHttpRequest();
        activeRequest = xhr;
        xhr.open("PUT", url);
        xhr.onload = () => {{
          activeRequest = null;
          if (xhr.status >= 200 && xhr.status < 300) {{
            resolve(JSON.parse(xhr.responseText || "{{}}"));
          }} else {{
            try {{
              const data = JSON.parse(xhr.responseText || "{{}}");
              reject(new Error(data.error || "Не удалось загрузить часть файла."));
            }} catch (error) {{
              reject(new Error("Не удалось загрузить часть файла."));
            }}
          }}
        }};
        xhr.onerror = () => {{
          activeRequest = null;
          reject(new Error("Соединение оборвалось. Повтори загрузку, уже загруженные части сохранятся."));
        }};
        xhr.onabort = () => {{
          activeRequest = null;
          reject(new Error("Загрузка отменена."));
        }};
        xhr.send(blob);
      }});
    }}

    fileInput.addEventListener("change", () => {{
      const file = fileInput.files[0];
      statusBox.innerHTML = "";
      progressWrap.style.display = "none";
      uploadButton.disabled = true;

      if (!file) {{
        fileInfo.textContent = "Файл еще не выбран.";
        return;
      }}

      fileInfo.textContent = `${{file.name}} · ${{formatBytes(file.size)}}`;
      const extension = extensionOf(file.name);
      if (!supportedExtensions.includes(extension)) {{
        showStatus("Этот формат не поддерживается. Выбери аудиофайл .ogg, .opus, .mp3, .m4a, .wav или .webm.", "error");
        return;
      }}
      if (file.size > maxUploadBytes) {{
        showStatus(`Файл слишком большой. Максимум: ${{formatBytes(maxUploadBytes)}}.`, "error");
        return;
      }}

      uploadButton.disabled = false;
    }});

    cancelButton.addEventListener("click", () => {{
      cancelled = true;
      if (activeRequest) activeRequest.abort();
      cancelButton.hidden = true;
      uploadButton.disabled = false;
      showStatus("Загрузка отменена. Можно нажать загрузку снова.", "error");
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;

      cancelled = false;
      uploadButton.disabled = true;
      cancelButton.hidden = false;
      statusBox.innerHTML = "";
      setProgress(0, "Подготовка загрузки...");

      try {{
        const init = await requestJson(`/upload/${{uploadId}}/init`, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ file_name: file.name, file_size: file.size }}),
        }});

        const chunkSize = init.chunk_size;
        const totalChunks = init.total_chunks;
        const received = new Set(init.received_chunks || []);

        for (let index = 0; index < totalChunks; index += 1) {{
          if (cancelled) throw new Error("Загрузка отменена.");
          if (!received.has(index)) {{
            const start = index * chunkSize;
            const end = Math.min(file.size, start + chunkSize);
            const blob = file.slice(start, end);
            let uploaded = false;
            for (let attempt = 1; attempt <= 3 && !uploaded; attempt += 1) {{
              try {{
                await putChunk(`/upload/${{uploadId}}/chunk/${{index}}`, blob);
                uploaded = true;
              }} catch (error) {{
                if (attempt === 3) throw error;
                await new Promise((resolve) => setTimeout(resolve, attempt * 900));
              }}
            }}
          }}
          const percent = ((index + 1) / totalChunks) * 100;
          setProgress(percent, `Загрузка: ${{Math.round(percent)}}%`);
        }}

        await requestJson(`/upload/${{uploadId}}/complete`, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{}}),
        }});

        cancelButton.hidden = true;
        setProgress(100, "Загрузка завершена");
        showStatus("Файл получен. Можно закрыть страницу, результат придет в Telegram.", "success");
      }} catch (error) {{
        cancelButton.hidden = true;
        uploadButton.disabled = false;
        showStatus(error.message || "Не удалось загрузить файл.", "error");
      }}
    }});
  </script>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


def render_upload_result_page(title: str, body: str) -> web.Response:
    """
    Render a final upload page.
    """
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f6f8; color: #111827; }}
    main {{ max-width: 520px; margin: 48px auto; padding: 24px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    p {{ line-height: 1.45; }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p>{escape(body)}</p>
  </main>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


def get_active_upload_session(upload_id: str) -> UploadSession:
    """
    Return an upload session or raise a user-safe validation error.
    """
    cleanup_upload_sessions()
    session = upload_sessions.get(upload_id)
    if session is None:
        raise ValueError("Ссылка недействительна. Запроси новую ссылку командой /long.")
    if session.used:
        raise ValueError("Ссылка уже использована. Запроси новую ссылку командой /long.")
    return session


def validate_upload_file_info(file_name: str, file_size: int) -> str:
    """
    Validate uploaded file metadata and return its safe basename.
    """
    safe_name = Path(file_name).name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError(
            "Поддерживаются только аудиофайлы .ogg, .opus, .mp3, .m4a, .wav, .webm."
        )
    if file_size <= 0:
        raise ValueError("Загруженный файл пустой.")
    if file_size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Файл слишком большой. Максимальный размер — {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ."
        )

    return safe_name


def reset_upload_session_chunks(session: UploadSession) -> None:
    """
    Clear previous partial upload data before starting a different file.
    """
    cleanup_upload_chunk_dir(session)


def json_error(message: str, status: int = 400) -> web.Response:
    """
    Return a JSON error response for the upload page JavaScript.
    """
    return web.json_response({"error": message}, status=status)


async def save_uploaded_audio(request: web.Request, destination: Path) -> str:
    """
    Save uploaded multipart audio to disk and return original file name.
    """
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "audio_file" or not field.filename:
        raise ValueError("Файл не найден в форме загрузки.")

    original_name = Path(field.filename).name
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError(
            "Поддерживаются только аудиофайлы .ogg, .opus, .mp3, .m4a, .wav, .webm."
        )

    written_bytes = 0
    with destination.open("wb") as output_file:
        while True:
            chunk = await field.read_chunk(size=1024 * 1024)
            if not chunk:
                break
            written_bytes += len(chunk)
            if written_bytes > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Файл слишком большой. Максимальный размер — {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ."
                )
            output_file.write(chunk)

    if written_bytes == 0:
        raise ValueError("Загруженный файл пустой.")

    return original_name


async def process_uploaded_audio(bot: Bot, chat_id: int, audio_path: Path, original_name: str) -> None:
    """
    Transcribe an uploaded audio file in the background and send text to Telegram.
    """
    status_message: Optional[Message] = None
    started_at = time.monotonic()
    try:
        logging.info(
            "Processing uploaded audio chat_id=%s original_name=%s path=%s size=%s",
            chat_id,
            original_name,
            audio_path,
            audio_path.stat().st_size if audio_path.exists() else None,
        )
        status_message = await bot.send_message(
            chat_id=chat_id,
            text="Файл загружен.\nПодготовка аудио...",
        )

        if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "Файл слишком большой для этого запуска бота. "
                    f"Максимальный размер — {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ."
                ),
            )
            return

        if groq_transcription_semaphore.locked():
            await safe_edit_message(
                status_message,
                "Файл загружен.\nЖду завершения предыдущего распознавания...",
            )

        async with groq_transcription_semaphore:
            await safe_edit_message(
                status_message,
                "Файл получен.\nПодготовка аудио для Groq Whisper...",
            )

            async def update_chunk_progress(index: int, total: int) -> None:
                if status_message is not None:
                    percent = round(index / total * 100)
                    await safe_edit_message(
                        status_message,
                        f"Расшифровка: {percent}%\nЧасть {index} из {total}",
                    )

            if audio_path.stat().st_size > MAX_GROQ_CHUNK_BYTES:
                await safe_edit_message(
                    status_message,
                    "Файл длинный.\nРазбиваю аудио на части для точного распознавания...",
                )

            text = await transcribe_audio_safely(
                audio_path,
                progress_callback=update_chunk_progress,
            )

        logging.info("Uploaded audio transcription finished, text_length=%s", len(text))
        if status_message is not None:
            elapsed_seconds = int(time.monotonic() - started_at)
            await safe_edit_message(
                status_message,
                f"Расшифровка завершена.\nВремя обработки: {elapsed_seconds // 60} мин {elapsed_seconds % 60} сек",
            )
        await send_transcription_result(bot, chat_id, text)
        logging.info("Sent uploaded transcription to chat_id=%s", chat_id)

    except Exception as exc:
        logging.exception("Failed to process uploaded audio: %s", exc)
        reason = describe_processing_error(exc)
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "Не получилось распознать загруженное аудио.\n\n"
                f"Причина: {reason}"
            ),
        )
    finally:
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception as cleanup_error:
            logging.warning("Could not delete uploaded temp file %s: %s", audio_path, cleanup_error)


async def health_handler(request: web.Request) -> web.Response:
    """
    Health endpoint for hosting checks.
    """
    return web.Response(text="OK")


async def upload_init_handler(request: web.Request) -> web.Response:
    """
    Initialize or resume a chunked upload.
    """
    upload_id = request.match_info["upload_id"]

    try:
        session = get_active_upload_session(upload_id)
        payload = await request.json()
        file_name = str(payload.get("file_name", ""))
        file_size = int(payload.get("file_size", 0))
        safe_name = validate_upload_file_info(file_name, file_size)

        same_file = (
            session.chunk_dir is not None
            and session.file_name == safe_name
            and session.file_size == file_size
        )
        if not same_file:
            reset_upload_session_chunks(session)
            session.chunk_dir = create_temp_directory(prefix=f"tg_upload_{upload_id[:8]}_")
            session.file_name = safe_name
            session.file_size = file_size
            session.total_chunks = (file_size + UPLOAD_CHUNK_BYTES - 1) // UPLOAD_CHUNK_BYTES
            session.received_chunks.clear()

        if session.chunk_dir is None:
            raise ValueError("Не удалось подготовить временную папку загрузки.")

        existing_chunks: set[int] = set()
        for index in range(session.total_chunks):
            chunk_path = session.chunk_dir / f"chunk_{index:05d}.part"
            expected_size = min(
                UPLOAD_CHUNK_BYTES,
                session.file_size - index * UPLOAD_CHUNK_BYTES,
            )
            if chunk_path.exists() and chunk_path.stat().st_size == expected_size:
                existing_chunks.add(index)
        session.received_chunks = existing_chunks

        return web.json_response(
            {
                "chunk_size": UPLOAD_CHUNK_BYTES,
                "total_chunks": session.total_chunks,
                "received_chunks": sorted(session.received_chunks),
                "max_upload_bytes": MAX_UPLOAD_BYTES,
            }
        )
    except ValueError as exc:
        return json_error(str(exc))
    except Exception as exc:
        logging.exception("Failed to initialize upload: %s", exc)
        return json_error("Не удалось подготовить загрузку. Запроси новую ссылку командой /long.", 500)


async def upload_chunk_handler(request: web.Request) -> web.Response:
    """
    Accept one chunk of a resumable upload.
    """
    upload_id = request.match_info["upload_id"]

    try:
        chunk_index = int(request.match_info["index"])
        session = get_active_upload_session(upload_id)

        if session.chunk_dir is None or session.total_chunks <= 0:
            raise ValueError("Сначала выбери файл и начни загрузку заново.")
        if chunk_index < 0 or chunk_index >= session.total_chunks:
            raise ValueError("Неверный номер части файла.")

        chunk_path = session.chunk_dir / f"chunk_{chunk_index:05d}.part"
        expected_size = min(
            UPLOAD_CHUNK_BYTES,
            session.file_size - chunk_index * UPLOAD_CHUNK_BYTES,
        )
        if chunk_path.exists() and chunk_path.stat().st_size == expected_size:
            session.received_chunks.add(chunk_index)
            return web.json_response({"received_chunks": sorted(session.received_chunks)})

        written_bytes = 0
        try:
            with chunk_path.open("wb") as output_file:
                async for body_part in request.content.iter_chunked(1024 * 1024):
                    written_bytes += len(body_part)
                    if written_bytes > expected_size:
                        raise ValueError("Полученная часть файла больше ожидаемого размера.")
                    output_file.write(body_part)

            if written_bytes != expected_size:
                raise ValueError("Часть файла загрузилась не полностью. Повтори загрузку.")

            session.received_chunks.add(chunk_index)
        finally:
            if written_bytes != expected_size and chunk_path.exists():
                chunk_path.unlink()

        return web.json_response({"received_chunks": sorted(session.received_chunks)})
    except ValueError as exc:
        return json_error(str(exc))
    except Exception as exc:
        logging.exception("Failed to accept upload chunk: %s", exc)
        return json_error("Не удалось принять часть файла. Повтори загрузку.", 500)


async def upload_complete_handler(request: web.Request) -> web.Response:
    """
    Assemble uploaded chunks and start background transcription.
    """
    upload_id = request.match_info["upload_id"]
    audio_path: Optional[Path] = None

    try:
        session = get_active_upload_session(upload_id)
        if session.chunk_dir is None or session.total_chunks <= 0:
            raise ValueError("Файл еще не загружен.")

        missing_chunks = [
            index
            for index in range(session.total_chunks)
            if index not in session.received_chunks
        ]
        if missing_chunks:
            raise ValueError("Не все части файла загружены. Повтори загрузку.")

        original_name = session.file_name
        suffix = Path(original_name).suffix.lower()
        audio_path = Path(tempfile.gettempdir()) / f"uploaded_audio_{uuid.uuid4().hex}{suffix}"

        with audio_path.open("wb") as output_file:
            for index in range(session.total_chunks):
                chunk_path = session.chunk_dir / f"chunk_{index:05d}.part"
                if not chunk_path.exists():
                    raise ValueError("Не все части файла найдены. Повтори загрузку.")
                with chunk_path.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output_file)

        if not audio_path.exists() or audio_path.stat().st_size != session.file_size:
            raise ValueError("Итоговый файл собрался некорректно. Повтори загрузку.")

        session.used = True
        cleanup_upload_chunk_dir(session)

        bot = request.app["bot"]
        logging.info(
            "Chunked upload accepted upload_id=%s chat_id=%s original_name=%s path=%s",
            upload_id,
            session.chat_id,
            original_name,
            audio_path,
        )
        asyncio.create_task(
            process_uploaded_audio(
                bot=bot,
                chat_id=session.chat_id,
                audio_path=audio_path,
                original_name=original_name,
            )
        )

        return web.json_response({"ok": True})
    except ValueError as exc:
        if audio_path is not None and audio_path.exists():
            audio_path.unlink()
        return json_error(str(exc))
    except Exception as exc:
        logging.exception("Failed to complete chunked upload: %s", exc)
        if audio_path is not None and audio_path.exists():
            audio_path.unlink()
        return json_error("Не удалось собрать файл. Запроси новую ссылку командой /long.", 500)


async def upload_page_handler(request: web.Request) -> web.Response:
    """
    Show a one-time upload page.
    """
    cleanup_upload_sessions()
    upload_id = request.match_info["upload_id"]
    session = upload_sessions.get(upload_id)
    if session is None:
        return render_upload_result_page("Ссылка недействительна", "Запроси новую ссылку командой /long.")
    if session.used:
        return render_upload_result_page("Ссылка уже использована", "Запроси новую ссылку командой /long.")
    return render_upload_page(upload_id)


async def upload_post_handler(request: web.Request) -> web.Response:
    """
    Accept an uploaded audio file and start background transcription.
    """
    cleanup_upload_sessions()
    upload_id = request.match_info["upload_id"]
    session = upload_sessions.get(upload_id)
    if session is None:
        return render_upload_result_page("Ссылка недействительна", "Запроси новую ссылку командой /long.")
    if session.used:
        return render_upload_result_page("Ссылка уже использована", "Запроси новую ссылку командой /long.")

    temp_path: Optional[Path] = None
    try:
        cleanup_upload_chunk_dir(session)
        temp_path = Path(tempfile.gettempdir()) / f"uploaded_audio_{uuid.uuid4().hex}.bin"
        original_name = await save_uploaded_audio(request, temp_path)
        suffix = Path(original_name).suffix.lower()
        audio_path = temp_path.with_suffix(suffix)
        temp_path.replace(audio_path)
        temp_path = None
        session.used = True

        bot = request.app["bot"]
        logging.info(
            "Upload accepted upload_id=%s chat_id=%s original_name=%s path=%s",
            upload_id,
            session.chat_id,
            original_name,
            audio_path,
        )
        asyncio.create_task(
            process_uploaded_audio(
                bot=bot,
                chat_id=session.chat_id,
                audio_path=audio_path,
                original_name=original_name,
            )
        )
        return render_upload_result_page(
            "Файл получен",
            "Можно вернуться в Telegram. Бот пришлет расшифровку после обработки.",
        )
    except ValueError as exc:
        session.used = False
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        return render_upload_page(upload_id, error=str(exc))
    except Exception as exc:
        session.used = False
        logging.exception("Failed to accept upload: %s", exc)
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        return render_upload_result_page(
            "Ошибка загрузки",
            "Не удалось принять файл. Запроси новую ссылку командой /long.",
        )


async def start_upload_server(bot: Bot) -> web.AppRunner:
    """
    Start the embedded HTTP upload server.
    """
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES + 10 * 1024 * 1024)
    app["bot"] = bot
    app.router.add_get("/health", health_handler)
    app.router.add_get("/upload/{upload_id}", upload_page_handler)
    app.router.add_post("/upload/{upload_id}", upload_post_handler)
    app.router.add_post("/upload/{upload_id}/init", upload_init_handler)
    app.router.add_put("/upload/{upload_id}/chunk/{index}", upload_chunk_handler)
    app.router.add_post("/upload/{upload_id}/complete", upload_complete_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, UPLOAD_HOST, UPLOAD_PORT)
    await site.start()
    logging.info("Upload server started on http://%s:%s", UPLOAD_HOST, UPLOAD_PORT)
    logging.info("Public upload base URL: %s", get_public_upload_base_url())
    return runner


async def download_telegram_file(bot: Bot, file_id: str, destination: Path) -> None:
    """
    Download a Telegram file to disk using aiogram Bot API methods.
    """
    telegram_file = await bot.get_file(file_id)
    if not telegram_file.file_path:
        raise RuntimeError("Telegram did not return a valid file path.")

    await bot.download_file(telegram_file.file_path, destination)


async def transcribe_with_groq(audio_path: Path) -> str:
    """
    Send local audio file to Groq Whisper Large V3 and return transcription text.
    The language is forced to Russian for better Russian transcription accuracy.
    """
    groq_client = get_groq_client()

    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/ogg"
    logging.info(
        "Sending audio to Groq: name=%s size=%s mime_type=%s",
        audio_path.name,
        audio_path.stat().st_size,
        mime_type,
    )

    try:
        with audio_path.open("rb") as audio_file:
            transcription = await groq_client.audio.transcriptions.create(
                file=(audio_path.name, audio_file, mime_type),
                model=GROQ_WHISPER_MODEL,
                language=TRANSCRIPTION_LANGUAGE,
                temperature=0,
            )
    finally:
        await groq_client.close()

    # Depending on SDK response format, transcription can be an object with .text
    # or a plain string if response_format="text" is used.
    if isinstance(transcription, str):
        return transcription.strip()

    text = getattr(transcription, "text", "")
    return str(text).strip()


def resolve_executable(command: str) -> str:
    """
    Resolve an executable name or direct path.
    """
    command_path = Path(command)
    if command_path.is_file():
        return str(command_path)

    found_path = shutil.which(command)
    if found_path:
        return found_path

    if sys.platform.startswith("win"):
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_packages.is_dir():
                executable_name = command
                if not executable_name.lower().endswith(".exe"):
                    executable_name = f"{executable_name}.exe"

                matches = sorted(
                    winget_packages.glob(f"**/{executable_name}"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if matches:
                    return str(matches[0])

    return ""


def has_executable(command: str) -> bool:
    """
    Check whether an executable name or direct path is available.
    """
    return bool(resolve_executable(command))


async def run_subprocess(command: list[str]) -> str:
    """
    Run a subprocess and return stdout, raising a readable error on failure.
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if process.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            f"{' '.join(command)}\n"
            f"{stderr_text[-2000:]}"
        )

    return stdout_text


async def get_audio_duration_seconds(audio_path: Path) -> float:
    """
    Return audio duration in seconds using ffprobe.
    """
    ffprobe_path = resolve_executable(FFPROBE_BINARY)
    if not ffprobe_path:
        raise RuntimeError(
            "ffprobe is not installed. Install ffmpeg to process long audio files."
        )

    output = await run_subprocess(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )

    try:
        duration = float(output)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned invalid duration: {output}") from exc

    if duration <= 0:
        raise RuntimeError("Could not detect a positive audio duration.")

    return duration


async def split_audio_with_ffmpeg(audio_path: Path, chunk_dir: Path) -> list[Path]:
    """
    Split long audio into small MP3 chunks that stay below Groq upload limits.
    """
    ffmpeg_path = resolve_executable(FFMPEG_BINARY)
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not installed. Install ffmpeg to process long audio files."
        )

    duration = await get_audio_duration_seconds(audio_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Path] = []
    stride_seconds = AUDIO_CHUNK_SECONDS - AUDIO_CHUNK_OVERLAP_SECONDS
    start_seconds = 0.0
    chunk_index = 1

    while start_seconds < duration:
        remaining_seconds = duration - start_seconds
        chunk_duration = min(float(AUDIO_CHUNK_SECONDS), remaining_seconds)
        if chunk_duration <= 0:
            break

        chunk_path = chunk_dir / f"chunk_{chunk_index:03d}.mp3"
        await run_subprocess(
            [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_seconds:.3f}",
                "-i",
                str(audio_path),
                "-t",
                f"{chunk_duration:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                AUDIO_CHUNK_BITRATE,
                str(chunk_path),
            ]
        )

        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg created an empty chunk: {chunk_path.name}")

        if chunk_path.stat().st_size > MAX_GROQ_CHUNK_BYTES:
            raise RuntimeError(
                f"Audio chunk {chunk_path.name} is too large for Groq "
                f"({chunk_path.stat().st_size} bytes). Reduce AUDIO_CHUNK_SECONDS "
                "or AUDIO_CHUNK_BITRATE."
            )

        chunks.append(chunk_path)
        if start_seconds + chunk_duration >= duration:
            break

        start_seconds += stride_seconds
        chunk_index += 1

    if not chunks:
        raise RuntimeError("ffmpeg did not create any audio chunks.")

    logging.info("Split audio into %s chunks", len(chunks))
    return chunks


def normalized_words(text: str) -> list[str]:
    """
    Normalize words for conservative duplicate detection at chunk boundaries.
    """
    return re.findall(r"[\wёЁ]+", text.lower(), flags=re.UNICODE)


def merge_transcription_parts(parts: list[str]) -> str:
    """
    Merge chunk transcriptions and remove exact repeated word overlap.
    """
    cleaned_parts = [part.strip() for part in parts if part.strip()]
    if not cleaned_parts:
        return ""

    merged = cleaned_parts[0]
    merged_words = normalized_words(merged)

    for part in cleaned_parts[1:]:
        part_words = normalized_words(part)
        overlap_words = 0
        max_overlap = min(40, len(merged_words), len(part_words))

        for candidate in range(max_overlap, 1, -1):
            if merged_words[-candidate:] == part_words[:candidate]:
                overlap_words = candidate
                break

        if overlap_words == 0:
            merged = f"{merged}\n\n{part}"
        else:
            words_to_skip = overlap_words
            split_words = part.split()
            part_without_overlap = " ".join(split_words[words_to_skip:]).strip()
            if part_without_overlap:
                merged = f"{merged} {part_without_overlap}"

        merged_words = normalized_words(merged)

    return merged.strip()


async def transcribe_audio_safely(
    audio_path: Path,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    Transcribe short files directly and long files through ffmpeg chunks.
    """
    audio_size = audio_path.stat().st_size
    if audio_size <= MAX_GROQ_CHUNK_BYTES:
        return await transcribe_with_groq(audio_path)

    chunk_dir = create_temp_directory(prefix="tg_voice_chunks_")
    try:
        chunks = await split_audio_with_ffmpeg(audio_path, chunk_dir)
        transcribed_parts: list[str] = []

        for index, chunk_path in enumerate(chunks, start=1):
            if progress_callback is not None:
                await progress_callback(index, len(chunks))

            logging.info(
                "Transcribing chunk %s/%s name=%s size=%s",
                index,
                len(chunks),
                chunk_path.name,
                chunk_path.stat().st_size,
            )
            transcribed_parts.append(await transcribe_with_groq(chunk_path))

        return merge_transcription_parts(transcribed_parts)
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


def describe_processing_error(exc: Exception) -> str:
    """
    Return a short user-safe explanation for the most common failures.
    """
    if isinstance(exc, TelegramBadRequest) and "file is too big" in str(exc).lower():
        return (
            "обычный Telegram Bot API не может скачать файл больше 20 МБ. "
            "Запусти локальный Telegram Bot API server и укажи TELEGRAM_API_BASE."
        )

    if isinstance(exc, AuthenticationError):
        return "Groq API отклонил ключ. Проверь GROQ_API_KEY в переменных Amvera."

    if isinstance(exc, PermissionDeniedError):
        return "Groq API запретил запрос. Проверь права/доступность ключа Groq."

    if isinstance(exc, RateLimitError):
        return "Groq API временно ограничил запросы. Подожди немного и попробуй снова."

    if isinstance(exc, BadRequestError):
        return "Groq API не принял аудиофайл. Возможно, формат или размер файла не поддержан."

    if isinstance(exc, APITimeoutError):
        return "Groq API слишком долго отвечал. Попробуй голосовое покороче или повтори позже."

    if isinstance(exc, APIConnectionError):
        return "Не удалось подключиться к Groq API с хостинга. Это похоже на сетевую проблему Amvera/Groq."

    if isinstance(exc, APIStatusError):
        return f"Groq API вернул ошибку HTTP {exc.status_code}. Подробности смотри в логах Amvera."

    if isinstance(exc, httpx.HTTPError):
        return "Сетевая ошибка при обращении к внешнему API. Подробности смотри в логах Amvera."

    if isinstance(exc, RuntimeError) and (
        "ffmpeg" in str(exc).lower() or "ffprobe" in str(exc).lower()
    ):
        return f"{exc}"

    return f"Техническая ошибка: {type(exc).__name__}. Подробности смотри в логах Amvera."


def run_environment_check() -> int:
    """
    Check the local environment without sending data to external services.
    """
    checks = [
        ("TELEGRAM_BOT_TOKEN", bool(TELEGRAM_BOT_TOKEN), True),
        ("GROQ_API_KEY", bool(GROQ_API_KEY), True),
        ("TELEGRAM_API_BASE", bool(TELEGRAM_API_BASE), False),
        ("PUBLIC_UPLOAD_BASE_URL", bool(PUBLIC_UPLOAD_BASE_URL), False),
        ("TELEGRAM_PROXY_URL", bool(TELEGRAM_PROXY_URL), False),
        ("GROQ_PROXY_URL", bool(GROQ_PROXY_URL), False),
        ("ffmpeg", has_executable(FFMPEG_BINARY), True),
        ("ffprobe", has_executable(FFPROBE_BINARY), True),
        ("voice.ogg", Path("voice.ogg").is_file(), True),
    ]

    print("Environment check:")
    for name, ok, required in checks:
        if ok:
            status = "OK"
        elif required:
            status = "MISSING"
        else:
            status = "not set (optional)"
        print(f"- {name}: {status}")

    try:
        validate_config()
    except RuntimeError as exc:
        print(f"\nConfiguration error: {exc}")
        return 1

    print("\nLimits:")
    print(
        "- Telegram download mode: "
        + ("local Bot API server" if TELEGRAM_API_BASE else "cloud Bot API (20 MB download limit)")
    )
    print(f"- Max Telegram input: {MAX_TELEGRAM_INPUT_BYTES // (1024 * 1024)} MB")
    print(f"- Max Groq chunk upload: {MAX_GROQ_CHUNK_BYTES // (1024 * 1024)} MB")
    print(f"- Audio chunk length: {AUDIO_CHUNK_SECONDS} seconds")
    print(f"- Audio chunk overlap: {AUDIO_CHUNK_OVERLAP_SECONDS} seconds")
    print(f"- Audio chunk bitrate: {AUDIO_CHUNK_BITRATE}")
    print(f"- Upload server bind: {UPLOAD_HOST}:{UPLOAD_PORT}")
    print(f"- Public upload base URL: {get_public_upload_base_url()}")
    print(f"- Max direct upload: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    print(f"- Browser upload chunk size: {UPLOAD_CHUNK_BYTES // (1024 * 1024)} MB")
    print(f"- Upload token TTL: {UPLOAD_TOKEN_TTL_SECONDS} seconds")

    print("\nLocal checks passed. External APIs are not contacted by --check.")
    return 0


async def run_diagnostics() -> int:
    """
    Check local config and Telegram connectivity.
    """
    config_ok = run_environment_check() == 0
    if not config_ok:
        return 1

    print("\nExternal connectivity:")
    telegram_ok = await check_telegram_connection()
    return 0 if telegram_ok else 1


async def check_groq_connection() -> bool:
    """
    Check that Groq API is reachable and the API key is accepted.
    """
    client = get_groq_client()
    try:
        models = await client.models.list()
        model_ids = {model.id for model in models.data}
        if GROQ_WHISPER_MODEL not in model_ids:
            print(f"Groq: FAILED ({GROQ_WHISPER_MODEL} is not available)")
            return False
        print(f"Groq: OK ({GROQ_WHISPER_MODEL})")
        return True
    except Exception as exc:
        print("Groq: FAILED")
        print(f"Reason: {type(exc).__name__}: {exc}")
        return False
    finally:
        await client.close()


async def run_full_diagnostics() -> int:
    """
    Check local config plus Telegram and Groq connectivity.
    """
    config_ok = run_environment_check() == 0
    if not config_ok:
        return 1

    print("\nExternal connectivity:")
    telegram_ok = await check_telegram_connection()
    groq_ok = await check_groq_connection()
    return 0 if telegram_ok and groq_ok else 1


async def transcribe_local_file(audio_path: Path) -> int:
    """
    Transcribe a local file from the command line.
    """
    if not audio_path.is_file():
        print(f"File not found: {audio_path}")
        return 1

    if audio_path.stat().st_size > MAX_TELEGRAM_INPUT_BYTES:
        print(
            "File is too large. "
            f"Maximum size is {MAX_TELEGRAM_INPUT_BYTES // (1024 * 1024)} MB."
        )
        return 1

    async def print_chunk_progress(index: int, total: int) -> None:
        print(f"Transcribing chunk {index}/{total}...")

    text = await transcribe_audio_safely(
        audio_path,
        progress_callback=print_chunk_progress,
    )
    print(text)
    return 0


# ============================================================
# Bot handlers
# ============================================================

@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    logging.info("Received /start from chat_id=%s", message.chat.id)
    await message.answer(
        "Привет! Отправь мне голосовое сообщение Telegram или .ogg аудиофайл, "
        "и я расшифрую его в текст на русском языке.\n\n"
        "Для длинного файла больше 20 МБ отправь /long и загрузи аудио через кнопку."
    )


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    logging.info("Received /help from chat_id=%s", message.chat.id)
    await message.answer(
        "Как пользоваться:\n"
        "1. Отправь обычное голосовое сообщение.\n"
        "2. Или отправь/перешли .ogg файл.\n"
        "3. Я скачаю аудио, отправлю его в Groq Whisper Large V3, "
        "пришлю текст и удалю временный файл.\n"
        "4. Для длинного файла больше 20 МБ отправь /long и загрузи аудио через кнопку."
    )


@router.message(Command("long"))
async def long_upload_handler(message: Message) -> None:
    logging.info("Received /long from chat_id=%s", message.chat.id)
    text, keyboard = create_upload_prompt(message.chat.id)
    await message.answer(text, reply_markup=keyboard)


@router.message(F.voice | F.audio | F.document)
async def audio_handler(message: Message, bot: Bot) -> None:
    logging.info(
        "Received audio-like message chat_id=%s voice=%s audio=%s document=%s",
        message.chat.id,
        bool(message.voice),
        bool(message.audio),
        bool(message.document),
    )

    if not is_supported_audio_message(message):
        await message.answer(
            "Я могу обработать только голосовые сообщения Telegram или .ogg аудиофайлы."
        )
        return

    local_path: Optional[Path] = None

    try:
        file_id, file_size, original_name = extract_file_id_and_size(message)
        logging.info(
            "Processing file chat_id=%s original_name=%s size=%s",
            message.chat.id,
            original_name,
            file_size,
        )

        if (
            file_size is not None
            and file_size > CLOUD_TELEGRAM_DOWNLOAD_LIMIT_BYTES
            and not TELEGRAM_API_BASE
        ):
            text, keyboard = create_upload_prompt(message.chat.id)
            await message.answer(text, reply_markup=keyboard)
            return

        if file_size is not None and file_size > MAX_TELEGRAM_INPUT_BYTES:
            await message.answer(
                "Файл слишком большой для этого запуска бота. "
                f"Максимальный размер — {MAX_TELEGRAM_INPUT_BYTES // (1024 * 1024)} МБ."
            )
            return

        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        started_at = time.monotonic()
        status_message = await message.answer("Скачиваю аудио...\nПодготовка распознавания...")

        safe_suffix = ".ogg"
        if "." in original_name:
            suffix_candidate = Path(original_name).suffix.lower()
            if suffix_candidate:
                safe_suffix = suffix_candidate

        temp_dir = Path(tempfile.gettempdir())
        local_path = temp_dir / f"tg_voice_{uuid.uuid4().hex}{safe_suffix}"

        await download_telegram_file(bot=bot, file_id=file_id, destination=local_path)
        logging.info("Downloaded Telegram file to %s", local_path)

        if not local_path.exists() or local_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty or missing.")

        if local_path.stat().st_size > MAX_TELEGRAM_INPUT_BYTES:
            await message.answer(
                "Файл слишком большой для этого запуска бота. "
                f"Максимальный размер — {MAX_TELEGRAM_INPUT_BYTES // (1024 * 1024)} МБ."
            )
            return

        if groq_transcription_semaphore.locked():
            await safe_edit_message(
                status_message,
                "Аудио получено.\nЖду завершения предыдущего распознавания...",
            )

        async with groq_transcription_semaphore:
            await safe_edit_message(
                status_message,
                "Аудио получено.\nПодготовка аудио для Groq Whisper...",
            )

            async def update_chunk_progress(index: int, total: int) -> None:
                percent = round(index / total * 100)
                await safe_edit_message(
                    status_message,
                    f"Расшифровка: {percent}%\nЧасть {index} из {total}",
                )

            if local_path.stat().st_size > MAX_GROQ_CHUNK_BYTES:
                await safe_edit_message(
                    status_message,
                    "Файл длинный.\nРазбиваю аудио на части для точного распознавания...",
                )

            text = await transcribe_audio_safely(
                local_path,
                progress_callback=update_chunk_progress,
            )
        logging.info("Groq transcription finished, text_length=%s", len(text))

        elapsed_seconds = int(time.monotonic() - started_at)
        await safe_edit_message(
            status_message,
            f"Расшифровка завершена.\nВремя обработки: {elapsed_seconds // 60} мин {elapsed_seconds % 60} сек",
        )
        await send_transcription_result(bot, message.chat.id, text)
        logging.info("Sent transcription to chat_id=%s", message.chat.id)

    except Exception as exc:
        logging.exception("Failed to process audio message: %s", exc)
        reason = describe_processing_error(exc)
        await message.answer(
            "Не получилось распознать аудио.\n\n"
            f"Причина: {reason}\n\n"
            "Попробуй отправить другое .ogg/голосовое сообщение или пришли мне лог Amvera "
            "со строкой `Failed to process audio message`."
        )

    finally:
        # Always delete local file after processing or failure.
        if local_path is not None:
            try:
                if local_path.exists():
                    local_path.unlink()
            except Exception as cleanup_error:
                logging.warning("Could not delete temp file %s: %s", local_path, cleanup_error)


@router.message()
async def fallback_handler(message: Message) -> None:
    logging.info("Received unsupported message chat_id=%s", message.chat.id)
    await message.answer(
        "Отправь голосовое сообщение Telegram или .ogg аудиофайл, "
        "и я переведу речь в текст. Для длинного файла больше 20 МБ отправь /long."
    )


# ============================================================
# Entrypoint
# ============================================================

async def main() -> None:
    validate_config()
    configure_logging()

    bot = create_bot()
    upload_runner: Optional[web.AppRunner] = None
    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Checking Telegram connection...")
    if TELEGRAM_API_BASE:
        logging.info("Using local Telegram Bot API server: %s", TELEGRAM_API_BASE)
    else:
        logging.info("Using cloud Telegram Bot API; downloads over 20 MB are not available")

    me = await bot.get_me()
    logging.info("Bot connected as @%s", me.username)
    await bot.delete_webhook(drop_pending_updates=False)
    upload_runner = await start_upload_server(bot)
    logging.info("Polling started. Press Ctrl+C to stop.")

    try:
        await dp.start_polling(bot)
    finally:
        if upload_runner is not None:
            await upload_runner.cleanup()
        await bot.session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telegram bot that transcribes Russian .ogg voice messages via Groq Whisper."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check local configuration without contacting Telegram or Groq",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="check local configuration and Telegram connectivity",
    )
    parser.add_argument(
        "--diagnose-full",
        action="store_true",
        help="check local configuration plus Telegram and Groq connectivity",
    )
    parser.add_argument(
        "--transcribe",
        type=Path,
        metavar="PATH",
        help="transcribe a local audio file via Groq and print the result",
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(run_environment_check())

    if args.diagnose:
        raise SystemExit(asyncio.run(run_diagnostics()))

    if args.diagnose_full:
        raise SystemExit(asyncio.run(run_full_diagnostics()))

    if args.transcribe:
        raise SystemExit(asyncio.run(transcribe_local_file(args.transcribe)))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped.")
