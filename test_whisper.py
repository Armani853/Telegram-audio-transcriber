import asyncio
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
import json
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
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message
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

from media_platform import (
    MediaPlatform,
    media_platform_config_from_env,
    register_media_platform_routes,
)
from youtube_downloader import (
    QUALITY_AUDIO,
    YouTubeDownloadNotification,
    YouTubeDownloadRecord,
    YouTubeDownloadRequest,
    YouTubeDownloadService,
    format_duration as format_download_duration,
    format_file_size,
    register_youtube_download_routes,
    youtube_download_config_from_env,
)


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
YTDLP_BINARY = os.getenv("YTDLP_BINARY", "yt-dlp").strip() or "yt-dlp"
YTDLP_JS_RUNTIME = os.getenv("YTDLP_JS_RUNTIME", "deno").strip() or "deno"
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "").strip()
YTDLP_PROXY_URL = os.getenv("YTDLP_PROXY_URL", TELEGRAM_PROXY_URL).strip()
YTDLP_EXTRACTOR_ARGS = (
    os.getenv(
        "YTDLP_EXTRACTOR_ARGS",
        "youtube:player_client=default,-android_sdkless",
    ).strip()
    or "youtube:player_client=default,-android_sdkless"
)
AUDIO_CHUNK_BITRATE = os.getenv("AUDIO_CHUNK_BITRATE", "64k").strip() or "64k"
PUBLIC_UPLOAD_BASE_URL = os.getenv("PUBLIC_UPLOAD_BASE_URL", "").strip()
UPLOAD_HOST = os.getenv("UPLOAD_HOST", "0.0.0.0").strip() or "0.0.0.0"
UPLOAD_PORT = read_positive_int_env(
    "PORT",
    read_positive_int_env("UPLOAD_PORT", 8080),
)

# Groq Whisper model
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3").strip() or "whisper-large-v3"
STT_PROVIDER = os.getenv("STT_PROVIDER", "groq").strip().lower() or "groq"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe").strip() or "gpt-4o-transcribe"
OPENAI_STT_TIMESTAMP_MODEL = (
    os.getenv("OPENAI_STT_TIMESTAMP_MODEL", "gpt-4o-transcribe-diarize").strip()
    or "gpt-4o-transcribe-diarize"
)
OPENAI_PROXY_URL = os.getenv("OPENAI_PROXY_URL", GROQ_PROXY_URL).strip()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3"
DEEPGRAM_PROXY_URL = os.getenv("DEEPGRAM_PROXY_URL", GROQ_PROXY_URL).strip()
STT_BACKUP_MODELS = (
    "deepgram:nova-3",
    "openai:gpt-4o-transcribe",
    "openai:gpt-4o-transcribe-diarize",
    "groq:whisper-large-v3-turbo",
)
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
ENABLE_GROQ_LLM_POSTPROCESSING = os.getenv("ENABLE_GROQ_LLM_POSTPROCESSING", "1").strip() not in {"0", "false", "False"}

# Default mode is exactly the old Russian-only behavior: language="ru".
DEFAULT_TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_LANGUAGE", "ru").strip().lower() or "ru"
SUPPORTED_TRANSCRIPTION_MODES = {"ru", "en", "es", "hy", "auto"}

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
TRANSCRIPTION_TAIL_RECOVERY_SECONDS = read_positive_int_env("TRANSCRIPTION_TAIL_RECOVERY_SECONDS", 45)
TRANSCRIPTION_TAIL_RECOVERY_MIN_DURATION_SECONDS = read_positive_int_env(
    "TRANSCRIPTION_TAIL_RECOVERY_MIN_DURATION_SECONDS",
    60,
)
YOUTUBE_MAX_DURATION_SECONDS = read_positive_int_env("YOUTUBE_MAX_DURATION_SECONDS", 12 * 60 * 60)
YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS = read_positive_int_env("YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS", 2 * 60 * 60)
YOUTUBE_AUDIO_FORMAT = os.getenv("YOUTUBE_AUDIO_FORMAT", "mp3").strip().lower() or "mp3"
YOUTUBE_MAX_ACTIVE_JOBS_PER_CHAT = read_positive_int_env("YOUTUBE_MAX_ACTIVE_JOBS_PER_CHAT", 1)
STT_MAX_CONCURRENT_JOBS = read_positive_int_env("STT_MAX_CONCURRENT_JOBS", 1)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = Path(__file__).with_name("bot_error.log")
SUPPORTED_UPLOAD_SUFFIXES = {".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm"}
SUPPORTED_DIRECT_AUDIO_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".amr",
    ".flac",
    ".m4a",
    ".m4b",
    ".mka",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
    ".weba",
    ".wma",
}
SUPPORTED_DIRECT_VIDEO_SUFFIXES = {
    ".3g2",
    ".3gp",
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ts",
    ".webm",
    ".wmv",
}
SUPPORTED_DIRECT_MEDIA_SUFFIXES = SUPPORTED_DIRECT_AUDIO_SUFFIXES | SUPPORTED_DIRECT_VIDEO_SUFFIXES
GROQ_DIRECT_AUDIO_SUFFIXES = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

router = Router()
groq_transcription_semaphore = asyncio.Semaphore(STT_MAX_CONCURRENT_JOBS)
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
media_platform: Optional[MediaPlatform] = None
youtube_download_service: Optional[YouTubeDownloadService] = None
chat_transcription_modes: dict[int, str] = {}
youtube_active_jobs_by_chat: dict[int, int] = {}
youtube_download_waiting_chats: set[int] = set()
youtube_download_active_chats: set[int] = set()


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

    if STT_PROVIDER not in {"groq", "openai", "deepgram"}:
        raise RuntimeError("STT_PROVIDER must be one of: groq, openai, deepgram.")

    if STT_PROVIDER == "groq" and not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env or environment variables."
        )
    if STT_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing for STT_PROVIDER=openai.")
    if STT_PROVIDER == "deepgram" and not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is missing for STT_PROVIDER=deepgram.")

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


def suffix_from_file_name(file_name: Optional[str]) -> str:
    """
    Return a normalized suffix from a Telegram-provided file name.
    """
    if not file_name:
        return ""
    return Path(file_name).suffix.lower()


def is_supported_media_name(file_name: Optional[str]) -> bool:
    """
    Check if a file name has an audio/video suffix that ffmpeg can usually read.
    """
    return suffix_from_file_name(file_name) in SUPPORTED_DIRECT_MEDIA_SUFFIXES


def is_supported_media_mime(mime_type: Optional[str]) -> bool:
    """
    Accept Telegram media by broad MIME family when Telegram provides one.
    """
    normalized = (mime_type or "").lower()
    return normalized.startswith("audio/") or normalized.startswith("video/")


def should_prepare_media_with_ffmpeg(message: Message, file_name: str) -> bool:
    """
    Keep the old direct path for Telegram voice/ogg-style audio. Use ffmpeg only
    when we need to extract video audio or convert a container Groq may reject.
    """
    if message.video_note or message.video:
        return True

    suffix = suffix_from_file_name(file_name)
    if message.audio and suffix in GROQ_DIRECT_AUDIO_SUFFIXES:
        return False

    if suffix in SUPPORTED_DIRECT_VIDEO_SUFFIXES:
        return True

    return suffix not in GROQ_DIRECT_AUDIO_SUFFIXES


def is_supported_audio_message(message: Message) -> bool:
    """
    The bot accepts:
    1. Telegram voice messages and video notes.
    2. Telegram audio/video messages.
    3. Document files that Telegram marks as audio/video or that have a
       common audio/video suffix.
    """
    if message.voice or message.video_note:
        return True

    if message.audio:
        return True

    if message.video:
        return True

    if message.document:
        mime_type = message.document.mime_type or ""
        file_name = message.document.file_name or ""
        return is_supported_media_mime(mime_type) or is_supported_media_name(file_name)

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

    if message.video_note:
        return (
            message.video_note.file_id,
            message.video_note.file_size,
            f"video_note_{message.video_note.file_unique_id}.mp4",
        )

    if message.audio:
        file_name = message.audio.file_name or f"audio_{message.audio.file_unique_id}.ogg"
        return message.audio.file_id, message.audio.file_size, file_name

    if message.video:
        file_name = message.video.file_name or f"video_{message.video.file_unique_id}.mp4"
        return message.video.file_id, message.video.file_size, file_name

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


ARMENIAN_TRANSLITERATION_TABLE = str.maketrans(
    {
        "Ա": "A", "ա": "a",
        "Բ": "B", "բ": "b",
        "Գ": "G", "գ": "g",
        "Դ": "D", "դ": "d",
        "Ե": "Ye", "ե": "e",
        "Զ": "Z", "զ": "z",
        "Է": "E", "է": "e",
        "Ը": "Y", "ը": "y",
        "Թ": "T", "թ": "t",
        "Ժ": "Zh", "ժ": "zh",
        "Ի": "I", "ի": "i",
        "Լ": "L", "լ": "l",
        "Խ": "Kh", "խ": "kh",
        "Ծ": "Ts", "ծ": "ts",
        "Կ": "K", "կ": "k",
        "Հ": "H", "հ": "h",
        "Ձ": "Dz", "ձ": "dz",
        "Ղ": "Gh", "ղ": "gh",
        "Ճ": "Ch", "ճ": "ch",
        "Մ": "M", "մ": "m",
        "Յ": "Y", "յ": "y",
        "Ն": "N", "ն": "n",
        "Շ": "Sh", "շ": "sh",
        "Ո": "Vo", "ո": "o",
        "Չ": "Ch", "չ": "ch",
        "Պ": "P", "պ": "p",
        "Ջ": "J", "ջ": "j",
        "Ռ": "R", "ռ": "r",
        "Ս": "S", "ս": "s",
        "Վ": "V", "վ": "v",
        "Տ": "T", "տ": "t",
        "Ր": "R", "ր": "r",
        "Ց": "Ts", "ց": "ts",
        "Ւ": "W", "ւ": "w",
        "Փ": "P", "փ": "p",
        "Ք": "Q", "ք": "q",
        "Օ": "O", "օ": "o",
        "Ֆ": "F", "ֆ": "f",
        "և": "ev",
        "։": ".",
        "՝": "'",
        "՛": "'",
        "՞": "?",
        "՜": "!",
        "«": "\"",
        "»": "\"",
    }
)


def transliterate_armenian_to_latin(text: str) -> str:
    """
    Convert Armenian script to a readable Latin-letter approximation.
    """
    text = re.sub(r"Ու", "U", text)
    text = re.sub(r"ՈՒ", "U", text)
    text = re.sub(r"ու", "u", text)
    text = re.sub(r"\bԵ", "Ye", text)
    text = re.sub(r"\bե", "ye", text)
    text = re.sub(r"\bՈ", "Vo", text)
    text = re.sub(r"\bո", "vo", text)
    return text.translate(ARMENIAN_TRANSLITERATION_TABLE)


def split_text_for_llm(text: str, max_chars: int = 6000) -> list[str]:
    """
    Split long text into LLM-friendly chunks on paragraph/sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    parts = re.split(r"(\n\n+|(?<=[.!?։])\s+)", text)
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current.strip():
            chunks.append(current.strip())
        if len(part) <= max_chars:
            current = part
        else:
            chunks.extend(part[i:i + max_chars] for i in range(0, len(part), max_chars))
            current = ""

    if current.strip():
        chunks.append(current.strip())
    return chunks


async def translate_text_to_armenian(text: str) -> str:
    """
    Translate transcript text to Armenian while preserving meaning and tone.
    """
    chunks = split_text_for_llm(text)
    if not chunks:
        return ""

    translated_parts: list[str] = []
    groq_client = get_groq_client()
    try:
        for chunk in chunks:
            response = await groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's transcript into natural Armenian accurately and literally enough to preserve every fact. "
                            "The transcript may be unpunctuated speech; first infer sentence boundaries and punctuation from context, then translate. "
                            "If the transcript is already Armenian, keep it Armenian and only lightly normalize obvious recognition noise. "
                            "Preserve meaning, verbs, negations, profanity, slang, names, numbers, and the informal spoken style. "
                            "Do not summarize, soften, omit, or replace actions with different actions. "
                            "For example, 'said' means 'ասաց', not 'տվեց'. "
                            "Return only the Armenian text, with no explanations."
                        ),
                    },
                    {"role": "user", "content": chunk},
                ],
            )
            translated_parts.append((response.choices[0].message.content or "").strip())
    finally:
        await groq_client.close()

    return "\n\n".join(part for part in translated_parts if part).strip()


async def format_transcription_for_mode(text: str, mode: str) -> str:
    """
    Add mode-specific presentation while preserving the raw transcription.
    """
    if normalize_transcription_mode(mode) != "hy":
        return text

    try:
        armenian_text = await translate_text_to_armenian(text)
    except Exception as exc:
        logging.warning("Could not translate transcript to Armenian: %s", exc)
        armenian_text = text

    armenian_text = armenian_text.strip()
    latin_text = transliterate_armenian_to_latin(armenian_text).strip()
    if not armenian_text:
        return text

    return (
        "Original:\n"
        f"{text.strip()}\n\n"
        "Armenian:\n"
        f"{armenian_text}\n\n"
        "Latin letters:\n"
        f"{latin_text}"
    )


async def send_transcription_result(bot: Bot, chat_id: int, text: str, mode: str = "ru") -> None:
    """
    Send short transcriptions as messages and long ones as preview plus .txt.
    """
    text = await format_transcription_for_mode(text.strip(), mode)
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


def format_elapsed(seconds: int) -> str:
    """
    Format elapsed seconds for compact Telegram status cards.
    """
    minutes = seconds // 60
    rest = seconds % 60
    if minutes:
        return f"{minutes} мин {rest} сек"
    return f"{rest} сек"


def progress_bar(percent: int, width: int = 12) -> str:
    """
    Render a compact text progress bar for Telegram messages.
    """
    safe_percent = max(0, min(100, percent))
    filled = round(width * safe_percent / 100)
    return "█" * filled + "░" * (width - filled)


def render_processing_status(
    title: str,
    percent: int,
    stage: str,
    detail: str = "",
    elapsed_seconds: Optional[int] = None,
) -> str:
    """
    Build a visually pleasant single-message progress card.
    """
    lines = [
        title,
        f"{progress_bar(percent)} {max(0, min(100, percent))}%",
        "",
        stage,
    ]
    if detail:
        lines.append(detail)
    if elapsed_seconds is not None:
        lines.extend(["", f"Время: {format_elapsed(elapsed_seconds)}"])
    return "\n".join(lines)


async def safe_edit_message(
    message: Message,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
    parse_mode: Optional[str] = None,
) -> bool:
    """
    Best-effort status update. Telegram can reject edits for old/deleted messages.
    """
    try:
        kwargs = {}
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
        await message.edit_text(text, **kwargs)
        return True
    except Exception as exc:
        logging.warning("Could not edit status message: %s", exc)
        return False


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


def normalize_transcription_mode(mode: str) -> str:
    """
    Normalize a requested transcription mode and fall back to Russian.
    """
    normalized = (mode or "").strip().lower()
    if normalized in {"eng", "english"}:
        return "en"
    if normalized in {"esp", "spanish", "espanol", "español"}:
        return "es"
    if normalized in {"arm", "armenian", "hay", "hayeren", "հայերեն"}:
        return "hy"
    if normalized in {"detect", "auto-detect", "autodetect"}:
        return "auto"
    if normalized in SUPPORTED_TRANSCRIPTION_MODES:
        return normalized
    return "ru"


def get_chat_transcription_mode(chat_id: int) -> str:
    """
    Return the current chat mode. Russian is the default and old behavior.
    """
    return chat_transcription_modes.get(
        chat_id,
        normalize_transcription_mode(DEFAULT_TRANSCRIPTION_MODE),
    )


def transcription_mode_label(mode: str) -> str:
    labels = {
        "ru": "Русский",
        "en": "English",
        "es": "Español",
        "hy": "Armenian",
        "auto": "Auto",
    }
    return labels.get(normalize_transcription_mode(mode), "Русский")


def build_transcription_mode_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    """
    Build language mode buttons for direct chat transcription.
    """
    current_mode = get_chat_transcription_mode(chat_id)

    def button(mode: str, text: str) -> InlineKeyboardButton:
        prefix = "✓ " if current_mode == mode else ""
        return InlineKeyboardButton(
            text=f"{prefix}{text}",
            callback_data=f"transcription_mode:{mode}",
        )

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                button("ru", "Русский"),
                button("en", "English"),
                button("es", "Español"),
                button("hy", "Armenian"),
                button("auto", "Auto"),
            ]
        ]
    )


def build_main_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    """
    Build the main Telegram keyboard for the bot entrypoint.
    """
    buttons = [
        [
            InlineKeyboardButton(
                text="🎬 Расшифровать YouTube",
                callback_data="youtube:help",
            )
        ],
        [
            InlineKeyboardButton(
                text="📥 Скачать видео с YouTube",
                callback_data="youtube_download:help",
            )
        ],
        [
            InlineKeyboardButton(
                text="📚 Медиатека",
                url=build_library_url(chat_id),
            ),
            InlineKeyboardButton(
                text=f"🌐 {transcription_mode_label(get_chat_transcription_mode(chat_id))}",
                callback_data="transcription_mode:menu",
            ),
        ]
    ]
    upload_id = create_upload_session(chat_id)
    buttons.append(
        [
            InlineKeyboardButton(
                text="📤 Загрузить большой файл",
                url=build_upload_url(upload_id),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def youtube_download_quality_label(request: YouTubeDownloadRequest, quality: str) -> str:
    size = request.estimated_sizes.get(quality, 0)
    size_hint = f" · ~{format_file_size(size)}" if size > 0 else ""
    if quality == QUALITY_AUDIO:
        return f"🎧 Только аудио · MP3{size_hint}"
    labels = {
        "360": "⚡ 360p · минимальный размер",
        "720": "✨ 720p · хорошее качество",
        "1080": "💎 1080p · максимальное качество",
    }
    return f"{labels.get(quality, quality + 'p')}{size_hint}"


def build_youtube_download_quality_keyboard(
    request: YouTubeDownloadRequest,
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=youtube_download_quality_label(request, quality),
                callback_data=f"youtube_download:select:{request.request_id}:{quality}",
            )
        ]
        for quality in request.available_qualities
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_library_url(chat_id: int) -> str:
    """
    Build a user-facing media library URL.
    """
    if media_platform is not None:
        return media_platform.library_url(chat_id)
    return f"{get_public_upload_base_url()}/app?chat_id={chat_id}"


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
            text=render_processing_status(
                "Расшифровка файла",
                15,
                "Файл загружен",
                "Проверяю размер и готовлю очередь.",
            ),
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
                render_processing_status(
                    "Расшифровка файла",
                    25,
                    "Файл принят",
                    "Жду завершения предыдущего распознавания.",
                    int(time.monotonic() - started_at),
                ),
            )

        async with groq_transcription_semaphore:
            await safe_edit_message(
                status_message,
                render_processing_status(
                    "Расшифровка файла",
                    40,
                    "Подготовка аудио",
                    "Собираю файл и готовлю отправку в Groq Whisper.",
                    int(time.monotonic() - started_at),
                ),
            )

            async def update_chunk_progress(index: int, total: int) -> None:
                if status_message is not None:
                    percent = 55 + round(index / total * 35)
                    await safe_edit_message(
                        status_message,
                        render_processing_status(
                            "Расшифровка файла",
                            percent,
                            "Распознаю речь",
                            f"Фрагмент {index} из {total}.",
                            int(time.monotonic() - started_at),
                        ),
                    )

            if audio_path.stat().st_size > MAX_GROQ_CHUNK_BYTES:
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка файла",
                        50,
                        "Файл длинный",
                        "Разбиваю аудио на части для точного распознавания.",
                        int(time.monotonic() - started_at),
                    ),
                )
            else:
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка файла",
                        70,
                        "Распознаю речь",
                        "Отправил аудио в Groq Whisper.",
                        int(time.monotonic() - started_at),
                    ),
                )

            text = await transcribe_audio_safely(
                audio_path,
                mode=get_chat_transcription_mode(chat_id),
                progress_callback=update_chunk_progress,
            )

        logging.info("Uploaded audio transcription finished, text_length=%s", len(text))
        if status_message is not None:
            elapsed_seconds = int(time.monotonic() - started_at)
            await safe_edit_message(
                status_message,
                render_processing_status(
                    "Расшифровка готова",
                    100,
                    "Готово",
                    "Сейчас пришлю текст.",
                    elapsed_seconds,
                ),
            )
        await send_transcription_result(bot, chat_id, text, mode=get_chat_transcription_mode(chat_id))
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


async def postprocess_transcript_with_groq(
    full_text: str,
    transcript_rows: list[dict],
    duration: float,
) -> Optional[dict]:
    """
    Ask Groq LLM for chapters, summary, key points, and action items.
    """
    if not ENABLE_GROQ_LLM_POSTPROCESSING:
        return None

    trimmed_text = full_text[:30000]
    timeline = [
        {
            "start_time": int(row.get("start_time", 0)),
            "end_time": int(row.get("end_time", 0)),
            "text": str(row.get("text", ""))[:1200],
        }
        for row in transcript_rows[:30]
    ]
    prompt = {
        "language": "ru",
        "duration_seconds": int(duration),
        "timeline": timeline,
        "transcript": trimmed_text,
        "required_json_schema": {
            "chapters": [
                {
                    "title": "string",
                    "start_time": "number seconds",
                    "end_time": "number seconds",
                    "description": "string",
                }
            ],
            "short_summary": "string",
            "detailed_summary": "string",
            "key_points": ["string"],
            "tasks": [
                {
                    "task_text": "string",
                    "timestamp": "number seconds",
                    "assignee": "string or null",
                    "due_date": "string or null",
                    "context": "string",
                }
            ],
        },
    }

    groq_client = get_groq_client()
    response = await groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты анализируешь русские расшифровки лекций, встреч и подкастов. "
                    "Верни только валидный JSON по схеме пользователя. "
                    "Таймкоды указывай в секундах."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        return None
    return parsed


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
    global media_platform, youtube_download_service
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES + 10 * 1024 * 1024)
    app["bot"] = bot
    media_config = media_platform_config_from_env(
        public_base_url=get_public_upload_base_url(),
        max_upload_bytes=MAX_UPLOAD_BYTES,
        upload_chunk_bytes=UPLOAD_CHUNK_BYTES,
        ffmpeg_binary=resolve_executable(FFMPEG_BINARY) or FFMPEG_BINARY,
        max_groq_chunk_bytes=MAX_GROQ_CHUNK_BYTES,
        audio_chunk_seconds=AUDIO_CHUNK_SECONDS,
        audio_chunk_overlap_seconds=AUDIO_CHUNK_OVERLAP_SECONDS,
    )
    media_platform = MediaPlatform(
        bot=bot,
        config=media_config,
        run_command=run_subprocess,
        get_duration=get_audio_duration_seconds,
        split_audio=split_audio_with_ffmpeg,
        transcribe_one=transcribe_with_selected_provider,
        merge_transcripts=merge_transcription_parts,
        create_temp_directory=create_temp_directory,
        describe_error=describe_processing_error,
        postprocess_transcript=postprocess_transcript_with_groq,
    )
    youtube_download_service = YouTubeDownloadService(
        config=youtube_download_config_from_env(
            public_base_url=get_public_upload_base_url(),
            default_data_dir=media_config.db_path.parent,
        ),
        run_command=run_subprocess,
        base_command=youtube_ytdlp_base_command,
        ffprobe_binary=resolve_executable(FFPROBE_BINARY) or FFPROBE_BINARY,
        fallback_base_command=lambda: youtube_ytdlp_base_command(use_proxy=False),
    )
    app.router.add_get("/health", health_handler)
    app.router.add_get("/upload/{upload_id}", upload_page_handler)
    app.router.add_post("/upload/{upload_id}", upload_post_handler)
    app.router.add_post("/upload/{upload_id}/init", upload_init_handler)
    app.router.add_put("/upload/{upload_id}/chunk/{index}", upload_chunk_handler)
    app.router.add_post("/upload/{upload_id}/complete", upload_complete_handler)
    register_media_platform_routes(app, media_platform)
    register_youtube_download_routes(app, youtube_download_service)

    notification_task: Optional[asyncio.Task] = None

    async def start_notification_delivery(_: web.Application) -> None:
        nonlocal notification_task
        notification_task = asyncio.create_task(
            youtube_download_notification_loop(bot, youtube_download_service)
        )

    async def stop_notification_delivery(_: web.Application) -> None:
        if notification_task is None:
            return
        notification_task.cancel()
        try:
            await notification_task
        except asyncio.CancelledError:
            pass

    # The service startup is already registered above. Start delivery after it,
    # and stop delivery before closing the service/database.
    app.on_startup.append(start_notification_delivery)
    app.on_cleanup.insert(0, stop_notification_delivery)

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


async def transcribe_with_groq(audio_path: Path, mode: str = "ru") -> str:
    """
    Send local audio file to Groq Whisper Large V3 and return transcription text.
    Russian mode intentionally matches the old behavior: language="ru" and no
    prompt. Armenian mode uses auto transcription first, then translation.
    """
    groq_client = get_groq_client()

    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/ogg"
    mode = normalize_transcription_mode(mode)
    transcription_kwargs = {
        "file": None,
        "model": GROQ_WHISPER_MODEL,
        "temperature": 0,
    }
    if mode in {"ru", "en", "es"}:
        transcription_kwargs["language"] = mode

    logging.info(
        "Sending audio to Groq: name=%s size=%s mime_type=%s mode=%s",
        audio_path.name,
        audio_path.stat().st_size,
        mime_type,
        mode,
    )

    try:
        with audio_path.open("rb") as audio_file:
            transcription_kwargs["file"] = (audio_path.name, audio_file, mime_type)
            transcription = await create_groq_transcription(groq_client, transcription_kwargs)
    finally:
        await groq_client.close()

    # Depending on SDK response format, transcription can be an object with .text
    # or a plain string if response_format="text" is used.
    if isinstance(transcription, str):
        return transcription.strip()

    text = getattr(transcription, "text", "")
    return str(text).strip()


def transcription_segment_value(segment: object, key: str, default: object = None) -> object:
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


async def create_groq_transcription(groq_client: AsyncGroq, kwargs: dict) -> object:
    """Create a transcription and log non-sensitive quota headers when available."""
    transcriptions = groq_client.audio.transcriptions
    raw_resource = getattr(transcriptions, "with_raw_response", None)
    if raw_resource is None:
        return await transcriptions.create(**kwargs)

    raw_response = await raw_resource.create(**kwargs)
    quota_headers = {
        key.lower(): value
        for key, value in raw_response.headers.items()
        if key.lower().startswith("x-ratelimit") or key.lower() == "retry-after"
    }
    if quota_headers:
        logging.info("Groq STT rate limits: %s", quota_headers)
    return raw_response.parse()


async def transcribe_with_groq_segments(audio_path: Path, mode: str = "auto") -> list[dict]:
    """
    Transcribe one audio file and return Whisper segment timestamps.
    """
    groq_client = get_groq_client()
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/ogg"
    mode = normalize_transcription_mode(mode)
    transcription_kwargs = {
        "file": None,
        "model": GROQ_WHISPER_MODEL,
        "temperature": 0,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if mode in {"ru", "en", "es"}:
        transcription_kwargs["language"] = mode

    try:
        with audio_path.open("rb") as audio_file:
            transcription_kwargs["file"] = (audio_path.name, audio_file, mime_type)
            transcription = await create_groq_transcription(groq_client, transcription_kwargs)
    finally:
        await groq_client.close()

    raw_segments = (
        transcription.get("segments", [])
        if isinstance(transcription, dict)
        else getattr(transcription, "segments", [])
    )
    result: list[dict] = []
    for segment in raw_segments or []:
        text = re.sub(r"\s+", " ", str(transcription_segment_value(segment, "text", ""))).strip()
        if not text:
            continue
        try:
            start = max(0.0, float(transcription_segment_value(segment, "start", 0) or 0))
            end = max(start, float(transcription_segment_value(segment, "end", start) or start))
        except (TypeError, ValueError):
            continue
        result.append({"start": start, "end": end, "text": text})

    if not result:
        fallback_text = (
            transcription.get("text", "")
            if isinstance(transcription, dict)
            else getattr(transcription, "text", "")
        )
        fallback_text = re.sub(r"\s+", " ", str(fallback_text)).strip()
        if fallback_text:
            result.append({"start": 0.0, "end": 0.0, "text": fallback_text})
    return result


def active_stt_model_name() -> str:
    if STT_PROVIDER == "openai":
        return OPENAI_STT_MODEL
    if STT_PROVIDER == "deepgram":
        return DEEPGRAM_STT_MODEL
    return GROQ_WHISPER_MODEL


def transcription_http_client_kwargs(proxy_url: str) -> dict:
    kwargs: dict = {
        "timeout": httpx.Timeout(600.0, connect=30.0),
        "trust_env": False,
    }
    effective_proxy = get_effective_proxy_url(proxy_url)
    if effective_proxy:
        kwargs["proxy"] = effective_proxy
    return kwargs


async def request_openai_transcription(
    audio_path: Path,
    mode: str,
    with_timestamps: bool,
) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for the selected STT provider.")
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/ogg"
    model = OPENAI_STT_TIMESTAMP_MODEL if with_timestamps else OPENAI_STT_MODEL
    data: dict[str, str] = {
        "model": model,
        "response_format": "diarized_json" if with_timestamps else "json",
    }
    normalized_mode = normalize_transcription_mode(mode)
    if normalized_mode in {"ru", "en", "es"}:
        data["language"] = normalized_mode
    if with_timestamps:
        data["chunking_strategy"] = "auto"

    async with httpx.AsyncClient(**transcription_http_client_kwargs(OPENAI_PROXY_URL)) as client:
        with audio_path.open("rb") as audio_file:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                data=data,
                files={"file": (audio_path.name, audio_file, mime_type)},
            )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI returned an unexpected transcription response.")
    return payload


async def request_deepgram_transcription(audio_path: Path, mode: str) -> dict:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured for the selected STT provider.")
    normalized_mode = normalize_transcription_mode(mode)
    params: dict[str, str] = {
        "model": DEEPGRAM_STT_MODEL,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    if normalized_mode in {"ru", "en", "es"}:
        params["language"] = normalized_mode
    else:
        params["detect_language"] = "true"
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/ogg"

    async with httpx.AsyncClient(**transcription_http_client_kwargs(DEEPGRAM_PROXY_URL)) as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": mime_type,
            },
            content=audio_path.read_bytes(),
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Deepgram returned an unexpected transcription response.")
    return payload


def deepgram_segments_from_payload(payload: dict) -> list[dict]:
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    utterances = results.get("utterances") if isinstance(results, dict) else []
    segments: list[dict] = []
    for utterance in utterances or []:
        if not isinstance(utterance, dict):
            continue
        text = re.sub(r"\s+", " ", str(utterance.get("transcript") or "")).strip()
        if text:
            segments.append(
                {
                    "start": float(utterance.get("start", 0) or 0),
                    "end": float(utterance.get("end", 0) or 0),
                    "text": text,
                }
            )
    return segments


def deepgram_text_from_payload(payload: dict) -> str:
    segments = deepgram_segments_from_payload(payload)
    if segments:
        return " ".join(str(segment["text"]) for segment in segments).strip()
    try:
        alternatives = payload["results"]["channels"][0]["alternatives"]
        return str(alternatives[0].get("transcript") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


async def transcribe_with_selected_provider(audio_path: Path, mode: str = "ru") -> str:
    """Transcribe with the provider selected by the STT_PROVIDER setting."""
    if STT_PROVIDER == "openai":
        payload = await request_openai_transcription(audio_path, mode, with_timestamps=False)
        return str(payload.get("text") or "").strip()
    if STT_PROVIDER == "deepgram":
        return deepgram_text_from_payload(await request_deepgram_transcription(audio_path, mode))
    if STT_PROVIDER != "groq":
        raise RuntimeError(f"Unsupported STT_PROVIDER: {STT_PROVIDER}")
    return await transcribe_with_groq(audio_path, mode=mode)


async def transcribe_segments_with_selected_provider(
    audio_path: Path,
    mode: str = "auto",
) -> list[dict]:
    """Return timestamped segments from the configured transcription provider."""
    if STT_PROVIDER == "openai":
        payload = await request_openai_transcription(audio_path, mode, with_timestamps=True)
        result: list[dict] = []
        for segment in payload.get("segments", []) or []:
            if not isinstance(segment, dict):
                continue
            text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
            if text:
                result.append(
                    {
                        "start": float(segment.get("start", 0) or 0),
                        "end": float(segment.get("end", 0) or 0),
                        "text": text,
                    }
                )
        return result
    if STT_PROVIDER == "deepgram":
        return deepgram_segments_from_payload(await request_deepgram_transcription(audio_path, mode))
    if STT_PROVIDER != "groq":
        raise RuntimeError(f"Unsupported STT_PROVIDER: {STT_PROVIDER}")
    return await transcribe_with_groq_segments(audio_path, mode=mode)


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


async def run_subprocess(command: list[str], timeout_seconds: Optional[int] = None) -> str:
    """
    Run a subprocess and return stdout, raising a readable error on failure.
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise RuntimeError(
            "Command timed out: "
            f"{format_safe_command(command)}"
        ) from exc

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if process.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            f"{format_safe_command(command)}\n"
            f"{stderr_text[-2000:]}"
        )

    return stdout_text


def format_safe_command(command: list[str]) -> str:
    """Render a subprocess command without leaking proxy credentials."""
    safe_parts: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            safe_parts.append("***")
            redact_next = False
            continue
        safe_parts.append(str(part))
        if part in {"--proxy"}:
            redact_next = True
    return " ".join(safe_parts)


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


async def prepare_media_for_transcription(source_path: Path) -> Path:
    """
    Extract and normalize any Telegram audio/video media to a Whisper-friendly MP3.
    """
    ffmpeg_path = resolve_executable(FFMPEG_BINARY)
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not installed. Install ffmpeg to process audio and video files."
        )

    output_path = Path(tempfile.gettempdir()) / f"tg_media_audio_{uuid.uuid4().hex}.mp3"
    try:
        await run_subprocess(
            [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                AUDIO_CHUNK_BITRATE,
                str(output_path),
            ]
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not create a normalized audio file.")
        return output_path
    except Exception:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                logging.warning("Could not delete failed normalized audio file %s", output_path)
        raise


YOUTUBE_URL_PATTERN = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?[^ \n\r\t<>]*v=|shorts/|live/)|youtu\.be/)[^ \n\r\t<>]+",
    flags=re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    """
    Return whether a URL points to a supported YouTube video form.
    """
    return bool(YOUTUBE_URL_PATTERN.fullmatch(url.strip()))


def extract_youtube_url(text: str) -> str:
    """
    Extract the first supported YouTube URL from arbitrary message text.
    """
    match = YOUTUBE_URL_PATTERN.search(text or "")
    if not match:
        return ""
    return match.group(0).rstrip(").,;!?]")


def should_handle_youtube_text(text: str) -> bool:
    """
    Return whether a text message should be handled as a YouTube transcription.
    """
    return bool(extract_youtube_url(text)) and not (text or "").lstrip().startswith("/")


def youtube_ytdlp_base_command(use_proxy: bool = True) -> list[str]:
    """
    Build the common yt-dlp prefix required by current YouTube extraction.

    YouTube now serves JavaScript challenges for many media URLs. The Deno
    runtime and matching EJS package are installed from requirements.txt.
    """
    command = [
        YTDLP_BINARY,
        "--js-runtimes",
        YTDLP_JS_RUNTIME,
        "--extractor-retries",
        "5",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--retry-sleep",
        "extractor:linear=1:5:1",
        "--retry-sleep",
        "http:exp=1:8",
        "--extractor-args",
        YTDLP_EXTRACTOR_ARGS,
    ]
    proxy_url = get_effective_proxy_url(YTDLP_PROXY_URL)
    if use_proxy and proxy_url:
        command.extend(["--proxy", proxy_url])
    elif not use_proxy:
        command.extend(["--proxy", ""])
    if YTDLP_COOKIES_FILE:
        command.extend(["--cookies", YTDLP_COOKIES_FILE])
    return command


def build_youtube_metadata_command(url: str, use_proxy: bool = True) -> list[str]:
    return [
        *youtube_ytdlp_base_command(use_proxy=use_proxy),
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        url,
    ]


def build_youtube_download_command(url: str, destination_dir: Path) -> list[str]:
    return [
        *youtube_ytdlp_base_command(),
        "--no-playlist",
        "-x",
        "--audio-format",
        YOUTUBE_AUDIO_FORMAT,
        "-o",
        str(destination_dir / "youtube_%(id)s.%(ext)s"),
        url,
    ]


def select_youtube_caption_track(metadata: dict) -> Optional[tuple[str, str]]:
    """
    Select a manual or automatic caption track in the video's original
    language. The returned tuple is (track type, yt-dlp language key).
    """
    video_language = str(metadata.get("language") or "").strip().lower()
    preferred_languages = [video_language]
    if "-" in video_language:
        preferred_languages.append(video_language.split("-", 1)[0])

    for track_type, metadata_key in (
        ("manual", "subtitles"),
        ("automatic", "automatic_captions"),
    ):
        tracks = metadata.get(metadata_key)
        if not isinstance(tracks, dict) or not tracks:
            continue

        usable_keys = [str(key) for key in tracks if str(key) != "live_chat"]
        if track_type == "automatic":
            for language in usable_keys:
                if language.lower().endswith("-orig"):
                    return track_type, language

        for language in preferred_languages:
            if language in tracks:
                return track_type, language

        for language in usable_keys:
            formats = tracks.get(language)
            if not isinstance(formats, list):
                continue
            if any("(original)" in str(item.get("name") or "").lower() for item in formats if isinstance(item, dict)):
                return track_type, language

        if usable_keys:
            return track_type, usable_keys[0]

    return None


def build_youtube_caption_command(
    url: str,
    destination_dir: Path,
    track_type: str,
    language: str,
) -> list[str]:
    write_option = "--write-subs" if track_type == "manual" else "--write-auto-subs"
    return [
        *youtube_ytdlp_base_command(),
        "--no-playlist",
        "--skip-download",
        write_option,
        "--sub-langs",
        language,
        "--sub-format",
        "json3",
        "-o",
        str(destination_dir / "youtube_%(id)s.%(ext)s"),
        url,
    ]


def parse_youtube_json3_payload(payload: dict) -> list[dict]:
    """
    Parse a YouTube json3 object into timestamped text segments.
    """
    segments: list[dict] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
            continue
        text = "".join(
            str(part.get("utf8") or "")
            for part in event["segs"]
            if isinstance(part, dict)
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        try:
            start = max(0.0, float(event.get("tStartMs", 0)) / 1000)
            duration = max(0.0, float(event.get("dDurationMs", 0)) / 1000)
        except (TypeError, ValueError):
            continue
        segments.append({"start": start, "end": start + duration, "text": text})

    if not segments:
        raise RuntimeError("YouTube captions are empty.")
    return segments


def parse_youtube_json3_captions(caption_path: Path) -> list[dict]:
    """Parse a downloaded YouTube json3 caption file."""
    try:
        payload = json.loads(caption_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("yt-dlp returned invalid YouTube captions.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("yt-dlp returned invalid YouTube captions.")
    return parse_youtube_json3_payload(payload)


def youtube_caption_json3_url(metadata: dict, track_type: str, language: str) -> str:
    metadata_key = "subtitles" if track_type == "manual" else "automatic_captions"
    tracks = metadata.get(metadata_key)
    if not isinstance(tracks, dict):
        return ""
    formats = tracks.get(language)
    if not isinstance(formats, list):
        return ""
    for item in formats:
        if isinstance(item, dict) and item.get("ext") == "json3" and item.get("url"):
            return str(item["url"])
    return ""


async def download_youtube_captions(
    url: str,
    destination_dir: Path,
    metadata: dict,
) -> Optional[list[dict]]:
    """
    Download original-language captions when the video exposes them.
    Return None when no caption track exists so audio transcription can run.
    """
    selection = select_youtube_caption_track(metadata)
    if selection is None:
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    track_type, language = selection

    caption_url = youtube_caption_json3_url(metadata, track_type, language)
    if caption_url:
        try:
            client_kwargs: dict = {
                "follow_redirects": True,
                "timeout": 120,
                "trust_env": False,
            }
            proxy_url = get_effective_proxy_url(YTDLP_PROXY_URL)
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(caption_url)
                response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("unexpected caption JSON")
            return parse_youtube_json3_payload(payload)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            logging.warning("Direct YouTube caption download failed, trying yt-dlp: %s", exc)

    await run_subprocess(
        build_youtube_caption_command(url, destination_dir, track_type, language),
        timeout_seconds=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS,
    )
    candidates = sorted(
        destination_dir.glob("youtube_*.json3"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("yt-dlp did not create a YouTube caption file.")
    return parse_youtube_json3_captions(candidates[0])


def validate_youtube_metadata(metadata: dict) -> None:
    """
    Validate yt-dlp metadata for the MVP single-video flow.
    """
    if metadata.get("_type") == "playlist" or metadata.get("entries"):
        raise ValueError("Плейлисты пока не поддерживаются. Пришли ссылку на одно YouTube-видео.")

    duration = metadata.get("duration")
    if duration is None:
        raise ValueError("Не удалось определить длительность YouTube-видео.")

    try:
        duration_seconds = int(float(duration))
    except (TypeError, ValueError) as exc:
        raise ValueError("Не удалось определить длительность YouTube-видео.") from exc

    if duration_seconds <= 0:
        raise ValueError("Не удалось определить длительность YouTube-видео.")

    if duration_seconds > YOUTUBE_MAX_DURATION_SECONDS:
        raise ValueError(
            "Видео слишком длинное. "
            f"Максимум: {YOUTUBE_MAX_DURATION_SECONDS // 3600} ч "
            f"{(YOUTUBE_MAX_DURATION_SECONDS % 3600) // 60} мин."
        )


async def read_youtube_metadata(url: str) -> dict:
    """
    Read YouTube metadata through yt-dlp without downloading media.
    """
    try:
        output = await run_subprocess(
            build_youtube_metadata_command(url),
            timeout_seconds=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except RuntimeError as exc:
        error_text = str(exc).lower()
        retry_direct = bool(get_effective_proxy_url(YTDLP_PROXY_URL)) and any(
            marker in error_text
            for marker in (
                "http error 403",
                "http error 429",
                "sign in to confirm you’re not a bot",
                "sign in to confirm you're not a bot",
            )
        )
        if not retry_direct:
            raise
        logging.warning("Primary YouTube metadata route was rejected; retrying directly.")
        output = await run_subprocess(
            build_youtube_metadata_command(url, use_proxy=False),
            timeout_seconds=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS,
        )
    try:
        metadata = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid metadata JSON.") from exc

    if not isinstance(metadata, dict):
        raise RuntimeError("yt-dlp returned unexpected metadata.")

    validate_youtube_metadata(metadata)
    return metadata


async def download_youtube_audio(
    url: str,
    destination_dir: Path,
    metadata: Optional[dict] = None,
) -> tuple[Path, dict]:
    """
    Download only the audio track from a single YouTube video.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata or await read_youtube_metadata(url)
    await run_subprocess(
        build_youtube_download_command(url, destination_dir),
        timeout_seconds=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS,
    )

    candidates = sorted(
        (
            path
            for path in destination_dir.glob("youtube_*")
            if (
                path.is_file()
                and path.suffix.lower() == f".{YOUTUBE_AUDIO_FORMAT}"
                and path.stat().st_size > 0
            )
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("yt-dlp did not create an audio file.")

    return candidates[0], metadata


async def extract_audio_tail_with_ffmpeg(source_path: Path, duration: float) -> Path:
    """
    Extract the last seconds of audio as a small MP3 for tail-loss recovery.
    """
    ffmpeg_path = resolve_executable(FFMPEG_BINARY)
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not installed. Install ffmpeg to recover audio tail."
        )

    tail_seconds = min(float(TRANSCRIPTION_TAIL_RECOVERY_SECONDS), max(1.0, duration))
    start_seconds = max(0.0, duration - tail_seconds)
    output_path = Path(tempfile.gettempdir()) / f"tg_audio_tail_{uuid.uuid4().hex}.mp3"
    try:
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
                str(source_path),
                "-t",
                f"{tail_seconds:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                AUDIO_CHUNK_BITRATE,
                str(output_path),
            ]
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not create a tail audio file.")
        return output_path
    except Exception:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                logging.warning("Could not delete failed tail audio file %s", output_path)
        raise


def normalized_words(text: str) -> list[str]:
    """
    Normalize words for conservative duplicate detection at chunk boundaries.
    """
    return re.findall(r"[\wёЁ]+", text.lower(), flags=re.UNICODE)


def find_word_sequence(haystack: list[str], needle: list[str]) -> int:
    """
    Return the start index of a word sequence, or -1 when it is not present.
    """
    if not needle or len(needle) > len(haystack):
        return -1

    first = needle[0]
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start] == first and haystack[start:start + len(needle)] == needle:
            return start
    return -1


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
        if not part_words:
            continue

        merged_tail_window = merged_words[-220:]
        contained_anchor_size = min(8, len(part_words))
        contained_anchor = part_words[-contained_anchor_size:]
        if find_word_sequence(merged_tail_window, contained_anchor) >= 0:
            continue

        prefix_anchor_size = min(8, len(part_words))
        append_from_word = 0
        for start in range(0, max(1, len(part_words) - prefix_anchor_size + 1)):
            anchor = part_words[start:start + prefix_anchor_size]
            if len(anchor) < prefix_anchor_size:
                break
            position = find_word_sequence(merged_tail_window, anchor)
            if position >= 0:
                append_from_word = start + prefix_anchor_size

        if append_from_word > 0:
            split_words = part.split()
            part_without_overlap = " ".join(split_words[append_from_word:]).strip()
            if part_without_overlap:
                merged = f"{merged} {part_without_overlap}"
                merged_words = normalized_words(merged)
            continue

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


async def transcribe_tail_for_recovery(audio_path: Path, mode: str) -> str:
    """
    Transcribe the final audio window separately to avoid losing last phrases.
    """
    if TRANSCRIPTION_TAIL_RECOVERY_SECONDS <= 0:
        return ""

    try:
        duration = await get_audio_duration_seconds(audio_path)
    except Exception as exc:
        logging.warning("Could not read duration for tail recovery: %s", exc)
        return ""

    if duration < TRANSCRIPTION_TAIL_RECOVERY_MIN_DURATION_SECONDS:
        return ""

    tail_path: Optional[Path] = None
    try:
        tail_path = await extract_audio_tail_with_ffmpeg(audio_path, duration)
        logging.info(
            "Transcribing tail recovery window: path=%s duration=%s seconds",
            tail_path,
            TRANSCRIPTION_TAIL_RECOVERY_SECONDS,
        )
        return await transcribe_with_selected_provider(tail_path, mode=mode)
    except Exception as exc:
        logging.warning("Tail recovery transcription failed: %s", exc)
        return ""
    finally:
        if tail_path is not None:
            try:
                if tail_path.exists():
                    tail_path.unlink()
            except OSError:
                logging.warning("Could not delete tail recovery file %s", tail_path)


async def transcribe_audio_safely(
    audio_path: Path,
    mode: str = "ru",
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    Transcribe short files directly and long files through ffmpeg chunks.
    """
    audio_size = audio_path.stat().st_size
    if audio_size <= MAX_GROQ_CHUNK_BYTES:
        main_text = await transcribe_with_selected_provider(audio_path, mode=mode)
        tail_text = await transcribe_tail_for_recovery(audio_path, mode=mode)
        return merge_transcription_parts([main_text, tail_text])

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
            transcribed_parts.append(await transcribe_with_selected_provider(chunk_path, mode=mode))

        tail_text = await transcribe_tail_for_recovery(audio_path, mode=mode)
        return merge_transcription_parts([*transcribed_parts, tail_text])
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


async def transcribe_audio_with_timestamps(
    audio_path: Path,
    mode: str = "auto",
    progress_callback: Optional[ProgressCallback] = None,
) -> list[dict]:
    """
    Transcribe audio with segment timestamps. Large files keep their absolute
    timeline after chunking, including the configured overlap.
    """
    if audio_path.stat().st_size <= MAX_GROQ_CHUNK_BYTES:
        return await transcribe_segments_with_selected_provider(audio_path, mode=mode)

    chunk_dir = create_temp_directory(prefix="tg_timestamp_chunks_")
    try:
        chunks = await split_audio_with_ffmpeg(audio_path, chunk_dir)
        stride_seconds = AUDIO_CHUNK_SECONDS - AUDIO_CHUNK_OVERLAP_SECONDS
        all_segments: list[dict] = []

        for index, chunk_path in enumerate(chunks):
            if progress_callback is not None:
                await progress_callback(index + 1, len(chunks))
            chunk_segments = await transcribe_segments_with_selected_provider(chunk_path, mode=mode)
            offset = float(index * stride_seconds)
            overlap_cutoff = offset + (AUDIO_CHUNK_OVERLAP_SECONDS if index else 0)

            for segment in chunk_segments:
                absolute_start = offset + float(segment["start"])
                absolute_end = offset + float(segment["end"])
                if index and (absolute_start + absolute_end) / 2 < overlap_cutoff:
                    continue
                all_segments.append(
                    {
                        "start": absolute_start,
                        "end": absolute_end,
                        "text": segment["text"],
                    }
                )

        return all_segments
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


def format_youtube_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_youtube_timestamp(value: str) -> Optional[float]:
    """Convert a YouTube-style MM:SS or HH:MM:SS timestamp to seconds."""
    parts = value.strip().split(":")
    if len(parts) not in {2, 3} or not all(part.isdigit() for part in parts):
        return None

    numbers = [int(part) for part in parts]
    if numbers[-1] >= 60:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        return float(minutes * 60 + seconds)

    hours, minutes, seconds = numbers
    if minutes >= 60:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def normalize_youtube_outline(
    chapters: list[dict],
    duration: float,
) -> list[dict]:
    """Validate, sort, deduplicate, and bound chapter markers."""
    normalized: list[dict] = []
    seen_starts: set[int] = set()
    maximum_start = max(0.0, float(duration))

    for chapter in chapters:
        try:
            start = max(0.0, float(chapter.get("start", chapter.get("start_time", 0)) or 0))
        except (TypeError, ValueError):
            continue
        title = re.sub(r"\s+", " ", str(chapter.get("title") or "")).strip(" \t-–—|,;")
        rounded_start = int(start)
        is_youtube_placeholder = bool(
            re.fullmatch(r"<?untitled chapter \d+>?", title, flags=re.IGNORECASE)
        )
        if (
            not title
            or is_youtube_placeholder
            or rounded_start in seen_starts
            or start > maximum_start + 5
        ):
            continue
        seen_starts.add(rounded_start)
        normalized.append({"start": start, "title": title[:180]})

    normalized.sort(key=lambda item: float(item["start"]))
    return normalized


def extract_youtube_outline(metadata: dict) -> list[dict]:
    """Prefer native YouTube chapters, then parse timestamp rows in the description."""
    try:
        duration = float(metadata.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    native_chapters = metadata.get("chapters")
    if isinstance(native_chapters, list):
        outline = normalize_youtube_outline(
            [chapter for chapter in native_chapters if isinstance(chapter, dict)],
            duration,
        )
        if outline:
            return outline

    description = str(metadata.get("description") or "")
    description_chapters: list[dict] = []
    chapter_pattern = re.compile(
        r"^\s*(?:[-–—*•]\s*)?"
        r"(?P<timestamp>(?:\d{1,3}:)?\d{1,3}:\d{2})"
        r"(?:\s*[-–—|:]\s*|\s+)"
        r"(?P<title>\S.*)\s*$"
    )
    for line in description.splitlines():
        match = chapter_pattern.match(line)
        if match is None:
            continue
        start = parse_youtube_timestamp(match.group("timestamp"))
        if start is None:
            continue
        description_chapters.append({"start": start, "title": match.group("title")})
    return normalize_youtube_outline(description_chapters, duration)


def build_youtube_outline_timeline(
    segments: list[dict],
    duration: float,
    max_windows: int = 80,
) -> list[dict]:
    """Build compact samples spanning the whole video for chapter generation."""
    grouped = group_youtube_transcript_segments(segments)
    if not grouped:
        return []

    duration = max(float(duration), float(grouped[-1].get("end", 0) or 0), 1.0)
    window_count = min(max_windows, max(1, int((duration + 299) // 300)))
    window_seconds = duration / window_count
    buckets: list[list[str]] = [[] for _ in range(window_count)]

    for segment in grouped:
        start = max(0.0, float(segment.get("start", 0) or 0))
        index = min(window_count - 1, int(start / window_seconds))
        current_length = sum(len(part) for part in buckets[index])
        if current_length < 420:
            buckets[index].append(str(segment.get("text") or ""))

    timeline: list[dict] = []
    for index, texts in enumerate(buckets):
        excerpt = re.sub(r"\s+", " ", " ".join(texts)).strip()[:420]
        if excerpt:
            timeline.append({"start": round(index * window_seconds), "text": excerpt})
    return timeline


def build_fallback_youtube_outline(segments: list[dict], duration: float) -> list[dict]:
    """Create a safe, transcript-derived outline when an LLM is unavailable."""
    timeline = build_youtube_outline_timeline(segments, duration, max_windows=24)
    chapters = []
    for item in timeline:
        title = str(item["text"]).strip()
        if len(title) > 90:
            title = title[:87].rstrip(" ,.;:-") + "…"
        chapters.append({"start": item["start"], "title": title})
    return normalize_youtube_outline(chapters, duration)


async def generate_youtube_outline(segments: list[dict], duration: float) -> list[dict]:
    """Generate topic chapters across the full transcript, with a deterministic fallback."""
    fallback = build_fallback_youtube_outline(segments, duration)
    if not ENABLE_GROQ_LLM_POSTPROCESSING or not GROQ_API_KEY:
        return fallback

    timeline = build_youtube_outline_timeline(segments, duration)
    if not timeline:
        return fallback

    client = get_groq_client()
    try:
        response = await client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create an accurate table of contents for a video from timestamped transcript samples. "
                        "Cover the full duration, merge samples about the same topic, and never invent topics. "
                        "Keep chapter titles in the transcript's original language. Return only JSON as "
                        "{\"chapters\":[{\"start_time\":number,\"title\":string}]}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "duration_seconds": int(duration),
                            "timeline": timeline,
                            "chapter_count_guidance": "Use 3-40 chapters depending on duration and topic changes.",
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        raw_chapters = payload.get("chapters") if isinstance(payload, dict) else None
        if not isinstance(raw_chapters, list):
            return fallback
        generated = normalize_youtube_outline(
            [chapter for chapter in raw_chapters if isinstance(chapter, dict)],
            duration,
        )
        return generated or fallback
    except Exception as exc:
        logging.warning("Could not generate YouTube outline, using transcript excerpts: %s", exc)
        return fallback
    finally:
        await client.close()


def group_youtube_transcript_segments(
    segments: list[dict],
    max_chars: int = 105,
    max_span_seconds: float = 12.0,
    max_gap_seconds: float = 3.0,
) -> list[dict]:
    """Combine tiny caption fragments into readable YouTube-style rows."""
    grouped: list[dict] = []
    current: Optional[dict] = None

    for source in sorted(segments, key=lambda item: float(item.get("start", 0) or 0)):
        text = re.sub(r"\s+", " ", str(source.get("text") or "")).strip()
        if not text:
            continue
        start = max(0.0, float(source.get("start", 0) or 0))
        end = max(start, float(source.get("end", start) or start))

        if current is None:
            current = {"start": start, "end": end, "text": text}
            continue
        if text == current["text"]:
            continue

        candidate_text = f"{current['text']} {text}".strip()
        gap = max(0.0, start - float(current["end"]))
        span = end - float(current["start"])
        should_split = (
            gap > max_gap_seconds
            or span > max_span_seconds
            or len(candidate_text) > max_chars
        )
        if should_split:
            grouped.append(current)
            current = {"start": start, "end": end, "text": text}
        else:
            current["end"] = max(float(current["end"]), end)
            current["text"] = candidate_text

    if current is not None:
        grouped.append(current)
    return grouped


def format_youtube_transcript(segments: list[dict]) -> str:
    """
    Render a readable two-column TXT layout: timestamp left, caption right.
    """
    lines: list[str] = []
    for segment in group_youtube_transcript_segments(segments):
        timestamp = format_youtube_timestamp(float(segment.get("start", 0) or 0))
        lines.append(f"{timestamp:<9}│ {segment['text']}")
    return "\n".join(lines).strip()


def format_youtube_outline(chapters: list[dict]) -> str:
    """Render video contents with the same timestamp-left layout as the transcript."""
    lines = []
    for chapter in chapters:
        timestamp = format_youtube_timestamp(float(chapter.get("start", 0) or 0))
        lines.append(f"{timestamp:<9}│ {chapter['title']}")
    return "\n".join(lines).strip()


def format_youtube_document(chapters: list[dict], segments: list[dict]) -> str:
    """Combine the table of contents and the complete transcript in one TXT."""
    outline = format_youtube_outline(chapters)
    transcript = format_youtube_transcript(segments)
    sections = ["СОДЕРЖАНИЕ ВИДЕО", "", outline, "", "РАСШИФРОВКА ВИДЕО", "", transcript]
    return "\n".join(sections).strip()


def youtube_transcript_file_name(title: str) -> str:
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title)
    safe_title = re.sub(r"\s+", " ", safe_title).strip(" .")[:90]
    return f"{safe_title or 'youtube_transcript'}.txt"


async def send_youtube_transcription_file(
    bot: Bot,
    chat_id: int,
    transcript: str,
    title: str,
) -> None:
    """Always deliver a YouTube transcript as a UTF-8 TXT document."""
    transcript = transcript.strip()
    if not transcript:
        raise RuntimeError("YouTube transcription is empty.")

    temp_path = Path(tempfile.gettempdir()) / f"youtube_transcript_{uuid.uuid4().hex}.txt"
    try:
        temp_path.write_text(transcript, encoding="utf-8")
        await bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(str(temp_path), filename=youtube_transcript_file_name(title)),
            caption=(
                "✅ Расшифровка YouTube готова\n"
                f"🎬 {title[:850]}\n"
                "🕒 В файле: таймкод слева, текст справа"
            ),
        )
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError as cleanup_error:
            logging.warning("Could not delete YouTube transcript txt %s: %s", temp_path, cleanup_error)


def youtube_active_jobs_for_chat(chat_id: int) -> int:
    return youtube_active_jobs_by_chat.get(chat_id, 0)


def increment_youtube_active_jobs(chat_id: int) -> None:
    youtube_active_jobs_by_chat[chat_id] = youtube_active_jobs_for_chat(chat_id) + 1


def decrement_youtube_active_jobs(chat_id: int) -> None:
    current = youtube_active_jobs_for_chat(chat_id)
    if current <= 1:
        youtube_active_jobs_by_chat.pop(chat_id, None)
        return
    youtube_active_jobs_by_chat[chat_id] = current - 1


async def process_youtube_url(bot: Bot, chat_id: int, url: str, status_message: Message) -> None:
    """
    Prefer YouTube captions, otherwise transcribe downloaded audio with Whisper
    timestamps. The result is always delivered as a TXT document.
    """
    started_at = time.monotonic()
    temp_dir: Optional[Path] = None
    prepared_audio_path: Optional[Path] = None
    try:
        await safe_edit_message(
            status_message,
            render_processing_status(
                "🎬 Расшифровка YouTube",
                15,
                "Проверяю ссылку",
                "Узнаю название и длительность видео.",
                int(time.monotonic() - started_at),
            ),
        )

        temp_dir = create_temp_directory(prefix="youtube_transcript_")
        metadata = await read_youtube_metadata(url)
        title = str(metadata.get("title") or "YouTube-видео")[:120]

        await safe_edit_message(
            status_message,
            render_processing_status(
                "🎬 Расшифровка YouTube",
                25,
                "Ищу готовые субтитры",
                "Так результат получится быстрее и точнее.",
                int(time.monotonic() - started_at),
            ),
        )

        try:
            segments = await download_youtube_captions(url, temp_dir, metadata)
        except RuntimeError as caption_error:
            logging.warning("Could not use YouTube captions, falling back to audio: %s", caption_error)
            segments = None
        if segments is not None:
            await safe_edit_message(
                status_message,
                render_processing_status(
                    "🎬 Расшифровка YouTube",
                    85,
                    "Субтитры получены",
                    "Собираю аккуратный TXT-файл.",
                    int(time.monotonic() - started_at),
                ),
            )
        else:
            await safe_edit_message(
                status_message,
                render_processing_status(
                    "🎬 Расшифровка YouTube",
                    35,
                    "Готовых субтитров нет",
                    "Распознаю речь по аудиодорожке — это займёт немного больше времени.",
                    int(time.monotonic() - started_at),
                ),
            )
            downloaded_audio_path, _ = await download_youtube_audio(url, temp_dir, metadata=metadata)
            prepared_audio_path = await prepare_media_for_transcription(downloaded_audio_path)

            if groq_transcription_semaphore.locked():
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "🎬 Расшифровка YouTube",
                        50,
                        "Аудио в очереди",
                        "Жду завершения предыдущего распознавания.",
                        int(time.monotonic() - started_at),
                    ),
                )

            async with groq_transcription_semaphore:
                async def update_chunk_progress(index: int, total: int) -> None:
                    percent = 55 + round(index / total * 35)
                    await safe_edit_message(
                        status_message,
                        render_processing_status(
                            "🎬 Расшифровка YouTube",
                            percent,
                            "Распознаю речь",
                            f"Фрагмент {index} из {total}.",
                            int(time.monotonic() - started_at),
                        ),
                    )

                segments = await transcribe_audio_with_timestamps(
                    prepared_audio_path,
                    mode="auto",
                    progress_callback=update_chunk_progress,
                )

        chapters = extract_youtube_outline(metadata)
        if not chapters:
            await safe_edit_message(
                status_message,
                render_processing_status(
                    "🎬 Расшифровка YouTube",
                    92,
                    "Создаю содержание",
                    "Выделяю основные темы по всей расшифровке.",
                    int(time.monotonic() - started_at),
                ),
            )
            chapters = await generate_youtube_outline(
                segments,
                float(metadata.get("duration") or 0),
            )

        transcript = format_youtube_document(chapters, segments)

        elapsed_seconds = int(time.monotonic() - started_at)
        await safe_edit_message(
            status_message,
            render_processing_status(
                "✅ Расшифровка готова",
                100,
                "Файл подготовлен",
                "Отправляю TXT с содержанием и полной расшифровкой.",
                elapsed_seconds,
            ),
        )
        await send_youtube_transcription_file(bot, chat_id, transcript, title)
        logging.info("Sent YouTube transcription to chat_id=%s url=%s", chat_id, url)

    except ValueError as exc:
        logging.info("YouTube validation failed chat_id=%s url=%s error=%s", chat_id, url, exc)
        await safe_edit_message(
            status_message,
            render_processing_status(
                "⚠️ Не удалось открыть видео",
                100,
                "Не могу обработать ссылку",
                str(exc),
                int(time.monotonic() - started_at),
            ),
        )
    except Exception as exc:
        logging.exception("Failed to process YouTube URL: %s", exc)
        reason = describe_processing_error(exc)
        await safe_edit_message(
            status_message,
            render_processing_status(
                "⚠️ Видео пока не обработано",
                100,
                "Попробуй ещё раз чуть позже",
                reason,
                int(time.monotonic() - started_at),
            ),
        )
    finally:
        decrement_youtube_active_jobs(chat_id)
        if prepared_audio_path is not None:
            try:
                if prepared_audio_path.exists():
                    prepared_audio_path.unlink()
            except OSError:
                logging.warning("Could not delete YouTube prepared audio %s", prepared_audio_path)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def describe_processing_error(exc: Exception) -> str:
    """
    Return a short user-safe explanation for the most common failures.
    """
    if isinstance(exc, TelegramBadRequest) and "file is too big" in str(exc).lower():
        return (
            "Telegram не передал такой большой файл. Нажми «📤 Загрузить большой файл» в /start."
        )

    if isinstance(exc, AuthenticationError):
        return "Сервис распознавания временно недоступен. Мы уже записали ошибку — попробуй позже."

    if isinstance(exc, PermissionDeniedError):
        return "Сервис распознавания временно недоступен. Попробуй немного позже."

    if isinstance(exc, RateLimitError):
        return "Сейчас обрабатывается слишком много аудио. Подожди несколько минут и повтори."

    if isinstance(exc, BadRequestError):
        return "Не удалось прочитать аудиодорожку. Попробуй другое видео или аудиофайл."

    if isinstance(exc, APITimeoutError):
        return "Распознавание заняло слишком много времени. Попробуй ещё раз позже."

    if isinstance(exc, APIConnectionError):
        return "Не удалось связаться с сервисом распознавания. Попробуй немного позже."

    if isinstance(exc, APIStatusError):
        return "Сервис распознавания временно ответил ошибкой. Попробуй немного позже."

    if isinstance(exc, httpx.HTTPError):
        return "Возникла временная сетевая ошибка. Проверь соединение и попробуй снова."

    if isinstance(exc, RuntimeError) and (
        "ffmpeg" in str(exc).lower() or "ffprobe" in str(exc).lower()
    ):
        return "Не удалось подготовить аудиодорожку этого файла. Попробуй другой формат."

    error_text = str(exc).lower()
    if isinstance(exc, RuntimeError) and (
        "blocked due to the claimed content" in error_text
        or "blocked on copyright grounds" in error_text
        or ("video unavailable" in error_text and "copyright" in error_text)
    ):
        return (
            "Правообладатель заблокировал это видео на YouTube. "
            "YouTube не отдаёт боту ни аудио, ни субтитры, поэтому расшифровать его нельзя."
        )

    if isinstance(exc, RuntimeError) and "private video" in error_text:
        return "Это приватное видео. Бот может обрабатывать только доступные по ссылке ролики."

    if isinstance(exc, RuntimeError) and (
        "not available in your country" in error_text
        or "not available in your region" in error_text
    ):
        return "Это видео недоступно в регионе, из которого работает бот."

    if isinstance(exc, RuntimeError) and (
        "http error 429" in error_text
        or "sign in to confirm you’re not a bot" in error_text
        or "sign in to confirm you're not a bot" in error_text
    ):
        return (
            "YouTube временно не разрешил загрузить это видео. "
            "Бот попробует резервный маршрут при следующей попытке — повтори через несколько минут."
        )

    if isinstance(exc, RuntimeError) and "no supported javascript runtime" in error_text:
        return "Сейчас не удалось открыть видео на стороне YouTube. Попробуй немного позже."

    if isinstance(exc, RuntimeError) and "requested format is not available" in error_text:
        return "YouTube не предоставляет выбранное качество для этого видео. Выбери другой вариант."

    if isinstance(exc, RuntimeError) and "instead of requested" in error_text:
        return "YouTube не отдал файл строго выбранного качества, поэтому бот не стал подменять результат."

    if isinstance(exc, RuntimeError) and "yt-dlp" in error_text:
        return "YouTube не отдал аудиодорожку. Возможно, видео закрыто, удалено или недоступно в регионе."

    return "Произошла временная ошибка обработки. Попробуй ещё раз немного позже."


def run_environment_check() -> int:
    """
    Check the local environment without sending data to external services.
    """
    checks = [
        ("TELEGRAM_BOT_TOKEN", bool(TELEGRAM_BOT_TOKEN), True),
        ("GROQ_API_KEY", bool(GROQ_API_KEY), STT_PROVIDER == "groq"),
        ("OPENAI_API_KEY", bool(OPENAI_API_KEY), STT_PROVIDER == "openai"),
        ("DEEPGRAM_API_KEY", bool(DEEPGRAM_API_KEY), STT_PROVIDER == "deepgram"),
        ("TELEGRAM_API_BASE", bool(TELEGRAM_API_BASE), False),
        ("PUBLIC_UPLOAD_BASE_URL", bool(PUBLIC_UPLOAD_BASE_URL), False),
        ("TELEGRAM_PROXY_URL", bool(TELEGRAM_PROXY_URL), False),
        ("GROQ_PROXY_URL", bool(GROQ_PROXY_URL), False),
        ("ffmpeg", has_executable(FFMPEG_BINARY), True),
        ("ffprobe", has_executable(FFPROBE_BINARY), True),
        ("yt-dlp", has_executable(YTDLP_BINARY), True),
        (YTDLP_JS_RUNTIME, has_executable(YTDLP_JS_RUNTIME), True),
        ("voice.ogg", Path("voice.ogg").is_file(), True),
    ]

    print("Environment check:")
    missing_required = []
    for name, ok, required in checks:
        if ok:
            status = "OK"
        elif required:
            status = "MISSING"
            missing_required.append(name)
        else:
            status = "not set (optional)"
        print(f"- {name}: {status}")

    if missing_required:
        print("\nMissing required local dependencies: " + ", ".join(missing_required))
        return 1

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
    print(f"- STT provider: {STT_PROVIDER}")
    print(f"- Active STT model: {active_stt_model_name()}")
    print(f"- Backup STT models: {', '.join(STT_BACKUP_MODELS)}")
    print(f"- Concurrent transcription jobs: {STT_MAX_CONCURRENT_JOBS}")
    print(f"- Max transcription chunk upload: {MAX_GROQ_CHUNK_BYTES // (1024 * 1024)} MB")
    print(f"- Default transcription mode: {normalize_transcription_mode(DEFAULT_TRANSCRIPTION_MODE)}")
    print(f"- Audio chunk length: {AUDIO_CHUNK_SECONDS} seconds")
    print(f"- Audio chunk overlap: {AUDIO_CHUNK_OVERLAP_SECONDS} seconds")
    print(f"- Audio chunk bitrate: {AUDIO_CHUNK_BITRATE}")
    print(f"- Tail recovery: last {TRANSCRIPTION_TAIL_RECOVERY_SECONDS} seconds")
    print(f"- Tail recovery min duration: {TRANSCRIPTION_TAIL_RECOVERY_MIN_DURATION_SECONDS} seconds")
    print(f"- YouTube max duration: {YOUTUBE_MAX_DURATION_SECONDS} seconds")
    print(f"- YouTube download timeout: {YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS} seconds")
    print(f"- YouTube audio format: {YOUTUBE_AUDIO_FORMAT}")
    print(f"- yt-dlp JavaScript runtime: {YTDLP_JS_RUNTIME}")
    print(f"- yt-dlp cookies file configured: {bool(YTDLP_COOKIES_FILE)}")
    print(f"- yt-dlp proxy configured: {bool(get_effective_proxy_url(YTDLP_PROXY_URL))}")
    print(f"- YouTube active jobs per chat: {YOUTUBE_MAX_ACTIVE_JOBS_PER_CHAT}")
    print(f"- Upload server bind: {UPLOAD_HOST}:{UPLOAD_PORT}")
    print(f"- Public upload base URL: {get_public_upload_base_url()}")
    print(f"- Max direct upload: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    print(f"- Browser upload chunk size: {UPLOAD_CHUNK_BYTES // (1024 * 1024)} MB")
    print(f"- Upload token TTL: {UPLOAD_TOKEN_TTL_SECONDS} seconds")
    media_config = media_platform_config_from_env(
        public_base_url=get_public_upload_base_url(),
        max_upload_bytes=MAX_UPLOAD_BYTES,
        upload_chunk_bytes=UPLOAD_CHUNK_BYTES,
        ffmpeg_binary=FFMPEG_BINARY,
        max_groq_chunk_bytes=MAX_GROQ_CHUNK_BYTES,
        audio_chunk_seconds=AUDIO_CHUNK_SECONDS,
        audio_chunk_overlap_seconds=AUDIO_CHUNK_OVERLAP_SECONDS,
    )
    print(f"- Media library DB: {media_config.db_path}")
    print(f"- Media object storage: {media_config.storage_dir}")
    print(f"- Media max duration: {media_config.max_duration_seconds} seconds")
    print(f"- Media upload session TTL: {media_config.upload_session_ttl_seconds} seconds")
    print(
        "- Groq LLM post-processing: "
        + (f"enabled ({GROQ_LLM_MODEL})" if ENABLE_GROQ_LLM_POSTPROCESSING else "disabled")
    )

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
        "👋 Привет! Я превращаю речь в аккуратный текст.\n\n"
        "🎙 Отправь голосовое, кружочек, аудио или видео.\n"
        "🎬 YouTube можно расшифровать или скачать отдельной кнопкой.\n"
        "📤 Большие файлы загружай через безопасную форму.\n"
        "📚 В медиатеке доступны поиск, главы и субтитры.\n\n"
        "Выбери действие ниже — дальше я всё подскажу.",
        reply_markup=build_main_keyboard(message.chat.id),
    )


@router.callback_query(F.data == "youtube:help")
async def youtube_help_callback(callback: CallbackQuery) -> None:
    """Explain the YouTube flow from a dedicated main-menu button."""
    if callback.message is None:
        await callback.answer("Не удалось открыть раздел.", show_alert=True)
        return
    youtube_download_waiting_chats.discard(callback.message.chat.id)
    await callback.message.answer(
        "🎬 Расшифровка YouTube\n\n"
        "1. Скопируй ссылку на видео.\n"
        "2. Отправь её следующим сообщением.\n"
        "3. Я найду субтитры или распознаю аудио.\n"
        "4. Готовый результат пришлю TXT-файлом: таймкод слева, текст справа.\n\n"
        "Поддерживаются отдельные видео длительностью до 12 часов.",
    )
    await callback.answer("Теперь просто отправь ссылку")


@router.callback_query(F.data == "youtube_download:help")
async def youtube_download_help_callback(callback: CallbackQuery) -> None:
    """Arm the next YouTube URL for downloading instead of transcription."""
    if callback.message is None:
        await callback.answer("Не удалось открыть раздел.", show_alert=True)
        return
    chat_id = callback.message.chat.id
    youtube_download_waiting_chats.add(chat_id)
    await callback.message.answer(
        "📥 Скачать видео с YouTube\n\n"
        "Отправь ссылку следующим сообщением. Я покажу только те варианты, "
        "которые YouTube действительно предоставляет: 360p, 720p, 1080p и аудио.\n\n"
        "После выбора подготовлю единый MP4 точного разрешения или MP3 и пришлю "
        "безопасную ссылку на готовый файл."
    )
    await callback.answer("Теперь отправь ссылку на видео")


@router.callback_query(F.data.startswith("transcription_mode:"))
async def transcription_mode_callback(callback: CallbackQuery) -> None:
    """
    Switch direct transcription language mode for the current chat.
    """
    chat_id = callback.message.chat.id if callback.message else None
    if chat_id is None:
        await callback.answer("Не удалось определить чат.", show_alert=True)
        return

    requested_mode = (callback.data or "").split(":", 1)[1]
    if requested_mode == "menu":
        await callback.message.answer(
            "Выбери язык для следующих расшифровок:",
            reply_markup=build_transcription_mode_keyboard(chat_id),
        )
        await callback.answer()
        return

    mode = normalize_transcription_mode(requested_mode)
    chat_transcription_modes[chat_id] = mode
    await callback.message.answer(
        f"Режим расшифровки: {transcription_mode_label(mode)}",
        reply_markup=build_transcription_mode_keyboard(chat_id),
    )
    await callback.answer(f"Выбран режим: {transcription_mode_label(mode)}")


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    logging.info("Received /help from chat_id=%s", message.chat.id)
    await message.answer(
        "✨ Что я умею\n\n"
        "🎙 Голосовые, кружочки, аудио и видео — отправь прямо в чат.\n"
        "🎬 YouTube до 12 часов — нажми кнопку в /start или сразу пришли ссылку.\n"
        "📥 Скачивание YouTube — кнопка в /start или /download ссылка.\n"
        "📤 Файл больше 20 МБ — используй /long.\n"
        "📚 Медиатека, главы и поиск — /library.\n"
        "🔎 Поиск по расшифровкам — /search запрос.\n"
        "🌐 Язык можно изменить кнопкой в /start.\n\n"
        "После обработки временные медиафайлы удаляются автоматически.",
        reply_markup=build_main_keyboard(message.chat.id),
    )


@router.message(Command("long"))
async def long_upload_handler(message: Message) -> None:
    logging.info("Received /long from chat_id=%s", message.chat.id)
    text, keyboard = create_upload_prompt(message.chat.id)
    await message.answer(text, reply_markup=keyboard)


@router.message(Command("library"))
async def library_handler(message: Message) -> None:
    logging.info("Received /library from chat_id=%s", message.chat.id)
    await message.answer(
        "Открой медиатеку: там можно загружать аудио и видео, искать по тексту, "
        "открывать таймкоды, скачивать TXT/SRT/VTT, смотреть главы, summary и задачи.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Открыть медиатеку",
                        url=build_library_url(message.chat.id),
                    )
                ]
            ]
        ),
    )


@router.message(Command("search"))
async def search_handler(message: Message) -> None:
    logging.info("Received /search from chat_id=%s", message.chat.id)
    query = (message.text or "").partition(" ")[2].strip()
    if not query:
        await message.answer("Напиши запрос после команды, например:\n/search нормализация базы данных")
        return
    if media_platform is None:
        await message.answer("Медиатека еще не запущена. Перезапусти бота и попробуй снова.")
        return

    results = await media_platform.search(query, chat_id=message.chat.id, limit=5)
    if not results:
        await message.answer("Ничего не нашел в медиатеке.")
        return

    lines = ["Нашел в медиатеке:"]
    for result in results:
        timestamp = int(result["start_time"])
        snippet = str(result["text"])[:260].replace("\n", " ")
        lines.append(
            f"\n{result['title']} — {timestamp // 3600:02d}:{(timestamp % 3600) // 60:02d}:{timestamp % 60:02d}\n"
            f"{snippet}\n"
            f"{media_platform.media_url(int(result['media_id']), timestamp)}"
        )
    await message.answer("\n".join(lines))


async def prepare_youtube_download_request(message: Message, url: str) -> None:
    """Read YouTube formats and present exact, available quality choices."""
    if youtube_download_service is None:
        await message.answer("Сервис скачивания ещё запускается. Попробуй снова через несколько секунд.")
        return

    started_at = time.monotonic()
    status_message = await message.answer(
        render_processing_status(
            "📥 Скачивание YouTube",
            10,
            "Проверяю видео",
            "Получаю название, длительность и доступные качества.",
        )
    )
    try:
        metadata = await read_youtube_metadata(url)
        request = youtube_download_service.create_request(message.chat.id, url, metadata)
        quality_names = [
            "аудио" if quality == QUALITY_AUDIO else f"{quality}p"
            for quality in request.available_qualities
        ]
        await safe_edit_message(
            status_message,
            (
                "📥 <b>Выбери качество</b>\n\n"
                f"🎬 {escape(request.title)}\n"
                f"⏱ {format_download_duration(request.duration)}\n"
                f"✅ Доступно: {', '.join(quality_names)}\n\n"
                "Я не подменяю качество: итоговый MP4 будет иметь ровно выбранную высоту."
            ),
            reply_markup=build_youtube_download_quality_keyboard(request),
            parse_mode="HTML",
        )
    except Exception as exc:
        logging.exception("Could not prepare YouTube download options: %s", exc)
        await safe_edit_message(
            status_message,
            render_processing_status(
                "⚠️ Не удалось открыть видео",
                100,
                "Варианты скачивания недоступны",
                describe_processing_error(exc),
                int(time.monotonic() - started_at),
            ),
        )


def youtube_download_result_keyboard(
    service: YouTubeDownloadService,
    record: YouTubeDownloadRecord,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⬇️ Скачать готовый файл",
                    url=service.landing_url(record),
                )
            ]
        ]
    )


def render_youtube_download_ready_text(
    service: YouTubeDownloadService,
    record: YouTubeDownloadRecord,
    preparation_seconds: float,
    was_cached: bool,
) -> str:
    quality_text = "MP3" if record.quality == QUALITY_AUDIO else f"{record.height}p MP4"
    cache_text = "Файл уже был в кэше." if was_cached else "Файл скачан и проверен."
    ttl_hours = max(1, (service.config.ttl_seconds + 3599) // 3600)
    return (
        "✅ <b>Видео готово</b>\n\n"
        f"🎬 {escape(record.title)}\n"
        f"🎞 Качество: {quality_text}\n"
        f"📦 Размер: {format_file_size(record.file_size)}\n"
        f"⏱ Подготовка: {preparation_seconds:.1f} сек.\n"
        f"⚡ {cache_text}\n\n"
        f"Файл хранится {ttl_hours} ч. Каждое открытие или продолжение скачивания "
        "продлевает этот срок."
    )


async def deliver_one_youtube_notification(
    bot: Bot,
    service: YouTubeDownloadService,
    notification: YouTubeDownloadNotification,
) -> bool:
    record = await service.get_by_token(notification.token)
    if record is None:
        await service.mark_notification_delivered(notification.notification_id)
        return False
    try:
        await bot.send_message(
            chat_id=notification.chat_id,
            text=notification.text,
            reply_markup=youtube_download_result_keyboard(service, record),
            parse_mode="HTML",
        )
    except Exception as exc:
        logging.warning(
            "Could not deliver persisted YouTube result notification id=%s: %s",
            notification.notification_id,
            exc,
        )
        await service.reschedule_notification(notification)
        return False
    await service.mark_notification_delivered(notification.notification_id)
    return True


async def deliver_pending_youtube_notifications(
    bot: Bot,
    service: YouTubeDownloadService,
    limit: int = 20,
) -> int:
    delivered = 0
    for notification in await service.pending_notifications(limit):
        if await deliver_one_youtube_notification(bot, service, notification):
            delivered += 1
    return delivered


async def youtube_download_notification_loop(
    bot: Bot,
    service: YouTubeDownloadService,
) -> None:
    while True:
        try:
            await deliver_pending_youtube_notifications(bot, service)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.exception("YouTube result delivery loop failed: %s", exc)
        await asyncio.sleep(15)


async def process_youtube_download_selection(
    bot: Bot,
    chat_id: int,
    request: YouTubeDownloadRequest,
    quality: str,
    status_message: Message,
) -> None:
    """Download, validate, cache, and deliver a ready file link."""
    if youtube_download_service is None:
        await safe_edit_message(status_message, "Сервис скачивания временно недоступен.")
        youtube_download_active_chats.discard(chat_id)
        return

    started_at = time.monotonic()
    try:
        await safe_edit_message(
            status_message,
            render_processing_status(
                "📥 Подготовка файла",
                25,
                "Скачиваю с YouTube",
                "Получаю видео и аудио параллельными фрагментами.",
            ),
        )
        record, was_cached, preparation_seconds = await youtube_download_service.download(
            request,
            quality,
        )
        elapsed_seconds = int(time.monotonic() - started_at)
        quality_text = "MP3" if quality == QUALITY_AUDIO else f"{record.height}p MP4"
        ready_text = render_youtube_download_ready_text(
            youtube_download_service,
            record,
            preparation_seconds,
            was_cached,
        )
        notification = await youtube_download_service.queue_notification(
            chat_id,
            record.token,
            ready_text,
        )
        edited = await safe_edit_message(
            status_message,
            ready_text,
            reply_markup=youtube_download_result_keyboard(youtube_download_service, record),
            parse_mode="HTML",
        )
        if edited:
            await youtube_download_service.mark_notification_delivered(
                notification.notification_id
            )
        else:
            await deliver_one_youtube_notification(
                bot,
                youtube_download_service,
                notification,
            )

        if record.file_size <= youtube_download_service.config.telegram_direct_limit_bytes:
            try:
                media = FSInputFile(str(record.file_path), filename=record.file_name)
                caption = (
                    f"✅ Готово · {quality_text} · {format_file_size(record.file_size)}\n"
                    "Ссылка выше останется запасным способом скачивания."
                )
                if quality == QUALITY_AUDIO:
                    await bot.send_document(chat_id=chat_id, document=media, caption=caption)
                else:
                    await bot.send_video(
                        chat_id=chat_id,
                        video=media,
                        caption=caption,
                        supports_streaming=True,
                    )
            except Exception as telegram_error:
                logging.warning(
                    "Could not additionally send small YouTube file through Telegram: %s",
                    telegram_error,
                )
        logging.info(
            "YouTube download ready chat_id=%s video_id=%s quality=%s size=%s cached=%s elapsed=%s",
            chat_id,
            request.video_id,
            quality,
            record.file_size,
            was_cached,
            elapsed_seconds,
        )
    except Exception as exc:
        logging.exception("Failed to download YouTube video: %s", exc)
        await safe_edit_message(
            status_message,
            (
                f"{render_processing_status(
                    '⚠️ Не удалось скачать видео',
                    100,
                    'Файл не подготовлен',
                    describe_processing_error(exc),
                    int(time.monotonic() - started_at),
                )}\n\n"
                "Попробуй ещё раз или выбери меньший размер / только аудио:"
            ),
            reply_markup=build_youtube_download_quality_keyboard(request),
        )
    finally:
        youtube_download_active_chats.discard(chat_id)


@router.message(Command("download"))
async def youtube_download_command_handler(message: Message) -> None:
    """Support /download URL in addition to the guided button flow."""
    url = extract_youtube_url((message.text or "").partition(" ")[2])
    if not url:
        youtube_download_waiting_chats.add(message.chat.id)
        await message.answer("Пришли YouTube-ссылку следующим сообщением.")
        return
    youtube_download_waiting_chats.discard(message.chat.id)
    await prepare_youtube_download_request(message, url)


@router.callback_query(F.data.startswith("youtube_download:select:"))
async def youtube_download_quality_callback(callback: CallbackQuery, bot: Bot) -> None:
    if callback.message is None or youtube_download_service is None:
        await callback.answer("Сервис скачивания недоступен.", show_alert=True)
        return
    parts = (callback.data or "").split(":")
    if len(parts) != 4:
        await callback.answer("Эта кнопка устарела.", show_alert=True)
        return
    request_id, quality = parts[2], parts[3]
    chat_id = callback.message.chat.id
    request = youtube_download_service.get_request(request_id, chat_id)
    if request is None:
        await callback.answer("Ссылка выбора истекла. Отправь видео ещё раз.", show_alert=True)
        return
    if chat_id in youtube_download_active_chats:
        await callback.answer("Одно скачивание уже выполняется.", show_alert=True)
        return
    if quality not in request.available_qualities:
        await callback.answer("Это качество недоступно.", show_alert=True)
        return

    youtube_download_active_chats.add(chat_id)
    await callback.answer("Начинаю скачивание")
    status_message = await callback.message.answer(
        render_processing_status(
            "📥 Подготовка файла",
            15,
            "Выбор принят",
            "Можно закрыть Telegram — я пришлю готовую ссылку.",
        )
    )
    asyncio.create_task(
        process_youtube_download_selection(bot, chat_id, request, quality, status_message)
    )


@router.message(lambda message: should_handle_youtube_text(message.text or ""))
async def youtube_url_handler(message: Message, bot: Bot) -> None:
    """
    Accept a YouTube URL directly in chat and transcribe its audio.
    """
    url = extract_youtube_url(message.text or "")
    if not url:
        return

    chat_id = message.chat.id
    if chat_id in youtube_download_waiting_chats:
        youtube_download_waiting_chats.discard(chat_id)
        await prepare_youtube_download_request(message, url)
        return

    if youtube_active_jobs_for_chat(chat_id) >= YOUTUBE_MAX_ACTIVE_JOBS_PER_CHAT:
        await message.answer(
            "⏳ Одно видео уже обрабатывается. Я обязательно пришлю файл — дождись завершения."
        )
        return

    increment_youtube_active_jobs(chat_id)
    status_message = await message.answer(
        render_processing_status(
            "🎬 Расшифровка YouTube",
            5,
            "Ссылка получена",
            "Начинаю обработку — можно заниматься своими делами.",
        )
    )
    asyncio.create_task(process_youtube_url(bot, chat_id, url, status_message))


@router.message(F.voice | F.video_note | F.audio | F.video | F.document)
async def audio_handler(message: Message, bot: Bot) -> None:
    logging.info(
        "Received media-like message chat_id=%s voice=%s video_note=%s audio=%s video=%s document=%s",
        message.chat.id,
        bool(message.voice),
        bool(message.video_note),
        bool(message.audio),
        bool(message.video),
        bool(message.document),
    )

    if not is_supported_audio_message(message):
        await message.answer(
            "Я могу обработать голосовые, кружочки, аудио и видеофайлы. "
            "Если файл пришел как документ без медиа-типа, добавь обычное аудио/видео расширение."
        )
        return

    local_path: Optional[Path] = None
    prepared_audio_path: Optional[Path] = None

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
        status_message = await message.answer(
            render_processing_status(
                "Расшифровка медиа",
                10,
                "Получаю файл",
                "Скачиваю медиа из Telegram.",
            )
        )

        safe_suffix = ".ogg"
        if "." in original_name:
            suffix_candidate = Path(original_name).suffix.lower()
            if suffix_candidate:
                safe_suffix = suffix_candidate

        temp_dir = Path(tempfile.gettempdir())
        local_path = temp_dir / f"tg_voice_{uuid.uuid4().hex}{safe_suffix}"

        await download_telegram_file(bot=bot, file_id=file_id, destination=local_path)
        logging.info("Downloaded Telegram file to %s", local_path)
        await safe_edit_message(
            status_message,
            render_processing_status(
                "Расшифровка медиа",
                25,
                "Файл получен",
                "Проверяю файл и готовлю распознавание.",
                int(time.monotonic() - started_at),
            ),
        )

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
                render_processing_status(
                    "Расшифровка медиа",
                    30,
                    "Файл в очереди",
                    "Жду завершения предыдущего распознавания.",
                    int(time.monotonic() - started_at),
                ),
            )

        async with groq_transcription_semaphore:
            transcription_path = local_path
            if should_prepare_media_with_ffmpeg(message, original_name):
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка медиа",
                        45,
                        "Готовлю аудио",
                        "Извлекаю и нормализую звуковую дорожку.",
                        int(time.monotonic() - started_at),
                    ),
                )
                prepared_audio_path = await prepare_media_for_transcription(local_path)
                transcription_path = prepared_audio_path
            else:
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка медиа",
                        45,
                        "Готовлю аудио",
                        "Файл подходит для прямого распознавания.",
                        int(time.monotonic() - started_at),
                    ),
                )

            async def update_chunk_progress(index: int, total: int) -> None:
                percent = 55 + round(index / total * 35)
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка медиа",
                        percent,
                        "Распознаю речь",
                        f"Фрагмент {index} из {total}.",
                        int(time.monotonic() - started_at),
                    ),
                )

            if transcription_path.stat().st_size > MAX_GROQ_CHUNK_BYTES:
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка медиа",
                        50,
                        "Файл длинный",
                        "Разбиваю аудио на части для точного распознавания.",
                        int(time.monotonic() - started_at),
                    ),
                )
            else:
                await safe_edit_message(
                    status_message,
                    render_processing_status(
                        "Расшифровка медиа",
                        70,
                        "Распознаю речь",
                        "Отправил аудио в Groq Whisper.",
                        int(time.monotonic() - started_at),
                    ),
                )

            text = await transcribe_audio_safely(
                transcription_path,
                mode=get_chat_transcription_mode(message.chat.id),
                progress_callback=update_chunk_progress,
            )
        logging.info("Groq transcription finished, text_length=%s", len(text))

        elapsed_seconds = int(time.monotonic() - started_at)
        await safe_edit_message(
            status_message,
            render_processing_status(
                "Расшифровка готова",
                100,
                "Готово",
                "Сейчас пришлю текст.",
                elapsed_seconds,
            ),
        )
        await send_transcription_result(
            bot,
            message.chat.id,
            text,
            mode=get_chat_transcription_mode(message.chat.id),
        )
        logging.info("Sent transcription to chat_id=%s", message.chat.id)

    except Exception as exc:
        logging.exception("Failed to process audio message: %s", exc)
        reason = describe_processing_error(exc)
        await message.answer(
            "Не получилось распознать аудио.\n\n"
            f"Причина: {reason}\n\n"
            "Попробуй отправить другое голосовое, кружочек, аудио/видео или пришли мне лог Amvera "
            "со строкой `Failed to process audio message`."
        )

    finally:
        # Always delete local file after processing or failure.
        for temp_path in (local_path, prepared_audio_path):
            if temp_path is None:
                continue
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as cleanup_error:
                logging.warning("Could not delete temp file %s: %s", temp_path, cleanup_error)


@router.message()
async def fallback_handler(message: Message) -> None:
    logging.info("Received unsupported message chat_id=%s", message.chat.id)
    await message.answer(
        "Отправь голосовое, кружочек, аудио или видеофайл, "
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
