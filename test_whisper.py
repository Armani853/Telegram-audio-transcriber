import asyncio
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import tempfile
import uuid
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from dotenv import load_dotenv
from groq import AsyncGroq
import httpx


# ============================================================
# Configuration
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip()
GROQ_PROXY_URL = os.getenv("GROQ_PROXY_URL", TELEGRAM_PROXY_URL).strip()

# Groq Whisper model
GROQ_WHISPER_MODEL = "whisper-large-v3"

# Forced transcription language
TRANSCRIPTION_LANGUAGE = "ru"

# Telegram message length limit is 4096; keep margin for safety.
TELEGRAM_SAFE_CHUNK_SIZE = 3500

# Groq Whisper Large V3 max file size is 100 MB according to Groq docs.
MAX_AUDIO_SIZE_BYTES = 100 * 1024 * 1024
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = Path(__file__).with_name("bot_error.log")

router = Router()


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


def get_groq_client() -> AsyncGroq:
    """
    Create the Groq client only after configuration is validated.
    """
    validate_config()
    http_client_kwargs = {
        "trust_env": False,
        "timeout": httpx.Timeout(180.0, connect=30.0),
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
    Create Telegram bot, optionally using TELEGRAM_PROXY_URL.
    """
    validate_config()
    telegram_proxy_url = get_effective_proxy_url(TELEGRAM_PROXY_URL)
    if telegram_proxy_url:
        return Bot(
            token=TELEGRAM_BOT_TOKEN,
            session=AiohttpSession(proxy=telegram_proxy_url),
        )

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

    try:
        with audio_path.open("rb") as audio_file:
            transcription = await groq_client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
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


def run_environment_check() -> int:
    """
    Check the local environment without sending data to external services.
    """
    checks = [
        ("TELEGRAM_BOT_TOKEN", bool(TELEGRAM_BOT_TOKEN), True),
        ("GROQ_API_KEY", bool(GROQ_API_KEY), True),
        ("TELEGRAM_PROXY_URL", bool(TELEGRAM_PROXY_URL), False),
        ("GROQ_PROXY_URL", bool(GROQ_PROXY_URL), False),
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

    if audio_path.stat().st_size > MAX_AUDIO_SIZE_BYTES:
        print("File is too large. Maximum size is 100 MB.")
        return 1

    text = await transcribe_with_groq(audio_path)
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
        "и я расшифрую его в текст на русском языке."
    )


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    logging.info("Received /help from chat_id=%s", message.chat.id)
    await message.answer(
        "Как пользоваться:\n"
        "1. Отправь обычное голосовое сообщение.\n"
        "2. Или отправь/перешли .ogg файл.\n"
        "3. Я скачаю аудио, отправлю его в Groq Whisper Large V3, "
        "пришлю текст и удалю временный файл."
    )


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

        if file_size is not None and file_size > MAX_AUDIO_SIZE_BYTES:
            await message.answer(
                "Файл слишком большой для распознавания. Максимальный размер — 100 МБ."
            )
            return

        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        status_message = await message.answer("Скачиваю аудио и начинаю распознавание...")

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

        if local_path.stat().st_size > MAX_AUDIO_SIZE_BYTES:
            await message.answer(
                "Файл слишком большой для распознавания. Максимальный размер — 100 МБ."
            )
            return

        await safe_edit_message(status_message, "Аудио получено. Распознаю речь через Groq Whisper...")

        text = await transcribe_with_groq(local_path)
        logging.info("Groq transcription finished, text_length=%s", len(text))

        await send_long_message(message, text)
        await safe_delete_message(status_message)
        logging.info("Sent transcription to chat_id=%s", message.chat.id)

    except Exception as exc:
        logging.exception("Failed to process audio message: %s", exc)
        await message.answer(
            "Не получилось распознать аудио. Возможные причины: файл повреждён, "
            "формат не поддерживается, аудио слишком длинное или временно недоступен Groq API. "
            "Попробуй отправить другое .ogg/голосовое сообщение."
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
        "и я переведу речь в текст."
    )


# ============================================================
# Entrypoint
# ============================================================

async def main() -> None:
    validate_config()
    configure_logging()

    bot = create_bot()
    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Checking Telegram connection...")
    me = await bot.get_me()
    logging.info("Bot connected as @%s", me.username)
    await bot.delete_webhook(drop_pending_updates=False)
    logging.info("Polling started. Press Ctrl+C to stop.")

    try:
        await dp.start_polling(bot)
    finally:
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
