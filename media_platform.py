import asyncio
from dataclasses import dataclass
from datetime import datetime
from html import escape
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from typing import Awaitable, Callable, Optional

from aiohttp import web
from aiogram import Bot
from aiogram.types import Message


SUPPORTED_MEDIA_SUFFIXES = {
    ".ogg",
    ".opus",
    ".mp3",
    ".m4a",
    ".wav",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
JOB_STATUS_UPLOADED = "UPLOADED"
JOB_STATUS_STORED = "STORED"
JOB_STATUS_AUDIO_EXTRACTED = "AUDIO_EXTRACTED"
JOB_STATUS_TRANSCRIBING = "TRANSCRIBING"
JOB_STATUS_SUBTITLES_READY = "SUBTITLES_READY"
JOB_STATUS_INDEXED = "INDEXED"
JOB_STATUS_SUMMARY_READY = "SUMMARY_READY"
JOB_STATUS_DONE = "DONE"
JOB_STATUS_FAILED = "FAILED"


ProgressCallback = Callable[[int, int], Awaitable[None]]
RunCommand = Callable[[list[str]], Awaitable[str]]
DurationReader = Callable[[Path], Awaitable[float]]
SplitAudio = Callable[[Path, Path], Awaitable[list[Path]]]
TranscribeOne = Callable[[Path], Awaitable[str]]
MergeTranscripts = Callable[[list[str]], str]
TempDirectoryFactory = Callable[[str], Path]
ErrorFormatter = Callable[[Exception], str]
PostprocessTranscript = Callable[[str, list[dict], float], Awaitable[Optional[dict]]]


@dataclass
class MediaPlatformConfig:
    public_base_url: str
    db_path: Path
    storage_dir: Path
    max_upload_bytes: int
    upload_chunk_bytes: int
    max_active_jobs_per_chat: int
    ffmpeg_binary: str
    max_groq_chunk_bytes: int
    audio_chunk_seconds: int
    audio_chunk_overlap_seconds: int
    upload_session_ttl_seconds: int
    max_duration_seconds: int


def media_platform_config_from_env(
    public_base_url: str,
    max_upload_bytes: int,
    upload_chunk_bytes: int,
    ffmpeg_binary: str,
    max_groq_chunk_bytes: int,
    audio_chunk_seconds: int,
    audio_chunk_overlap_seconds: int,
) -> MediaPlatformConfig:
    raw_data_dir = os.getenv("MEDIA_DATA_DIR", "").strip()
    default_data_dir = Path(raw_data_dir).expanduser()
    if not raw_data_dir:
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            default_data_dir = Path(local_app_data) / "VoiceAssistant"
        else:
            default_data_dir = Path("data")

    return MediaPlatformConfig(
        public_base_url=public_base_url.rstrip("/"),
        db_path=Path(os.getenv("MEDIA_DB_PATH", str(default_data_dir / "media_library.sqlite3"))),
        storage_dir=Path(os.getenv("MEDIA_STORAGE_DIR", str(default_data_dir / "storage"))),
        max_upload_bytes=max_upload_bytes,
        upload_chunk_bytes=upload_chunk_bytes,
        max_active_jobs_per_chat=int(os.getenv("MEDIA_MAX_ACTIVE_JOBS", "2")),
        ffmpeg_binary=ffmpeg_binary,
        max_groq_chunk_bytes=max_groq_chunk_bytes,
        audio_chunk_seconds=audio_chunk_seconds,
        audio_chunk_overlap_seconds=audio_chunk_overlap_seconds,
        upload_session_ttl_seconds=int(os.getenv("MEDIA_UPLOAD_SESSION_TTL_SECONDS", "86400")),
        max_duration_seconds=int(os.getenv("MEDIA_MAX_DURATION_SECONDS", str(3 * 60 * 60))),
    )


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def format_seconds(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    rest = seconds_int % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{rest:02d}"
    return f"{minutes:02d}:{rest:02d}"


def subtitle_timestamp(seconds: float, separator: str) -> str:
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    whole = max(0, int(seconds))
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    rest = whole % 60
    return f"{hours:02d}:{minutes:02d}:{rest:02d}{separator}{milliseconds:03d}"


def split_text_for_captions(text: str, max_chars: int = 170) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    captions: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                captions.append(current)
            current = sentence
    if current:
        captions.append(current)
    if not captions and text.strip():
        captions.append(text.strip()[:max_chars])
    return captions


def safe_title_from_name(file_name: str) -> str:
    title = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
    return title or file_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_transcript_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MediaDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._lock = asyncio.Lock()
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS media_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                original_name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                suffix TEXT NOT NULL,
                sha256 TEXT,
                duration_seconds REAL DEFAULT 0,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                original_object TEXT,
                audio_object TEXT,
                txt_object TEXT,
                srt_object TEXT,
                vtt_object TEXT,
                telegram_status_chat_id INTEGER,
                telegram_status_message_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES users(chat_id)
            );

            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                object_key TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS processing_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                raw_text TEXT,
                cleaned_text TEXT,
                text TEXT NOT NULL,
                FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS search_index
            USING fts5(media_id UNINDEXED, chunk_id UNINDEXED, title, text);

            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                description TEXT,
                FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS summaries (
                media_id INTEGER PRIMARY KEY,
                short_summary TEXT,
                detailed_summary TEXT,
                key_points_json TEXT,
                FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                task_text TEXT NOT NULL,
                timestamp REAL DEFAULT 0,
                assignee TEXT,
                due_date TEXT,
                context TEXT,
                FOREIGN KEY(media_id) REFERENCES media_items(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS upload_sessions (
                token TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_dir TEXT NOT NULL,
                total_chunks INTEGER NOT NULL,
                received_chunks_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.migrate_schema()
        self.connection.commit()

    def migrate_schema(self) -> None:
        self.ensure_column("media_items", "telegram_status_chat_id", "INTEGER")
        self.ensure_column("media_items", "telegram_status_message_id", "INTEGER")
        self.ensure_column("transcript_chunks", "raw_text", "TEXT")
        self.ensure_column("transcript_chunks", "cleaned_text", "TEXT")

    def ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    async def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        async with self._lock:
            cursor = self.connection.execute(sql, params)
            self.connection.commit()
            return cursor

    async def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        async with self._lock:
            cursor = self.connection.execute(sql, params)
            return cursor.fetchone()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        async with self._lock:
            cursor = self.connection.execute(sql, params)
            return cursor.fetchall()


class LocalObjectStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_file(self, source: Path, kind: str, media_id: int, suffix: str) -> tuple[str, int]:
        key = f"objects/{media_id}/{kind}{suffix}"
        destination = self.path_for(key)
        shutil.copy2(source, destination)
        return key, destination.stat().st_size

    def save_text(self, text: str, kind: str, media_id: int, suffix: str) -> tuple[str, int]:
        key = f"objects/{media_id}/{kind}{suffix}"
        destination = self.path_for(key)
        destination.write_text(text, encoding="utf-8")
        return key, destination.stat().st_size

    def read_text(self, key: str) -> str:
        return (self.root / key).read_text(encoding="utf-8")


class MediaPlatform:
    def __init__(
        self,
        *,
        bot: Bot,
        config: MediaPlatformConfig,
        run_command: RunCommand,
        get_duration: DurationReader,
        split_audio: SplitAudio,
        transcribe_one: TranscribeOne,
        merge_transcripts: MergeTranscripts,
        create_temp_directory: TempDirectoryFactory,
        describe_error: ErrorFormatter,
        postprocess_transcript: Optional[PostprocessTranscript] = None,
    ) -> None:
        self.bot = bot
        self.config = config
        self.db = MediaDatabase(config.db_path)
        self.storage = LocalObjectStorage(config.storage_dir)
        self.run_command = run_command
        self.get_duration = get_duration
        self.split_audio = split_audio
        self.transcribe_one = transcribe_one
        self.merge_transcripts = merge_transcripts
        self.create_temp_directory = create_temp_directory
        self.describe_error = describe_error
        self.postprocess_transcript = postprocess_transcript
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.worker_loop())

    async def stop(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    def library_url(self, chat_id: int) -> str:
        return f"{self.config.public_base_url}/app?chat_id={chat_id}"

    def media_url(self, media_id: int, timestamp: float = 0) -> str:
        suffix = f"&t={int(timestamp)}" if timestamp else ""
        return f"{self.config.public_base_url}/app/media/{media_id}?{suffix.lstrip('&')}"

    async def register_user(self, chat_id: int) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO users(chat_id, created_at) VALUES(?, ?)",
            (chat_id, utc_now()),
        )

    async def cleanup_expired_upload_sessions(self) -> None:
        rows = await self.db.fetchall("SELECT * FROM upload_sessions")
        now = datetime.utcnow()
        for row in rows:
            created_at = datetime.fromisoformat(row["created_at"])
            if (now - created_at).total_seconds() <= self.config.upload_session_ttl_seconds:
                continue
            chunk_dir = Path(row["chunk_dir"])
            shutil.rmtree(chunk_dir, ignore_errors=True)
            await self.db.execute("DELETE FROM upload_sessions WHERE token = ?", (row["token"],))

    async def active_jobs_for_chat(self, chat_id: int) -> int:
        row = await self.db.fetchone(
            """
            SELECT COUNT(*) AS count
            FROM media_items
            WHERE chat_id = ? AND status NOT IN (?, ?)
            """,
            (chat_id, JOB_STATUS_DONE, JOB_STATUS_FAILED),
        )
        return int(row["count"]) if row else 0

    async def create_upload_session(self, chat_id: int, file_name: str, file_size: int) -> dict:
        await self.cleanup_expired_upload_sessions()
        await self.register_user(chat_id)
        if await self.active_jobs_for_chat(chat_id) >= self.config.max_active_jobs_per_chat:
            raise ValueError("Сейчас уже обрабатывается максимум файлов для этого чата. Дождись завершения.")

        safe_name = self.validate_file_info(file_name, file_size)
        token = uuid.uuid4().hex
        chunk_dir = self.create_temp_directory(prefix=f"media_upload_{token[:8]}_")
        total_chunks = (file_size + self.config.upload_chunk_bytes - 1) // self.config.upload_chunk_bytes
        now = utc_now()
        await self.db.execute(
            """
            INSERT INTO upload_sessions(
                token, chat_id, file_name, file_size, chunk_dir, total_chunks,
                received_chunks_json, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (token, chat_id, safe_name, file_size, str(chunk_dir), total_chunks, "[]", now, now),
        )
        return {
            "token": token,
            "chunk_size": self.config.upload_chunk_bytes,
            "total_chunks": total_chunks,
            "received_chunks": [],
        }

    def validate_file_info(self, file_name: str, file_size: int) -> str:
        safe_name = Path(file_name).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_MEDIA_SUFFIXES:
            raise ValueError("Поддерживаются аудио и видео: .ogg, .opus, .mp3, .m4a, .wav, .webm, .mp4, .mov, .mkv.")
        if file_size <= 0:
            raise ValueError("Файл пустой.")
        if file_size > self.config.max_upload_bytes:
            raise ValueError(f"Файл слишком большой. Максимум: {self.config.max_upload_bytes // (1024 * 1024)} МБ.")
        return safe_name

    async def get_upload_session(self, token: str) -> sqlite3.Row:
        await self.cleanup_expired_upload_sessions()
        row = await self.db.fetchone("SELECT * FROM upload_sessions WHERE token = ?", (token,))
        if row is None:
            raise ValueError("Сессия загрузки не найдена. Открой медиатеку и выбери файл заново.")
        return row

    async def save_upload_chunk(self, token: str, index: int, request: web.Request) -> list[int]:
        session = await self.get_upload_session(token)
        total_chunks = int(session["total_chunks"])
        if index < 0 or index >= total_chunks:
            raise ValueError("Неверный номер части файла.")

        chunk_dir = Path(session["chunk_dir"])
        chunk_dir.mkdir(parents=True, exist_ok=True)
        expected_size = min(
            self.config.upload_chunk_bytes,
            int(session["file_size"]) - index * self.config.upload_chunk_bytes,
        )
        chunk_path = chunk_dir / f"chunk_{index:05d}.part"
        written = 0
        try:
            with chunk_path.open("wb") as output_file:
                async for part in request.content.iter_chunked(1024 * 1024):
                    written += len(part)
                    if written > expected_size:
                        raise ValueError("Часть файла больше ожидаемого размера.")
                    output_file.write(part)
            if written != expected_size:
                raise ValueError("Часть файла загрузилась не полностью.")
        finally:
            if written != expected_size and chunk_path.exists():
                chunk_path.unlink()

        received = set(json.loads(session["received_chunks_json"]))
        received.add(index)
        received_sorted = sorted(received)
        await self.db.execute(
            "UPDATE upload_sessions SET received_chunks_json = ?, updated_at = ? WHERE token = ?",
            (json.dumps(received_sorted), utc_now(), token),
        )
        return received_sorted

    async def complete_upload(self, token: str) -> int:
        session = await self.get_upload_session(token)
        chunk_dir = Path(session["chunk_dir"])
        total_chunks = int(session["total_chunks"])
        received = set(json.loads(session["received_chunks_json"]))
        missing = [index for index in range(total_chunks) if index not in received]
        if missing:
            raise ValueError("Не все части файла загружены.")

        safe_name = str(session["file_name"])
        suffix = Path(safe_name).suffix.lower()
        assembled_path = Path(tempfile.gettempdir()) / f"media_upload_{uuid.uuid4().hex}{suffix}"
        try:
            with assembled_path.open("wb") as output_file:
                for index in range(total_chunks):
                    chunk_path = chunk_dir / f"chunk_{index:05d}.part"
                    if not chunk_path.exists():
                        raise ValueError("Не все части файла найдены.")
                    with chunk_path.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output_file)

            if assembled_path.stat().st_size != int(session["file_size"]):
                raise ValueError("Файл собрался некорректно.")

            media_id = await self.create_media_item(
                chat_id=int(session["chat_id"]),
                file_name=safe_name,
                source_path=assembled_path,
            )
            await self.db.execute("DELETE FROM upload_sessions WHERE token = ?", (token,))
            shutil.rmtree(chunk_dir, ignore_errors=True)
            media = await self.get_media(media_id)
            if media is not None and media["status"] == JOB_STATUS_DONE:
                await self.notify_done(media_id)
            else:
                await self.create_telegram_status_message(media_id)
                await self.queue.put(media_id)
            return media_id
        finally:
            if assembled_path.exists():
                try:
                    assembled_path.unlink()
                except OSError:
                    logging.warning("Could not delete assembled upload %s", assembled_path)

    async def create_media_item(self, chat_id: int, file_name: str, source_path: Path) -> int:
        await self.register_user(chat_id)
        suffix = Path(file_name).suffix.lower()
        media_type = "video" if suffix in VIDEO_SUFFIXES else "audio"
        file_hash = sha256_file(source_path)

        existing = await self.db.fetchone(
            """
            SELECT *
            FROM media_items
            WHERE sha256 = ? AND status = ?
            ORDER BY id DESC LIMIT 1
            """,
            (file_hash, JOB_STATUS_DONE),
        )
        if existing:
            media_id = await self.clone_existing_media(chat_id, existing, file_name)
            return media_id

        now = utc_now()
        cursor = await self.db.execute(
            """
            INSERT INTO media_items(
                chat_id, title, original_name, media_type, suffix, sha256, status, stage,
                progress, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                safe_title_from_name(file_name),
                file_name,
                media_type,
                suffix,
                file_hash,
                JOB_STATUS_UPLOADED,
                "Файл загружен",
                5,
                now,
                now,
            ),
        )
        media_id = int(cursor.lastrowid)
        original_key, size_bytes = self.storage.save_file(source_path, "original", media_id, suffix)
        await self.record_media_file(media_id, "original", original_key, size_bytes)
        await self.update_media(media_id, original_object=original_key)
        await self.create_job(media_id, JOB_STATUS_UPLOADED, "Файл загружен", 5)
        return media_id

    async def clone_existing_media(self, chat_id: int, existing: sqlite3.Row, file_name: str) -> int:
        now = utc_now()
        cursor = await self.db.execute(
            """
            INSERT INTO media_items(
                chat_id, title, original_name, media_type, suffix, sha256, duration_seconds,
                status, stage, progress, original_object, audio_object, txt_object,
                srt_object, vtt_object, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                safe_title_from_name(file_name),
                file_name,
                existing["media_type"],
                existing["suffix"],
                existing["sha256"],
                existing["duration_seconds"],
                JOB_STATUS_DONE,
                "Файл уже был обработан",
                100,
                existing["original_object"],
                existing["audio_object"],
                existing["txt_object"],
                existing["srt_object"],
                existing["vtt_object"],
                now,
                now,
            ),
        )
        media_id = int(cursor.lastrowid)
        original_files = await self.db.fetchall(
            "SELECT kind, object_key, size_bytes FROM media_files WHERE media_id = ?",
            (existing["id"],),
        )
        for file_row in original_files:
            await self.record_media_file(
                media_id,
                file_row["kind"],
                file_row["object_key"],
                int(file_row["size_bytes"]),
            )
        await self.clone_derived_data(media_id, int(existing["id"]), safe_title_from_name(file_name))
        await self.create_job(media_id, JOB_STATUS_DONE, "Дубликат найден", 100)
        return media_id

    async def clone_derived_data(self, new_media_id: int, source_media_id: int, title: str) -> None:
        chunks = await self.db.fetchall(
            "SELECT * FROM transcript_chunks WHERE media_id = ? ORDER BY chunk_index",
            (source_media_id,),
        )
        for chunk in chunks:
            cursor = await self.db.execute(
                """
                INSERT INTO transcript_chunks(
                    media_id, chunk_index, start_time, end_time, raw_text, cleaned_text, text
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_media_id,
                    chunk["chunk_index"],
                    chunk["start_time"],
                    chunk["end_time"],
                    chunk["raw_text"],
                    chunk["cleaned_text"],
                    chunk["text"],
                ),
            )
            await self.db.execute(
                "INSERT INTO search_index(media_id, chunk_id, title, text) VALUES(?, ?, ?, ?)",
                (new_media_id, int(cursor.lastrowid), title, chunk["text"]),
            )

        chapters = await self.db.fetchall("SELECT * FROM chapters WHERE media_id = ?", (source_media_id,))
        for chapter in chapters:
            await self.db.execute(
                """
                INSERT INTO chapters(media_id, title, start_time, end_time, description)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    new_media_id,
                    chapter["title"],
                    chapter["start_time"],
                    chapter["end_time"],
                    chapter["description"],
                ),
            )

        summary = await self.db.fetchone("SELECT * FROM summaries WHERE media_id = ?", (source_media_id,))
        if summary:
            await self.db.execute(
                """
                INSERT INTO summaries(media_id, short_summary, detailed_summary, key_points_json)
                VALUES(?, ?, ?, ?)
                """,
                (
                    new_media_id,
                    summary["short_summary"],
                    summary["detailed_summary"],
                    summary["key_points_json"],
                ),
            )

        tasks = await self.db.fetchall("SELECT * FROM tasks WHERE media_id = ?", (source_media_id,))
        for task in tasks:
            await self.db.execute(
                """
                INSERT INTO tasks(media_id, task_text, timestamp, assignee, due_date, context)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    new_media_id,
                    task["task_text"],
                    task["timestamp"],
                    task["assignee"],
                    task["due_date"],
                    task["context"],
                ),
            )

    async def create_job(self, media_id: int, status: str, stage: str, progress: int) -> int:
        now = utc_now()
        cursor = await self.db.execute(
            """
            INSERT INTO processing_jobs(media_id, status, stage, progress, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (media_id, status, stage, progress, now, now),
        )
        return int(cursor.lastrowid)

    async def record_media_file(self, media_id: int, kind: str, key: str, size_bytes: int) -> None:
        await self.db.execute(
            "INSERT INTO media_files(media_id, kind, object_key, size_bytes, created_at) VALUES(?, ?, ?, ?, ?)",
            (media_id, kind, key, size_bytes, utc_now()),
        )

    async def update_media(self, media_id: int, **fields: object) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = tuple(fields.values()) + (media_id,)
        await self.db.execute(f"UPDATE media_items SET {assignments} WHERE id = ?", values)

    async def update_job(self, media_id: int, status: str, stage: str, progress: int, error: str = "") -> None:
        await self.update_media(media_id, status=status, stage=stage, progress=progress, error=error or None)
        await self.db.execute(
            """
            UPDATE processing_jobs
            SET status = ?, stage = ?, progress = ?, error = ?, updated_at = ?
            WHERE media_id = ?
            """,
            (status, stage, progress, error or None, utc_now(), media_id),
        )
        await self.edit_telegram_status_message(media_id, status, stage, progress, error)

    async def create_telegram_status_message(self, media_id: int) -> None:
        media = await self.get_media(media_id)
        if media is None:
            return
        try:
            message = await self.bot.send_message(
                chat_id=int(media["chat_id"]),
                text=(
                    "Файл принят в медиатеку.\n"
                    f"{media['title']}\n"
                    "Статус: поставлен в очередь"
                ),
            )
            await self.update_media(
                media_id,
                telegram_status_chat_id=message.chat.id,
                telegram_status_message_id=message.message_id,
            )
        except Exception as exc:
            logging.warning("Could not create Telegram media status message: %s", exc)

    async def edit_telegram_status_message(
        self,
        media_id: int,
        status: str,
        stage: str,
        progress: int,
        error: str = "",
    ) -> None:
        media = await self.get_media(media_id)
        if media is None or not media["telegram_status_message_id"]:
            return
        text = (
            f"Медиатека: {media['title']}\n"
            f"Статус: {status}\n"
            f"Этап: {stage}\n"
            f"Прогресс: {progress}%"
        )
        if error:
            text += f"\nОшибка: {error[:500]}"
        try:
            await self.bot.edit_message_text(
                chat_id=int(media["telegram_status_chat_id"]),
                message_id=int(media["telegram_status_message_id"]),
                text=text,
            )
        except Exception as exc:
            logging.warning("Could not edit Telegram media status message: %s", exc)

    async def worker_loop(self) -> None:
        while True:
            media_id = await self.queue.get()
            try:
                await self.process_media(media_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.exception("Media pipeline failed for media_id=%s: %s", media_id, exc)
                await self.update_job(media_id, JOB_STATUS_FAILED, self.describe_error(exc), 100, str(exc))
            finally:
                self.queue.task_done()

    async def process_media(self, media_id: int) -> None:
        media = await self.get_media(media_id)
        if media is None:
            return

        original_path = self.storage.root / media["original_object"]
        temp_files: list[Path] = []
        try:
            await self.update_job(media_id, JOB_STATUS_STORED, "Файл сохранен в хранилище", 15)

            audio_path = await self.prepare_audio(media, original_path)
            temp_files.append(audio_path)
            duration = await self.get_duration(audio_path)
            if duration > self.config.max_duration_seconds:
                raise RuntimeError(
                    f"Файл слишком длинный: {format_seconds(duration)}. "
                    f"Максимум: {format_seconds(self.config.max_duration_seconds)}."
                )
            await self.update_media(media_id, duration_seconds=duration)
            metadata = {
                "media_id": media_id,
                "title": media["title"],
                "original_name": media["original_name"],
                "media_type": media["media_type"],
                "sha256": media["sha256"],
                "duration_seconds": duration,
                "created_at": media["created_at"],
            }
            metadata_key, metadata_size = self.storage.save_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                "metadata",
                media_id,
                ".json",
            )
            await self.record_media_file(media_id, "metadata", metadata_key, metadata_size)
            audio_key, audio_size = self.storage.save_file(audio_path, "audio", media_id, ".mp3")
            await self.record_media_file(media_id, "audio", audio_key, audio_size)
            await self.update_media(media_id, audio_object=audio_key)
            await self.update_job(media_id, JOB_STATUS_AUDIO_EXTRACTED, "Аудио извлечено и нормализовано", 30)

            transcript_rows = await self.transcribe_media(media_id, audio_path, duration)
            full_text = self.merge_transcripts([row["text"] for row in transcript_rows])
            txt_key, txt_size = self.storage.save_text(full_text, "transcript", media_id, ".txt")
            await self.record_media_file(media_id, "txt", txt_key, txt_size)
            await self.update_media(media_id, txt_object=txt_key)

            srt_text = self.build_srt(transcript_rows)
            vtt_text = self.build_vtt(transcript_rows)
            srt_key, srt_size = self.storage.save_text(srt_text, "subtitles", media_id, ".srt")
            vtt_key, vtt_size = self.storage.save_text(vtt_text, "subtitles", media_id, ".vtt")
            await self.record_media_file(media_id, "srt", srt_key, srt_size)
            await self.record_media_file(media_id, "vtt", vtt_key, vtt_size)
            await self.update_media(media_id, srt_object=srt_key, vtt_object=vtt_key)
            await self.update_job(media_id, JOB_STATUS_SUBTITLES_READY, "Субтитры готовы", 72)

            await self.index_transcript(media_id, media["title"], transcript_rows)
            await self.update_job(media_id, JOB_STATUS_INDEXED, "Текст проиндексирован", 82)

            await self.generate_chapters_summary_tasks(media_id, transcript_rows, full_text, duration)
            await self.update_job(media_id, JOB_STATUS_SUMMARY_READY, "Главы, summary и задачи готовы", 92)
            await self.update_job(media_id, JOB_STATUS_DONE, "Обработка завершена", 100)
            await self.notify_done(media_id)
        finally:
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except OSError:
                    logging.warning("Could not delete media temp file %s", temp_file)

    async def prepare_audio(self, media: sqlite3.Row, original_path: Path) -> Path:
        output_path = Path(tempfile.gettempdir()) / f"media_audio_{uuid.uuid4().hex}.mp3"
        await self.run_command(
            [
                self.config.ffmpeg_binary,
                "-y",
                "-i",
                str(original_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
                "-af",
                "loudnorm",
                str(output_path),
            ]
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not create a normalized audio file.")
        return output_path

    async def transcribe_media(self, media_id: int, audio_path: Path, duration: float) -> list[dict]:
        await self.update_job(media_id, JOB_STATUS_TRANSCRIBING, "Расшифровываю аудио", 35)
        transcript_rows: list[dict] = []
        if audio_path.stat().st_size <= self.config.max_groq_chunk_bytes:
            raw_text = await self.transcribe_one(audio_path)
            text = clean_transcript_text(raw_text)
            transcript_rows.append(
                {
                    "chunk_index": 1,
                    "start_time": 0.0,
                    "end_time": duration,
                    "raw_text": raw_text,
                    "cleaned_text": text,
                    "text": text,
                }
            )
        else:
            chunk_dir = self.create_temp_directory(prefix="media_transcribe_chunks_")
            try:
                chunks = await self.split_audio(audio_path, chunk_dir)
                step = max(1, self.config.audio_chunk_seconds - self.config.audio_chunk_overlap_seconds)
                for index, chunk_path in enumerate(chunks, start=1):
                    progress = 35 + int(index / max(1, len(chunks)) * 30)
                    await self.update_job(
                        media_id,
                        JOB_STATUS_TRANSCRIBING,
                        f"Расшифровка: часть {index} из {len(chunks)}",
                        progress,
                    )
                    raw_text = await self.transcribe_one(chunk_path)
                    text = clean_transcript_text(raw_text)
                    start_time = float((index - 1) * step)
                    end_time = min(duration, start_time + self.config.audio_chunk_seconds)
                    transcript_rows.append(
                        {
                            "chunk_index": index,
                            "start_time": start_time,
                            "end_time": end_time,
                            "raw_text": raw_text,
                            "cleaned_text": text,
                            "text": text,
                        }
                    )
            finally:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        await self.db.execute("DELETE FROM transcript_chunks WHERE media_id = ?", (media_id,))
        for row in transcript_rows:
            await self.db.execute(
                """
                INSERT INTO transcript_chunks(media_id, chunk_index, start_time, end_time, raw_text, cleaned_text, text)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    row["chunk_index"],
                    row["start_time"],
                    row["end_time"],
                    row.get("raw_text", row["text"]),
                    row.get("cleaned_text", row["text"]),
                    row["text"],
                ),
            )
        return transcript_rows

    async def index_transcript(self, media_id: int, title: str, transcript_rows: list[dict]) -> None:
        await self.db.execute("DELETE FROM search_index WHERE media_id = ?", (media_id,))
        chunk_records = await self.db.fetchall(
            "SELECT id, chunk_index, text FROM transcript_chunks WHERE media_id = ? ORDER BY chunk_index",
            (media_id,),
        )
        for chunk in chunk_records:
            await self.db.execute(
                "INSERT INTO search_index(media_id, chunk_id, title, text) VALUES(?, ?, ?, ?)",
                (media_id, chunk["id"], title, chunk["text"]),
            )

    async def generate_chapters_summary_tasks(
        self,
        media_id: int,
        transcript_rows: list[dict],
        full_text: str,
        duration: float,
    ) -> None:
        await self.db.execute("DELETE FROM chapters WHERE media_id = ?", (media_id,))
        await self.db.execute("DELETE FROM summaries WHERE media_id = ?", (media_id,))
        await self.db.execute("DELETE FROM tasks WHERE media_id = ?", (media_id,))

        llm_result = await self.try_postprocess_transcript(full_text, transcript_rows, duration)
        chapters = llm_result.get("chapters") if llm_result else None
        if not chapters:
            chapters = self.derive_chapters(transcript_rows, duration)
        for chapter in chapters:
            chapter_title = str(chapter.get("title") or "Глава").strip()[:120]
            chapter_start = float(chapter.get("start_time", 0) or 0)
            chapter_end = float(chapter.get("end_time", min(duration, chapter_start + 60)) or 0)
            chapter_description = str(chapter.get("description") or "").strip()[:500]
            await self.db.execute(
                """
                INSERT INTO chapters(media_id, title, start_time, end_time, description)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    chapter_title,
                    chapter_start,
                    chapter_end,
                    chapter_description,
                ),
            )

        if llm_result:
            short_summary = str(llm_result.get("short_summary") or "").strip()
            detailed_summary = str(llm_result.get("detailed_summary") or "").strip()
            key_points = llm_result.get("key_points") or []
            if not short_summary or not detailed_summary:
                short_summary, detailed_summary, key_points = self.derive_summary(full_text)
        else:
            short_summary, detailed_summary, key_points = self.derive_summary(full_text)
        await self.db.execute(
            """
            INSERT INTO summaries(media_id, short_summary, detailed_summary, key_points_json)
            VALUES(?, ?, ?, ?)
            """,
            (media_id, short_summary, detailed_summary, json.dumps(key_points, ensure_ascii=False)),
        )

        tasks = llm_result.get("tasks") if llm_result else None
        if not tasks:
            tasks = self.derive_tasks(transcript_rows)
        for task in tasks:
            task_text = str(task.get("task_text") or "").strip()
            if not task_text:
                continue
            await self.db.execute(
                """
                INSERT INTO tasks(media_id, task_text, timestamp, assignee, due_date, context)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    task_text[:700],
                    float(task.get("timestamp", 0) or 0),
                    task.get("assignee"),
                    task.get("due_date"),
                    str(task.get("context") or "")[:700],
                ),
            )

    async def try_postprocess_transcript(
        self,
        full_text: str,
        transcript_rows: list[dict],
        duration: float,
    ) -> Optional[dict]:
        if self.postprocess_transcript is None:
            return None
        try:
            result = await self.postprocess_transcript(full_text, transcript_rows, duration)
        except Exception as exc:
            logging.warning("LLM post-processing failed, using fallback: %s", exc)
            return None
        if not isinstance(result, dict):
            return None
        return result

    def derive_chapters(self, transcript_rows: list[dict], duration: float) -> list[dict]:
        if not transcript_rows:
            return []
        chapters: list[dict] = []
        for row in transcript_rows:
            words = row["text"].split()
            title = " ".join(words[:7]).strip(".,!?;:") or f"Часть {row['chunk_index']}"
            chapters.append(
                {
                    "title": title[:80],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "description": row["text"][:240],
                }
            )
        return chapters[:20]

    def derive_summary(self, full_text: str) -> tuple[str, str, list[str]]:
        normalized = re.sub(r"\s+", " ", full_text).strip()
        if not normalized:
            return "Summary не создан: текст пустой.", "", []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        short_summary = " ".join(sentences[:3])[:900]
        detailed_summary = " ".join(sentences[:12])[:4000]
        key_points = sentences[:8]
        return short_summary, detailed_summary, key_points

    def derive_tasks(self, transcript_rows: list[dict]) -> list[dict]:
        tasks: list[dict] = []
        markers = re.compile(r"\b(надо|нужно|сделать|задача|должен|должны|поручить|проверить|дедлайн)\b", re.I)
        for row in transcript_rows:
            sentences = re.split(r"(?<=[.!?])\s+", row["text"])
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and markers.search(sentence):
                    tasks.append(
                        {
                            "task_text": sentence[:500],
                            "timestamp": row["start_time"],
                            "context": row["text"][:500],
                        }
                    )
        return tasks[:30]

    def build_caption_segments(self, transcript_rows: list[dict]) -> list[dict]:
        segments: list[dict] = []
        for row in transcript_rows:
            captions = split_text_for_captions(row["text"])
            if not captions:
                continue
            span = max(1.0, row["end_time"] - row["start_time"])
            segment_duration = span / len(captions)
            for index, caption in enumerate(captions):
                start = row["start_time"] + index * segment_duration
                end = min(row["end_time"], start + segment_duration)
                segments.append({"start": start, "end": end, "text": caption})
        return segments

    def build_srt(self, transcript_rows: list[dict]) -> str:
        lines: list[str] = []
        for index, segment in enumerate(self.build_caption_segments(transcript_rows), start=1):
            lines.append(str(index))
            lines.append(
                f"{subtitle_timestamp(segment['start'], ',')} --> {subtitle_timestamp(segment['end'], ',')}"
            )
            lines.append(segment["text"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def build_vtt(self, transcript_rows: list[dict]) -> str:
        lines = ["WEBVTT", ""]
        for segment in self.build_caption_segments(transcript_rows):
            lines.append(
                f"{subtitle_timestamp(segment['start'], '.')} --> {subtitle_timestamp(segment['end'], '.')}"
            )
            lines.append(segment["text"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    async def notify_done(self, media_id: int) -> None:
        media = await self.get_media(media_id)
        if media is None:
            return
        summary = await self.db.fetchone("SELECT short_summary FROM summaries WHERE media_id = ?", (media_id,))
        text = (
            "Медиафайл обработан.\n"
            f"{media['title']} — {format_seconds(media['duration_seconds'])}\n\n"
        )
        if summary and summary["short_summary"]:
            text += f"Кратко: {summary['short_summary'][:1200]}\n\n"
        text += f"Открыть: {self.media_url(media_id)}"
        try:
            await self.bot.send_message(chat_id=int(media["chat_id"]), text=text)
        except Exception as exc:
            logging.warning("Could not send media completion message: %s", exc)

    async def get_media(self, media_id: int) -> Optional[sqlite3.Row]:
        return await self.db.fetchone("SELECT * FROM media_items WHERE id = ?", (media_id,))

    async def list_media(self, chat_id: Optional[int] = None) -> list[sqlite3.Row]:
        if chat_id:
            return await self.db.fetchall(
                "SELECT * FROM media_items WHERE chat_id = ? ORDER BY id DESC LIMIT 100",
                (chat_id,),
            )
        return await self.db.fetchall("SELECT * FROM media_items ORDER BY id DESC LIMIT 100")

    def fts_query(self, query: str) -> str:
        terms = re.findall(r"[\wа-яА-ЯёЁ]+", query)
        return " AND ".join(terms)

    async def search(self, query: str, chat_id: Optional[int] = None, limit: int = 10) -> list[sqlite3.Row]:
        fts = self.fts_query(query)
        if not fts:
            return []
        try:
            if chat_id:
                rows = await self.db.fetchall(
                    """
                    SELECT m.id AS media_id, m.title, c.start_time, c.text
                    FROM search_index s
                    JOIN media_items m ON m.id = s.media_id
                    JOIN transcript_chunks c ON c.id = s.chunk_id
                    WHERE s.text MATCH ? AND m.chat_id = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts, chat_id, limit),
                )
            else:
                rows = await self.db.fetchall(
                """
                SELECT m.id AS media_id, m.title, c.start_time, c.text
                FROM search_index s
                JOIN media_items m ON m.id = s.media_id
                JOIN transcript_chunks c ON c.id = s.chunk_id
                WHERE s.text MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts, limit),
                )
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass

        terms = re.findall(r"[\wа-яА-ЯёЁ]+", query.lower())
        if not terms:
            return []
        where_terms = " AND ".join("LOWER(c.text) LIKE ?" for _ in terms)
        params: list[object] = [f"%{term}%" for term in terms]
        chat_filter = ""
        if chat_id:
            chat_filter = " AND m.chat_id = ?"
            params.append(chat_id)
        params.append(limit)
        return await self.db.fetchall(
            f"""
            SELECT m.id AS media_id, m.title, c.start_time, c.text
            FROM transcript_chunks c
            JOIN media_items m ON m.id = c.media_id
            WHERE {where_terms}{chat_filter}
            LIMIT ?
            """,
            tuple(params),
        )


def html_page(title: str, body: str) -> web.Response:
    return web.Response(
        text=f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f4f7fb; color: #182230; font-family: Arial, sans-serif; }}
    header, main {{ max-width: 1040px; margin: 0 auto; padding: 18px; }}
    header {{ display: flex; justify-content: space-between; gap: 14px; align-items: center; }}
    h1 {{ margin: 0; font-size: 26px; }}
    h2 {{ margin: 24px 0 12px; font-size: 20px; }}
    a {{ color: #0f6b55; text-decoration: none; }}
    .panel {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin: 12px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .muted {{ color: #667085; }}
    .status {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #eef4ff; font-size: 13px; }}
    input, button {{ font-size: 16px; }}
    input[type="text"], input[type="file"] {{ width: 100%; padding: 10px; border: 1px solid #cdd5df; border-radius: 8px; box-sizing: border-box; }}
    button {{ width: 100%; min-height: 44px; border: 0; border-radius: 8px; background: #1f7a5b; color: white; font-weight: 700; cursor: pointer; }}
    .progress {{ height: 10px; background: #e4eaf1; border-radius: 999px; overflow: hidden; }}
    .bar {{ width: 0%; height: 100%; background: #1f7a5b; }}
    .snippet {{ color: #344054; line-height: 1.45; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    audio, video {{ width: 100%; margin: 10px 0; background: #111827; border-radius: 8px; }}
    .transcript-line {{ border-bottom: 1px solid #edf1f5; padding: 10px 0; cursor: pointer; }}
    .time {{ font-variant-numeric: tabular-nums; color: #0f6b55; font-weight: 700; }}
  </style>
</head>
<body>
{body}
</body>
</html>""",
        content_type="text/html",
    )


def get_platform(request: web.Request) -> MediaPlatform:
    return request.app["media_platform"]


async def app_page(request: web.Request) -> web.Response:
    platform = get_platform(request)
    chat_id_raw = request.query.get("chat_id", "").strip()
    chat_id = int(chat_id_raw) if chat_id_raw.isdigit() else None
    query = request.query.get("q", "").strip()
    media_items = await platform.list_media(chat_id)
    results = await platform.search(query, chat_id, 10) if query else []

    upload_action = "/app/upload/init"
    chat_value = str(chat_id or "")
    result_html = ""
    if query:
        result_cards = []
        for result in results:
            url = f"/app/media/{result['media_id']}?t={int(result['start_time'])}"
            result_cards.append(
                f"""<div class="panel">
  <div><a href="{url}"><b>{escape(result['title'])}</b> — {format_seconds(result['start_time'])}</a></div>
  <div class="snippet">{escape(result['text'][:420])}</div>
</div>"""
            )
        result_html = "<h2>Результаты поиска</h2>" + ("".join(result_cards) or '<div class="panel muted">Ничего не найдено.</div>')

    media_cards = []
    for item in media_items:
        media_cards.append(
            f"""<div class="panel">
  <div class="row"><b>{escape(item['title'])}</b> <span class="status">{escape(item['status'])}</span></div>
  <div class="muted">{escape(item['original_name'])} · {format_seconds(item['duration_seconds'] or 0)} · {item['progress']}%</div>
  <div class="progress"><div class="bar" style="width:{int(item['progress'])}%"></div></div>
  <p><a href="/app/media/{item['id']}">Открыть</a></p>
</div>"""
        )

    body = f"""<header>
  <h1>Медиатека</h1>
  <a href="/health">health</a>
</header>
<main>
  <section class="panel">
    <h2>Загрузить лекцию, видео, подкаст или встречу</h2>
    <input id="fileInput" type="file" accept=".ogg,.opus,.mp3,.m4a,.wav,.webm,.mp4,.mov,.mkv,audio/*,video/*">
    <input id="chatId" type="hidden" value="{escape(chat_value)}">
    <p id="fileInfo" class="muted">Файл не выбран.</p>
    <button id="uploadButton" disabled>Загрузить в медиатеку</button>
    <div class="progress" style="margin-top:12px"><div id="uploadBar" class="bar"></div></div>
    <p id="uploadStatus" class="muted"></p>
  </section>
  <section class="panel">
    <form method="get" action="/app">
      <input type="hidden" name="chat_id" value="{escape(chat_value)}">
      <input type="text" name="q" value="{escape(query)}" placeholder="Поиск по содержимому: нормализация базы данных">
      <p><button type="submit">Искать</button></p>
    </form>
  </section>
  {result_html}
  <h2>Файлы</h2>
  <div class="grid">{''.join(media_cards) or '<div class="panel muted">Пока нет файлов.</div>'}</div>
</main>
<script>
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const fileInfo = document.getElementById('fileInfo');
const uploadStatus = document.getElementById('uploadStatus');
const uploadBar = document.getElementById('uploadBar');
const chatId = document.getElementById('chatId').value || '0';
let selectedFile = null;
function setProgress(value) {{ uploadBar.style.width = `${{Math.round(value)}}%`; }}
async function jsonRequest(url, options) {{
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({{}}));
  if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
  return data;
}}
function putChunk(url, blob) {{
  return new Promise((resolve, reject) => {{
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', url);
    xhr.onload = () => xhr.status >= 200 && xhr.status < 300 ? resolve(JSON.parse(xhr.responseText || '{{}}')) : reject(new Error('Не удалось загрузить часть файла'));
    xhr.onerror = () => reject(new Error('Соединение оборвалось'));
    xhr.send(blob);
  }});
}}
fileInput.addEventListener('change', () => {{
  selectedFile = fileInput.files[0];
  uploadButton.disabled = !selectedFile;
  fileInfo.textContent = selectedFile ? `${{selectedFile.name}} · ${{Math.round(selectedFile.size / 1024 / 1024)}} МБ` : 'Файл не выбран.';
}});
uploadButton.addEventListener('click', async () => {{
  if (!selectedFile) return;
  uploadButton.disabled = true;
  uploadStatus.textContent = 'Подготовка загрузки...';
  try {{
    const init = await jsonRequest('{upload_action}', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ chat_id: Number(chatId), file_name: selectedFile.name, file_size: selectedFile.size }})
    }});
    const received = new Set(init.received_chunks || []);
    for (let index = 0; index < init.total_chunks; index++) {{
      if (!received.has(index)) {{
        const start = index * init.chunk_size;
        const end = Math.min(selectedFile.size, start + init.chunk_size);
        await putChunk(`/app/upload/${{init.token}}/chunk/${{index}}`, selectedFile.slice(start, end));
      }}
      const percent = ((index + 1) / init.total_chunks) * 100;
      setProgress(percent);
      uploadStatus.textContent = `Загрузка: ${{Math.round(percent)}}%`;
    }}
    const complete = await jsonRequest(`/app/upload/${{init.token}}/complete`, {{ method: 'POST' }});
    uploadStatus.textContent = 'Файл принят. Обработка идет в фоне.';
    window.location.href = `/app/media/${{complete.media_id}}`;
  }} catch (error) {{
    uploadStatus.textContent = error.message || 'Ошибка загрузки';
    uploadButton.disabled = false;
  }}
}});
</script>"""
    return html_page("Медиатека", body)


async def app_upload_init(request: web.Request) -> web.Response:
    platform = get_platform(request)
    try:
        payload = await request.json()
        chat_id = int(payload.get("chat_id") or 0)
        if chat_id <= 0:
            raise ValueError("Открой медиатеку из Telegram-кнопки, чтобы я понял твой чат.")
        data = await platform.create_upload_session(
            chat_id=chat_id,
            file_name=str(payload.get("file_name", "")),
            file_size=int(payload.get("file_size", 0)),
        )
        return web.json_response(data)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        logging.exception("Media upload init failed: %s", exc)
        return web.json_response({"error": "Не удалось начать загрузку."}, status=500)


async def app_upload_chunk(request: web.Request) -> web.Response:
    platform = get_platform(request)
    try:
        token = request.match_info["token"]
        index = int(request.match_info["index"])
        received = await platform.save_upload_chunk(token, index, request)
        return web.json_response({"received_chunks": received})
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        logging.exception("Media upload chunk failed: %s", exc)
        return web.json_response({"error": "Не удалось принять часть файла."}, status=500)


async def app_upload_complete(request: web.Request) -> web.Response:
    platform = get_platform(request)
    try:
        media_id = await platform.complete_upload(request.match_info["token"])
        return web.json_response({"media_id": media_id})
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        logging.exception("Media upload complete failed: %s", exc)
        return web.json_response({"error": "Не удалось завершить загрузку."}, status=500)


async def media_detail_page(request: web.Request) -> web.Response:
    platform = get_platform(request)
    media_id = int(request.match_info["media_id"])
    media = await platform.get_media(media_id)
    if media is None:
        return html_page("Не найдено", "<main><div class='panel'>Файл не найден.</div></main>")

    timestamp = int(request.query.get("t", "0") or 0)
    chunks = await platform.db.fetchall(
        "SELECT * FROM transcript_chunks WHERE media_id = ? ORDER BY chunk_index",
        (media_id,),
    )
    chapters = await platform.db.fetchall("SELECT * FROM chapters WHERE media_id = ? ORDER BY start_time", (media_id,))
    tasks = await platform.db.fetchall("SELECT * FROM tasks WHERE media_id = ? ORDER BY timestamp", (media_id,))
    summary = await platform.db.fetchone("SELECT * FROM summaries WHERE media_id = ?", (media_id,))

    source_kind = "original" if media["media_type"] == "video" else "audio"
    media_tag = "video" if media["media_type"] == "video" else "audio"
    player = f"""<{media_tag} id="player" controls src="/app/media/{media_id}/file/{source_kind}#t={timestamp}"></{media_tag}>"""
    download_links = " ".join(
        f'<a href="/app/media/{media_id}/download/{kind}">{kind.upper()}</a>'
        for kind in ("txt", "srt", "vtt")
        if media[f"{kind}_object"]
    )
    if tasks:
        download_links += f' <a href="/app/media/{media_id}/download/tasks.txt">TASKS.TXT</a>'
    transcript_html = "".join(
        f"""<div class="transcript-line" data-time="{int(chunk['start_time'])}">
  <span class="time">{format_seconds(chunk['start_time'])}</span>
  {escape(chunk['text'])}
</div>"""
        for chunk in chunks
    )
    chapter_html = "".join(
        f"""<div class="panel">
  <a href="/app/media/{media_id}?t={int(chapter['start_time'])}"><b>{escape(chapter['title'])}</b> — {format_seconds(chapter['start_time'])}</a>
  <div class="muted">{escape(chapter['description'] or '')}</div>
</div>"""
        for chapter in chapters
    )
    task_html = "".join(
        f"""<div class="panel">
  <span class="time">{format_seconds(task['timestamp'])}</span> {escape(task['task_text'])}
</div>"""
        for task in tasks
    )
    summary_html = ""
    if summary:
        points = json.loads(summary["key_points_json"] or "[]")
        summary_html = f"""<section class="panel">
  <h2>Краткое содержание</h2>
  <p>{escape(summary['short_summary'] or '')}</p>
  <h2>Ключевые мысли</h2>
  <ul>{''.join(f'<li>{escape(point)}</li>' for point in points)}</ul>
</section>"""

    body = f"""<header>
  <h1>{escape(media['title'])}</h1>
  <a href="/app?chat_id={media['chat_id']}">Медиатека</a>
</header>
<main>
  <section class="panel">
    <div class="row"><span id="jobStatus" class="status">{escape(media['status'])}</span><span id="jobStage" class="muted">{media['progress']}% · {escape(media['stage'])}</span></div>
    <div class="progress" style="margin:10px 0"><div id="jobBar" class="bar" style="width:{int(media['progress'])}%"></div></div>
    {player}
    <p>{download_links}</p>
  </section>
  {summary_html}
  <h2>Главы</h2>
  {chapter_html or '<div class="panel muted">Главы появятся после обработки.</div>'}
  <h2>Задачи</h2>
  {task_html or '<div class="panel muted">Задачи не найдены.</div>'}
  <h2>Расшифровка</h2>
  <section class="panel">{transcript_html or 'Расшифровка еще не готова.'}</section>
</main>
<script>
const player = document.getElementById('player');
document.querySelectorAll('.transcript-line').forEach((line) => {{
  line.addEventListener('click', () => {{
    player.currentTime = Number(line.dataset.time || 0);
    player.play();
  }});
}});
const jobStatus = document.getElementById('jobStatus');
const jobStage = document.getElementById('jobStage');
const jobBar = document.getElementById('jobBar');
if (window.EventSource && jobStatus && !['DONE', 'FAILED'].includes(jobStatus.textContent)) {{
  const events = new EventSource('/app/media/{media_id}/events');
  events.onmessage = (event) => {{
    const data = JSON.parse(event.data);
    jobStatus.textContent = data.status;
    jobStage.textContent = `${{data.progress}}% · ${{data.stage || ''}}`;
    jobBar.style.width = `${{data.progress || 0}}%`;
    if (data.status === 'DONE' || data.status === 'FAILED') {{
      events.close();
      if (data.status === 'DONE') window.setTimeout(() => window.location.reload(), 1200);
    }}
  }};
}}
</script>"""
    return html_page(media["title"], body)


async def media_file(request: web.Request) -> web.FileResponse:
    platform = get_platform(request)
    media = await platform.get_media(int(request.match_info["media_id"]))
    if media is None:
        raise web.HTTPNotFound()
    kind = request.match_info["kind"]
    key = media[f"{kind}_object"] if kind in {"original", "audio", "txt", "srt", "vtt"} else None
    if not key:
        raise web.HTTPNotFound()
    return web.FileResponse(platform.storage.root / key)


async def media_download(request: web.Request) -> web.FileResponse:
    return await media_file(request)


async def media_tasks_download(request: web.Request) -> web.Response:
    platform = get_platform(request)
    media_id = int(request.match_info["media_id"])
    media = await platform.get_media(media_id)
    if media is None:
        raise web.HTTPNotFound()
    tasks = await platform.db.fetchall("SELECT * FROM tasks WHERE media_id = ? ORDER BY timestamp", (media_id,))
    lines = [f"Задачи: {media['title']}", ""]
    for task in tasks:
        lines.append(f"[{format_seconds(task['timestamp'])}] {task['task_text']}")
        if task["assignee"]:
            lines.append(f"Исполнитель: {task['assignee']}")
        if task["due_date"]:
            lines.append(f"Дедлайн: {task['due_date']}")
        if task["context"]:
            lines.append(f"Контекст: {task['context']}")
        lines.append("")
    return web.Response(
        text="\n".join(lines).strip() + "\n",
        content_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="tasks_{media_id}.txt"'},
    )


async def media_events(request: web.Request) -> web.StreamResponse:
    platform = get_platform(request)
    media_id = int(request.match_info["media_id"])
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)
    last_payload = ""
    for _ in range(300):
        media = await platform.get_media(media_id)
        if media is None:
            break
        payload = json.dumps(
            {
                "status": media["status"],
                "stage": media["stage"],
                "progress": media["progress"],
                "error": media["error"],
            },
            ensure_ascii=False,
        )
        if payload != last_payload:
            await response.write(f"data: {payload}\n\n".encode("utf-8"))
            last_payload = payload
        if media["status"] in {JOB_STATUS_DONE, JOB_STATUS_FAILED}:
            break
        await asyncio.sleep(1)
    await response.write_eof()
    return response


def register_media_platform_routes(app: web.Application, platform: MediaPlatform) -> None:
    app["media_platform"] = platform
    app.router.add_get("/app", app_page)
    app.router.add_post("/app/upload/init", app_upload_init)
    app.router.add_put("/app/upload/{token}/chunk/{index}", app_upload_chunk)
    app.router.add_post("/app/upload/{token}/complete", app_upload_complete)
    app.router.add_get("/app/media/{media_id}", media_detail_page)
    app.router.add_get("/app/media/{media_id}/events", media_events)
    app.router.add_get("/events/{media_id}", media_events)
    app.router.add_get("/app/media/{media_id}/file/{kind}", media_file)
    app.router.add_get("/app/media/{media_id}/download/tasks.txt", media_tasks_download)
    app.router.add_get("/app/media/{media_id}/download/{kind}", media_download)

    async def startup(_: web.Application) -> None:
        await platform.start()

    async def cleanup(_: web.Application) -> None:
        await platform.stop()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
