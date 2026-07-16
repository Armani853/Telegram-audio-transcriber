import asyncio
from dataclasses import dataclass, replace
from html import escape
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
from urllib.parse import quote

from aiohttp import web


RunCommand = Callable[[list[str], Optional[int]], Awaitable[str]]
BaseCommand = Callable[[], list[str]]

VIDEO_QUALITIES = (360, 720, 1080)
QUALITY_AUDIO = "audio"


@dataclass(frozen=True)
class YouTubeDownloadConfig:
    public_base_url: str
    storage_dir: Path
    db_path: Path
    ttl_seconds: int
    request_ttl_seconds: int
    timeout_seconds: int
    telegram_direct_limit_bytes: int
    concurrent_fragments: int
    max_concurrent_downloads: int


@dataclass
class YouTubeDownloadRequest:
    request_id: str
    chat_id: int
    url: str
    video_id: str
    title: str
    duration: float
    available_qualities: tuple[str, ...]
    estimated_sizes: dict[str, int]
    created_at: float


@dataclass(frozen=True)
class YouTubeDownloadRecord:
    cache_key: str
    token: str
    video_id: str
    quality: str
    title: str
    file_name: str
    file_path: Path
    file_size: int
    width: int
    height: int
    duration: float
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class YouTubeDownloadNotification:
    notification_id: int
    chat_id: int
    token: str
    text: str
    attempts: int
    next_attempt_at: float


def youtube_download_config_from_env(
    public_base_url: str,
    default_data_dir: Path,
) -> YouTubeDownloadConfig:
    storage_dir = Path(
        os.getenv(
            "YOUTUBE_DOWNLOAD_STORAGE_DIR",
            str(default_data_dir / "youtube_downloads"),
        )
    ).expanduser()
    return YouTubeDownloadConfig(
        public_base_url=public_base_url.rstrip("/"),
        storage_dir=storage_dir,
        db_path=Path(
            os.getenv(
                "YOUTUBE_DOWNLOAD_DB_PATH",
                str(default_data_dir / "youtube_downloads.sqlite3"),
            )
        ).expanduser(),
        ttl_seconds=max(3600, int(os.getenv("YOUTUBE_DOWNLOAD_TTL_SECONDS", "259200"))),
        request_ttl_seconds=max(
            300,
            int(os.getenv("YOUTUBE_DOWNLOAD_REQUEST_TTL_SECONDS", "1800")),
        ),
        timeout_seconds=max(
            300,
            int(os.getenv("YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS", str(6 * 60 * 60))),
        ),
        telegram_direct_limit_bytes=max(
            0,
            int(os.getenv("YOUTUBE_TELEGRAM_DIRECT_LIMIT_BYTES", str(49 * 1024 * 1024))),
        ),
        concurrent_fragments=max(
            1,
            int(os.getenv("YOUTUBE_DOWNLOAD_CONCURRENT_FRAGMENTS", "8")),
        ),
        max_concurrent_downloads=max(
            1,
            int(os.getenv("YOUTUBE_DOWNLOAD_MAX_CONCURRENT", "2")),
        ),
    )


def safe_download_title(title: str, max_length: int = 100) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return (cleaned or "youtube_video")[:max_length]


def format_file_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            precision = 0 if unit in {"Б", "КБ"} else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return f"{value:.1f} ТБ"


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def exact_video_qualities(metadata: dict) -> tuple[str, ...]:
    mp4_video_heights: set[int] = set()
    combined_mp4_heights: set[int] = set()
    has_audio = False
    has_m4a_audio = False
    for item in metadata.get("formats") or []:
        if not isinstance(item, dict):
            continue
        has_video_stream = str(item.get("vcodec") or "none") != "none"
        has_audio_stream = str(item.get("acodec") or "none") != "none"
        extension = str(item.get("ext") or "").lower()
        if has_video_stream and extension == "mp4":
            try:
                height = int(item.get("height") or 0)
            except (TypeError, ValueError):
                height = 0
            if height in VIDEO_QUALITIES:
                mp4_video_heights.add(height)
                if has_audio_stream:
                    combined_mp4_heights.add(height)
        if has_audio_stream:
            has_audio = True
            if not has_video_stream and extension == "m4a":
                has_m4a_audio = True
    result = [
        str(height)
        for height in VIDEO_QUALITIES
        if height in mp4_video_heights
        and (has_m4a_audio or height in combined_mp4_heights)
    ]
    if has_audio:
        result.append(QUALITY_AUDIO)
    return tuple(result)


def estimated_quality_sizes(metadata: dict) -> dict[str, int]:
    formats = [item for item in (metadata.get("formats") or []) if isinstance(item, dict)]
    audio_sizes = [
        int(item.get("filesize") or item.get("filesize_approx") or 0)
        for item in formats
        if str(item.get("acodec") or "none") != "none"
        and str(item.get("vcodec") or "none") == "none"
    ]
    best_audio = max(audio_sizes, default=0)
    estimates: dict[str, int] = {QUALITY_AUDIO: best_audio}
    for height in VIDEO_QUALITIES:
        video_sizes = [
            int(item.get("filesize") or item.get("filesize_approx") or 0)
            for item in formats
            if str(item.get("vcodec") or "none") != "none"
            and int(item.get("height") or 0) == height
        ]
        if video_sizes:
            estimates[str(height)] = max(video_sizes) + best_audio
    return estimates


def build_youtube_media_download_command(
    base_command: list[str],
    url: str,
    quality: str,
    destination_dir: Path,
    concurrent_fragments: int,
) -> list[str]:
    output = str(destination_dir / "download.%(ext)s")
    common = [
        *base_command,
        "--no-playlist",
        "--concurrent-fragments",
        str(concurrent_fragments),
        "--newline",
        "--no-progress",
    ]
    if quality == QUALITY_AUDIO:
        return [
            *common,
            "-f",
            "bestaudio[ext=m4a]/bestaudio",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",
            "-o",
            output,
            url,
        ]

    height = int(quality)
    return [
        *common,
        "-f",
        (
            f"bestvideo[height={height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height={height}][ext=mp4]"
        ),
        "--merge-output-format",
        "mp4",
        "-o",
        output,
        url,
    ]


class YouTubeDownloadService:
    def __init__(
        self,
        config: YouTubeDownloadConfig,
        run_command: RunCommand,
        base_command: BaseCommand,
        ffprobe_binary: str,
        fallback_base_command: Optional[BaseCommand] = None,
    ) -> None:
        self.config = config
        self.run_command = run_command
        self.base_command = base_command
        self.fallback_base_command = fallback_base_command
        self.ffprobe_binary = ffprobe_binary
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.config.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._db_lock = asyncio.Lock()
        self._download_semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._records_by_cache_key: dict[str, YouTubeDownloadRecord] = {}
        self._records_by_token: dict[str, YouTubeDownloadRecord] = {}
        self.requests: dict[str, YouTubeDownloadRequest] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS youtube_downloads (
                cache_key TEXT PRIMARY KEY,
                token TEXT UNIQUE NOT NULL,
                video_id TEXT NOT NULL,
                quality TEXT NOT NULL,
                title TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                duration REAL NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_youtube_download_token ON youtube_downloads(token)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_youtube_download_expiry ON youtube_downloads(expires_at)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS youtube_download_notifications (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                text TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                delivered_at REAL,
                created_at REAL NOT NULL,
                UNIQUE(chat_id, token)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_youtube_notification_pending
            ON youtube_download_notifications(delivered_at, next_attempt_at)
            """
        )
        self.connection.commit()
        now = time.time()
        rows = self.connection.execute(
            "SELECT * FROM youtube_downloads WHERE expires_at > ?",
            (now,),
        ).fetchall()
        for row in rows:
            record = self._record_from_row(row)
            if not record.file_path.is_file():
                continue
            self._records_by_cache_key[record.cache_key] = record
            self._records_by_token[record.token] = record

    async def start(self) -> None:
        await self.cleanup_expired()
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        self.connection.close()

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(min(3600, max(300, self.config.ttl_seconds // 4)))
            await self.cleanup_expired()

    def create_request(self, chat_id: int, url: str, metadata: dict) -> YouTubeDownloadRequest:
        self.cleanup_requests()
        request = YouTubeDownloadRequest(
            request_id=uuid.uuid4().hex[:16],
            chat_id=chat_id,
            url=url,
            video_id=str(metadata.get("id") or "").strip(),
            title=str(metadata.get("title") or "YouTube video")[:180],
            duration=float(metadata.get("duration") or 0),
            available_qualities=exact_video_qualities(metadata),
            estimated_sizes=estimated_quality_sizes(metadata),
            created_at=time.time(),
        )
        if not request.video_id:
            raise ValueError("YouTube не вернул идентификатор видео.")
        if not request.available_qualities:
            raise ValueError("YouTube не предоставил доступные форматы для скачивания.")
        self.requests[request.request_id] = request
        return request

    def cleanup_requests(self) -> None:
        now = time.time()
        expired = [
            request_id
            for request_id, request in self.requests.items()
            if now - request.created_at > self.config.request_ttl_seconds
        ]
        for request_id in expired:
            self.requests.pop(request_id, None)

    def get_request(self, request_id: str, chat_id: int) -> Optional[YouTubeDownloadRequest]:
        self.cleanup_requests()
        request = self.requests.get(request_id)
        if request is None or request.chat_id != chat_id:
            return None
        return request

    def landing_url(self, record: YouTubeDownloadRecord) -> str:
        return f"{self.config.public_base_url}/youtube-download/{record.token}"

    def file_url(self, record: YouTubeDownloadRecord) -> str:
        return f"{self.config.public_base_url}/youtube-download/{record.token}/file"

    async def _fetchone(self, sql: str, params: tuple) -> Optional[sqlite3.Row]:
        async with self._db_lock:
            return self.connection.execute(sql, params).fetchone()

    async def _execute(self, sql: str, params: tuple = ()) -> None:
        async with self._db_lock:
            self.connection.execute(sql, params)
            self.connection.commit()

    def _record_from_row(self, row: sqlite3.Row) -> YouTubeDownloadRecord:
        return YouTubeDownloadRecord(
            cache_key=row["cache_key"],
            token=row["token"],
            video_id=row["video_id"],
            quality=row["quality"],
            title=row["title"],
            file_name=row["file_name"],
            file_path=Path(row["file_path"]),
            file_size=int(row["file_size"]),
            width=int(row["width"]),
            height=int(row["height"]),
            duration=float(row["duration"]),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    async def get_cached(self, video_id: str, quality: str) -> Optional[YouTubeDownloadRecord]:
        cache_key = f"{video_id}:{quality}"
        memory_record = self._records_by_cache_key.get(cache_key)
        if memory_record is not None:
            if memory_record.expires_at <= time.time() or not memory_record.file_path.is_file():
                await self._delete_record(memory_record)
                return None
            return await self.touch_record(memory_record)
        row = await self._fetchone(
            "SELECT * FROM youtube_downloads WHERE cache_key = ?",
            (cache_key,),
        )
        if row is None:
            return None
        record = self._record_from_row(row)
        if record.expires_at <= time.time() or not record.file_path.is_file():
            await self._delete_record(record)
            return None
        self._records_by_cache_key[record.cache_key] = record
        self._records_by_token[record.token] = record
        return await self.touch_record(record)

    async def get_by_token(self, token: str) -> Optional[YouTubeDownloadRecord]:
        memory_record = self._records_by_token.get(token)
        if memory_record is not None:
            if memory_record.expires_at <= time.time() or not memory_record.file_path.is_file():
                await self._delete_record(memory_record)
                return None
            return await self.touch_record(memory_record)
        row = await self._fetchone(
            "SELECT * FROM youtube_downloads WHERE token = ?",
            (token,),
        )
        if row is None:
            return None
        record = self._record_from_row(row)
        if record.expires_at <= time.time() or not record.file_path.is_file():
            await self._delete_record(record)
            return None
        self._records_by_cache_key[record.cache_key] = record
        self._records_by_token[record.token] = record
        return await self.touch_record(record)

    async def touch_record(self, record: YouTubeDownloadRecord) -> YouTubeDownloadRecord:
        """Keep an actively opened or resumed download alive for another full TTL."""
        now = time.time()
        refresh_window = max(60, self.config.ttl_seconds // 10)
        target_expiry = now + self.config.ttl_seconds
        if record.expires_at >= target_expiry - refresh_window:
            return record
        new_expiry = max(record.expires_at, target_expiry)
        await self._execute(
            "UPDATE youtube_downloads SET expires_at = ? WHERE cache_key = ?",
            (new_expiry, record.cache_key),
        )
        touched = replace(record, expires_at=new_expiry)
        self._records_by_cache_key[touched.cache_key] = touched
        self._records_by_token[touched.token] = touched
        return touched

    def _notification_from_row(self, row: sqlite3.Row) -> YouTubeDownloadNotification:
        return YouTubeDownloadNotification(
            notification_id=int(row["notification_id"]),
            chat_id=int(row["chat_id"]),
            token=str(row["token"]),
            text=str(row["text"]),
            attempts=int(row["attempts"]),
            next_attempt_at=float(row["next_attempt_at"]),
        )

    async def queue_notification(
        self,
        chat_id: int,
        token: str,
        text: str,
    ) -> YouTubeDownloadNotification:
        """Persist delivery before contacting Telegram so a result cannot be lost."""
        now = time.time()
        first_retry_at = now + 15
        await self._execute(
            """
            INSERT INTO youtube_download_notifications(
                chat_id, token, text, attempts, next_attempt_at, delivered_at, created_at
            ) VALUES(?, ?, ?, 0, ?, NULL, ?)
            ON CONFLICT(chat_id, token) DO UPDATE SET
                text = excluded.text,
                attempts = 0,
                next_attempt_at = excluded.next_attempt_at,
                delivered_at = NULL
            """,
            (chat_id, token, text, first_retry_at, now),
        )
        row = await self._fetchone(
            """
            SELECT * FROM youtube_download_notifications
            WHERE chat_id = ? AND token = ?
            """,
            (chat_id, token),
        )
        if row is None:
            raise RuntimeError("Could not persist YouTube result notification.")
        return self._notification_from_row(row)

    async def pending_notifications(
        self,
        limit: int = 20,
    ) -> list[YouTubeDownloadNotification]:
        now = time.time()
        async with self._db_lock:
            rows = self.connection.execute(
                """
                SELECT * FROM youtube_download_notifications
                WHERE delivered_at IS NULL AND next_attempt_at <= ?
                ORDER BY next_attempt_at, notification_id
                LIMIT ?
                """,
                (now, max(1, limit)),
            ).fetchall()
        return [self._notification_from_row(row) for row in rows]

    async def mark_notification_delivered(self, notification_id: int) -> None:
        await self._execute(
            """
            UPDATE youtube_download_notifications
            SET delivered_at = ?
            WHERE notification_id = ?
            """,
            (time.time(), notification_id),
        )

    async def reschedule_notification(
        self,
        notification: YouTubeDownloadNotification,
    ) -> None:
        attempts = notification.attempts + 1
        delay_seconds = min(3600, 15 * (2 ** min(attempts - 1, 8)))
        await self._execute(
            """
            UPDATE youtube_download_notifications
            SET attempts = ?, next_attempt_at = ?
            WHERE notification_id = ? AND delivered_at IS NULL
            """,
            (attempts, time.time() + delay_seconds, notification.notification_id),
        )

    async def _delete_record(self, record: YouTubeDownloadRecord) -> None:
        self._records_by_cache_key.pop(record.cache_key, None)
        self._records_by_token.pop(record.token, None)
        try:
            if record.file_path.is_file():
                record.file_path.unlink()
        except OSError as exc:
            logging.warning("Could not delete expired YouTube download %s: %s", record.file_path, exc)
        await self._execute(
            "DELETE FROM youtube_downloads WHERE cache_key = ?",
            (record.cache_key,),
        )
        await self._execute(
            "DELETE FROM youtube_download_notifications WHERE token = ?",
            (record.token,),
        )

    async def cleanup_expired(self) -> None:
        now = time.time()
        async with self._db_lock:
            rows = self.connection.execute(
                "SELECT * FROM youtube_downloads WHERE expires_at <= ?",
                (now,),
            ).fetchall()
        for row in rows:
            await self._delete_record(self._record_from_row(row))

    async def _probe(self, path: Path) -> dict:
        output = await self.run_command(
            [
                self.ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height:format=duration,size,format_name",
                "-of",
                "json",
                str(path),
            ],
            300,
        )
        payload = json.loads(output)
        if not isinstance(payload, dict):
            raise RuntimeError("ffprobe returned invalid media metadata.")
        return payload

    async def download(
        self,
        request: YouTubeDownloadRequest,
        quality: str,
    ) -> tuple[YouTubeDownloadRecord, bool, float]:
        if quality not in request.available_qualities:
            raise ValueError("Выбранное качество недоступно для этого видео.")

        cached = await self.get_cached(request.video_id, quality)
        if cached is not None:
            return cached, True, 0.0

        cache_key = f"{request.video_id}:{quality}"
        key_lock = self._key_locks.setdefault(cache_key, asyncio.Lock())
        async with key_lock:
            cached = await self.get_cached(request.video_id, quality)
            if cached is not None:
                return cached, True, 0.0

            started_at = time.monotonic()
            temp_dir = Path(tempfile.mkdtemp(prefix=f"youtube_download_{request.video_id}_"))
            try:
                command = build_youtube_media_download_command(
                    self.base_command(),
                    request.url,
                    quality,
                    temp_dir,
                    self.config.concurrent_fragments,
                )
                async with self._download_semaphore:
                    try:
                        await self.run_command(command, self.config.timeout_seconds)
                    except RuntimeError as exc:
                        error_text = str(exc).lower()
                        can_retry_direct = self.fallback_base_command is not None and any(
                            marker in error_text
                            for marker in (
                                "http error 403",
                                "http error 429",
                                "sign in to confirm you’re not a bot",
                                "sign in to confirm you're not a bot",
                            )
                        )
                        if not can_retry_direct:
                            raise
                        logging.warning(
                            "Primary YouTube download route was rejected; retrying directly."
                        )
                        for partial in temp_dir.glob("download.*"):
                            try:
                                partial.unlink()
                            except OSError:
                                pass
                        fallback_command = build_youtube_media_download_command(
                            self.fallback_base_command(),
                            request.url,
                            quality,
                            temp_dir,
                            self.config.concurrent_fragments,
                        )
                        await self.run_command(
                            fallback_command,
                            self.config.timeout_seconds,
                        )

                candidates = sorted(
                    [
                        path
                        for path in temp_dir.glob("download.*")
                        if path.is_file() and path.suffix.lower() not in {".part", ".ytdl"}
                    ],
                    key=lambda path: path.stat().st_size,
                    reverse=True,
                )
                if not candidates:
                    raise RuntimeError("YouTube download did not create a media file.")
                downloaded_path = candidates[0]
                probe = await self._probe(downloaded_path)
                streams = [item for item in probe.get("streams", []) if isinstance(item, dict)]
                video_streams = [item for item in streams if item.get("codec_type") == "video"]
                audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
                if not audio_streams:
                    raise RuntimeError("Downloaded file has no audio stream.")

                width = 0
                height = 0
                extension = ".mp3" if quality == QUALITY_AUDIO else ".mp4"
                if quality == QUALITY_AUDIO:
                    if video_streams:
                        raise RuntimeError("Audio-only download unexpectedly contains video.")
                else:
                    if not video_streams:
                        raise RuntimeError("Downloaded file has no video stream.")
                    width = int(video_streams[0].get("width") or 0)
                    height = int(video_streams[0].get("height") or 0)
                    if height != int(quality):
                        raise RuntimeError(
                            f"YouTube returned {height}p instead of requested {quality}p."
                        )

                token = uuid.uuid4().hex
                suffix_label = "audio" if quality == QUALITY_AUDIO else f"{quality}p"
                file_name = f"{safe_download_title(request.title)} [{suffix_label}]{extension}"
                destination_dir = self.config.storage_dir / request.video_id / quality
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{token}{extension}"
                shutil.move(str(downloaded_path), destination)

                now = time.time()
                format_payload = probe.get("format") if isinstance(probe.get("format"), dict) else {}
                duration = float(format_payload.get("duration") or request.duration)
                record = YouTubeDownloadRecord(
                    cache_key=cache_key,
                    token=token,
                    video_id=request.video_id,
                    quality=quality,
                    title=request.title,
                    file_name=file_name,
                    file_path=destination,
                    file_size=destination.stat().st_size,
                    width=width,
                    height=height,
                    duration=duration,
                    created_at=now,
                    expires_at=now + self.config.ttl_seconds,
                )
                await self._execute(
                    """
                    INSERT OR REPLACE INTO youtube_downloads(
                        cache_key, token, video_id, quality, title, file_name,
                        file_path, file_size, width, height, duration, created_at, expires_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.cache_key,
                        record.token,
                        record.video_id,
                        record.quality,
                        record.title,
                        record.file_name,
                        str(record.file_path),
                        record.file_size,
                        record.width,
                        record.height,
                        record.duration,
                        record.created_at,
                        record.expires_at,
                    ),
                )
                self._records_by_cache_key[record.cache_key] = record
                self._records_by_token[record.token] = record
                return record, False, time.monotonic() - started_at
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


def download_landing_page(record: YouTubeDownloadRecord, file_url: str) -> web.Response:
    remaining = max(0, int(record.expires_at - time.time()))
    hours, remainder = divmod(remaining, 3600)
    minutes = remainder // 60
    quality = "Только аудио · MP3" if record.quality == QUALITY_AUDIO else f"{record.height}p · MP4"
    body = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Скачать {escape(record.title)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f4f7fb;color:#172033;font-family:Inter,Arial,sans-serif}}
.wrap{{max-width:720px;margin:0 auto;padding:32px 18px}}
.card{{background:#fff;border-radius:24px;padding:28px;box-shadow:0 16px 50px rgba(31,52,89,.12)}}
.ok{{display:inline-block;background:#e6f8ed;color:#168447;padding:8px 12px;border-radius:999px;font-weight:700}}
h1{{font-size:25px;line-height:1.25;margin:18px 0 10px;overflow-wrap:anywhere}} .meta{{color:#667085;line-height:1.8}}
.button{{display:flex;align-items:center;justify-content:center;min-height:52px;margin-top:24px;padding:14px 20px;border-radius:15px;background:#2481cc;color:#fff;text-align:center;text-decoration:none;font-size:18px;font-weight:800;touch-action:manipulation;-webkit-tap-highlight-color:transparent}}
.note{{margin-top:18px;color:#7a8496;font-size:14px;line-height:1.5}}
@media(max-width:480px){{.wrap{{padding:16px 12px}}.card{{padding:20px 16px;border-radius:18px}}h1{{font-size:21px}}.button{{font-size:17px}}}}
</style></head><body><main class="wrap"><section class="card">
<span class="ok">✓ Файл готов</span><h1>{escape(record.title)}</h1>
<div class="meta">Качество: <b>{escape(quality)}</b><br>Размер: <b>{escape(format_file_size(record.file_size))}</b><br>Длительность: <b>{escape(format_duration(record.duration))}</b></div>
<a class="button" href="{escape(file_url)}">⬇️ Скачать готовый файл</a>
<div class="note">Ссылка действует ещё примерно {hours} ч {minutes} мин. Открытие и повторная докачка продлевают срок. Если интернет оборвался, снова открой эту страницу и продолжи загрузку — сервер поддерживает докачку по частям.</div>
</section></main></body></html>"""
    return web.Response(
        text=body,
        content_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


YOUTUBE_DOWNLOAD_SERVICE_APP_KEY = web.AppKey(
    "youtube_download_service",
    YouTubeDownloadService,
)


def register_youtube_download_routes(app: web.Application, service: YouTubeDownloadService) -> None:
    app[YOUTUBE_DOWNLOAD_SERVICE_APP_KEY] = service

    async def landing(request: web.Request) -> web.Response:
        record = await service.get_by_token(request.match_info["token"])
        if record is None:
            raise web.HTTPGone(text="Ссылка истекла или файл уже удалён.")
        return download_landing_page(record, service.file_url(record))

    async def file_download(request: web.Request) -> web.FileResponse:
        record = await service.get_by_token(request.match_info["token"])
        if record is None:
            raise web.HTTPGone(text="Ссылка истекла или файл уже удалён.")
        encoded_name = quote(record.file_name)
        ascii_extension = ".mp3" if record.quality == QUALITY_AUDIO else ".mp4"
        ascii_name = f"youtube_{record.video_id}_{record.quality}{ascii_extension}"
        return web.FileResponse(
            record.file_path,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded_name}'
                ),
                "Cache-Control": "private, max-age=3600",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.router.add_get("/youtube-download/{token}", landing)
    app.router.add_get("/youtube-download/{token}/file", file_download)

    async def startup(_: web.Application) -> None:
        await service.start()

    async def cleanup(_: web.Application) -> None:
        await service.stop()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
