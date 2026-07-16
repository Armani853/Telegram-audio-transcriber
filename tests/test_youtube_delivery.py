from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import test_whisper as bot_module
from youtube_downloader import (
    YouTubeDownloadConfig,
    YouTubeDownloadNotification,
    YouTubeDownloadRecord,
    YouTubeDownloadRequest,
    YouTubeDownloadService,
)


class FakeBot:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.messages = []

    async def send_message(self, **kwargs):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("Telegram temporarily offline")
        self.messages.append(kwargs)


class YouTubeDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        root = Path(self.temp_dir.name)
        media_path = root / "storage" / "video" / "360" / "token.mp4"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(b"ready-video")

        async def forbidden_run(command, timeout_seconds=None):
            raise AssertionError(f"Unexpected command: {command}")

        self.service = YouTubeDownloadService(
            config=YouTubeDownloadConfig(
                public_base_url="https://files.example.com",
                storage_dir=root / "storage",
                db_path=root / "downloads.sqlite3",
                ttl_seconds=3600,
                request_ttl_seconds=1800,
                timeout_seconds=3600,
                telegram_direct_limit_bytes=0,
                concurrent_fragments=4,
                max_concurrent_downloads=1,
            ),
            run_command=forbidden_run,
            base_command=lambda: ["yt-dlp"],
            ffprobe_binary="ffprobe",
        )
        now = time.time()
        self.record = YouTubeDownloadRecord(
            cache_key="video:360",
            token="delivery-token",
            video_id="video",
            quality="360",
            title="Video",
            file_name="Video [360p].mp4",
            file_path=media_path,
            file_size=media_path.stat().st_size,
            width=640,
            height=360,
            duration=10,
            created_at=now,
            expires_at=now + 3600,
        )
        await self.service._execute(
            """
            INSERT INTO youtube_downloads(
                cache_key, token, video_id, quality, title, file_name,
                file_path, file_size, width, height, duration, created_at, expires_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.record.cache_key,
                self.record.token,
                self.record.video_id,
                self.record.quality,
                self.record.title,
                self.record.file_name,
                str(self.record.file_path),
                self.record.file_size,
                self.record.width,
                self.record.height,
                self.record.duration,
                self.record.created_at,
                self.record.expires_at,
            ),
        )

    async def asyncTearDown(self):
        await self.service.stop()
        self.temp_dir.cleanup()

    async def test_telegram_failure_is_rescheduled_then_delivered(self):
        notification = await self.service.queue_notification(
            42,
            self.record.token,
            "✅ <b>Видео готово</b>",
        )
        telegram = FakeBot(failures=1)

        first_result = await bot_module.deliver_one_youtube_notification(
            telegram,
            self.service,
            notification,
        )
        self.assertFalse(first_result)
        self.assertEqual(telegram.messages, [])

        await self.service._execute(
            """
            UPDATE youtube_download_notifications
            SET next_attempt_at = ?
            WHERE notification_id = ?
            """,
            (time.time() - 1, notification.notification_id),
        )
        delivered = await bot_module.deliver_pending_youtube_notifications(
            telegram,
            self.service,
        )

        self.assertEqual(delivered, 1)
        self.assertEqual(len(telegram.messages), 1)
        self.assertEqual(telegram.messages[0]["chat_id"], 42)
        self.assertEqual(
            telegram.messages[0]["reply_markup"].inline_keyboard[0][0].url,
            "https://files.example.com/youtube-download/delivery-token",
        )
        self.assertEqual(await self.service.pending_notifications(), [])

    async def test_server_finishes_when_client_status_updates_are_offline(self):
        class OfflineStatusMessage:
            async def edit_text(self, text, **kwargs):
                raise ConnectionError("client connection disappeared")

        class PreparedService:
            def __init__(self, real_service, record):
                self.config = real_service.config
                self.real_service = real_service
                self.record = record
                self.download_finished = False

            async def download(self, request, quality):
                self.download_finished = True
                return self.record, False, 1.25

            def landing_url(self, record):
                return self.real_service.landing_url(record)

            async def queue_notification(self, chat_id, token, text):
                return YouTubeDownloadNotification(
                    notification_id=77,
                    chat_id=chat_id,
                    token=token,
                    text=text,
                    attempts=0,
                    next_attempt_at=time.time(),
                )

            async def get_by_token(self, token):
                return self.record

            async def mark_notification_delivered(self, notification_id):
                return None

            async def reschedule_notification(self, notification):
                raise AssertionError("Telegram delivery should succeed in this test")

        request = YouTubeDownloadRequest(
            request_id="request",
            chat_id=42,
            url="https://youtu.be/video",
            video_id="video",
            title="Video",
            duration=10,
            available_qualities=("360",),
            estimated_sizes={},
            created_at=time.time(),
        )
        prepared_service = PreparedService(self.service, self.record)
        telegram = FakeBot()
        bot_module.youtube_download_active_chats.add(42)

        with patch.object(bot_module, "youtube_download_service", prepared_service):
            await bot_module.process_youtube_download_selection(
                telegram,
                42,
                request,
                "360",
                OfflineStatusMessage(),
            )

        self.assertTrue(prepared_service.download_finished)
        self.assertEqual(len(telegram.messages), 1)
        self.assertNotIn(42, bot_module.youtube_download_active_chats)

    async def test_small_video_falls_back_to_telegram_document(self):
        class Telegram:
            def __init__(self):
                self.video_attempts = 0
                self.documents = []

            async def send_video(self, **kwargs):
                self.video_attempts += 1
                raise RuntimeError("codec is not accepted as Telegram video")

            async def send_document(self, **kwargs):
                self.documents.append(kwargs)

        telegram = Telegram()
        delivered = await bot_module.send_small_youtube_file_through_telegram(
            telegram,
            42,
            self.record,
        )

        self.assertTrue(delivered)
        self.assertEqual(telegram.video_attempts, 1)
        self.assertEqual(len(telegram.documents), 1)
        self.assertEqual(telegram.documents[0]["chat_id"], 42)
        self.assertIn("максимальной совместимости", telegram.documents[0]["caption"])

    async def test_private_link_is_clearly_labeled_as_home_network_only(self):
        self.service.config = replace(
            self.service.config,
            public_base_url="http://192.168.1.72:8080",
        )

        text = bot_module.render_youtube_download_ready_text(
            self.service,
            self.record,
            preparation_seconds=1,
            was_cached=True,
        )
        keyboard = bot_module.youtube_download_result_keyboard(self.service, self.record)

        self.assertIn("только в домашней сети", text)
        self.assertEqual(
            keyboard.inline_keyboard[0][0].text,
            "🏠 Скачать в домашней сети",
        )

    def test_public_https_detection_rejects_private_and_local_addresses(self):
        self.assertFalse(bot_module.is_public_https_url("http://192.168.1.72:8080"))
        self.assertFalse(bot_module.is_public_https_url("https://127.0.0.1:8080"))
        self.assertFalse(bot_module.is_public_https_url("https://localhost:8080"))
        self.assertTrue(bot_module.is_public_https_url("https://files.example.com"))
