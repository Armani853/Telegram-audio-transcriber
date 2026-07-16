import json
from pathlib import Path
import tempfile
import time
import unittest

from youtube_downloader import (
    QUALITY_AUDIO,
    YouTubeDownloadConfig,
    YouTubeDownloadRecord,
    YouTubeDownloadService,
    build_youtube_media_download_command,
    estimated_quality_sizes,
    exact_video_qualities,
)


def sample_metadata() -> dict:
    return {
        "id": "video123",
        "title": "Course / lesson",
        "duration": 120,
        "formats": [
            {
                "format_id": "18",
                "height": 360,
                "ext": "mp4",
                "vcodec": "avc1",
                "acodec": "aac",
                "filesize": 10_000,
            },
            {
                "format_id": "136",
                "height": 720,
                "ext": "mp4",
                "vcodec": "avc1",
                "acodec": "none",
                "filesize": 20_000,
            },
            {
                "format_id": "137",
                "height": 1080,
                "ext": "mp4",
                "vcodec": "avc1",
                "acodec": "none",
                "filesize": 40_000,
            },
            {
                "format_id": "140",
                "height": None,
                "ext": "m4a",
                "vcodec": "none",
                "acodec": "aac",
                "filesize": 5_000,
            },
        ],
    }


class YouTubeDownloadFormattingTests(unittest.TestCase):
    def test_only_exact_supported_qualities_are_offered(self):
        metadata = sample_metadata()
        metadata["formats"] = [
            metadata["formats"][0],
            metadata["formats"][1],
            metadata["formats"][3],
        ]

        self.assertEqual(exact_video_qualities(metadata), ("360", "720", QUALITY_AUDIO))

    def test_estimated_size_combines_video_and_audio(self):
        estimates = estimated_quality_sizes(sample_metadata())

        self.assertEqual(estimates["720"], 25_000)
        self.assertEqual(estimates["1080"], 45_000)
        self.assertEqual(estimates[QUALITY_AUDIO], 5_000)

    def test_video_command_requires_the_selected_height(self):
        command = build_youtube_media_download_command(
            ["yt-dlp", "--proxy", "***"],
            "https://youtu.be/video123",
            "720",
            Path("downloads"),
            8,
        )

        format_selector = command[command.index("-f") + 1]
        self.assertIn("height=720", format_selector)
        self.assertNotIn("height<=720", format_selector)
        self.assertIn("--merge-output-format", command)
        self.assertIn("--concurrent-fragments", command)

    def test_audio_command_produces_universal_mp3(self):
        command = build_youtube_media_download_command(
            ["yt-dlp"],
            "https://youtu.be/video123",
            QUALITY_AUDIO,
            Path("downloads"),
            4,
        )

        self.assertIn("--audio-format", command)
        self.assertEqual(command[command.index("--audio-format") + 1], "mp3")


class YouTubeDownloadServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_is_validated_persisted_and_reused_from_cache(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_name:
            root = Path(temp_name)
            calls: list[list[str]] = []

            async def fake_run(command, timeout_seconds=None):
                calls.append(command)
                if command[0] == "ffprobe":
                    return json.dumps(
                        {
                            "streams": [
                                {
                                    "codec_type": "video",
                                    "codec_name": "h264",
                                    "width": 1280,
                                    "height": 720,
                                },
                                {"codec_type": "audio", "codec_name": "aac"},
                            ],
                            "format": {"duration": "120", "size": "10"},
                        }
                    )
                if "primary-proxy" in command:
                    raise RuntimeError("HTTP Error 429: Too Many Requests")
                output_template = Path(command[command.index("-o") + 1])
                output_path = Path(str(output_template).replace("%(ext)s", "mp4"))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"valid-media")
                return ""

            config = YouTubeDownloadConfig(
                public_base_url="https://files.example.com",
                storage_dir=root / "storage",
                db_path=root / "downloads.sqlite3",
                ttl_seconds=3600,
                request_ttl_seconds=1800,
                timeout_seconds=3600,
                telegram_direct_limit_bytes=49 * 1024 * 1024,
                concurrent_fragments=8,
                max_concurrent_downloads=2,
            )
            service = YouTubeDownloadService(
                config=config,
                run_command=fake_run,
                base_command=lambda: ["yt-dlp", "--proxy", "primary-proxy"],
                ffprobe_binary="ffprobe",
                fallback_base_command=lambda: ["yt-dlp", "--proxy", ""],
            )
            try:
                request = service.create_request(
                    42,
                    "https://youtu.be/video123",
                    sample_metadata(),
                )
                record, cached, _ = await service.download(request, "720")
                second_record, second_cached, second_seconds = await service.download(request, "720")

                self.assertFalse(cached)
                self.assertTrue(second_cached)
                self.assertEqual(second_seconds, 0)
                self.assertEqual(record.token, second_record.token)
                self.assertTrue(record.file_path.is_file())
                self.assertEqual(record.height, 720)
                self.assertEqual(record.file_name, "Course lesson [720p].mp4")
                self.assertEqual(service.landing_url(record), f"https://files.example.com/youtube-download/{record.token}")
                self.assertEqual(sum(1 for command in calls if command[0] == "yt-dlp"), 2)
            finally:
                await service.stop()

    async def test_opening_file_extends_sliding_expiry(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_name:
            root = Path(temp_name)
            media_path = root / "storage" / "video" / "360" / "token.mp4"
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(b"media")

            async def forbidden_run(command, timeout_seconds=None):
                raise AssertionError(f"Unexpected command: {command}")

            service = YouTubeDownloadService(
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
            old_expiry = time.time() + 30
            record = YouTubeDownloadRecord(
                cache_key="video:360",
                token="sliding-token",
                video_id="video",
                quality="360",
                title="Video",
                file_name="Video [360p].mp4",
                file_path=media_path,
                file_size=media_path.stat().st_size,
                width=640,
                height=360,
                duration=10,
                created_at=time.time(),
                expires_at=old_expiry,
            )
            await service._execute(
                """
                INSERT INTO youtube_downloads(
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
            try:
                touched = await service.get_by_token(record.token)

                self.assertIsNotNone(touched)
                self.assertGreater(touched.expires_at, old_expiry + 3500)
            finally:
                await service.stop()

    async def test_pending_result_notification_survives_service_restart(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_name:
            root = Path(temp_name)
            config = YouTubeDownloadConfig(
                public_base_url="https://files.example.com",
                storage_dir=root / "storage",
                db_path=root / "downloads.sqlite3",
                ttl_seconds=3600,
                request_ttl_seconds=1800,
                timeout_seconds=3600,
                telegram_direct_limit_bytes=0,
                concurrent_fragments=4,
                max_concurrent_downloads=1,
            )

            async def forbidden_run(command, timeout_seconds=None):
                raise AssertionError(f"Unexpected command: {command}")

            def make_service():
                return YouTubeDownloadService(
                    config=config,
                    run_command=forbidden_run,
                    base_command=lambda: ["yt-dlp"],
                    ffprobe_binary="ffprobe",
                )

            first_service = make_service()
            queued = await first_service.queue_notification(42, "token", "ready")
            await first_service.stop()

            second_service = make_service()
            try:
                await second_service._execute(
                    """
                    UPDATE youtube_download_notifications
                    SET next_attempt_at = ?
                    WHERE notification_id = ?
                    """,
                    (time.time() - 1, queued.notification_id),
                )
                pending = await second_service.pending_notifications()

                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0].notification_id, queued.notification_id)
                self.assertEqual(pending[0].chat_id, 42)
                self.assertEqual(pending[0].text, "ready")
            finally:
                await second_service.stop()
