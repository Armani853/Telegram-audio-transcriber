import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import test_whisper as bot
from youtube_downloader import YouTubeDownloadConfig, YouTubeDownloadService
from tests.test_youtube_downloader import sample_metadata


class ConcurrentYouTubeLoadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bot.youtube_metadata_cache.clear()
        bot.youtube_metadata_tasks.clear()
        bot.youtube_transcript_cache.clear()
        bot.youtube_transcript_tasks.clear()

    async def asyncTearDown(self):
        bot.youtube_metadata_cache.clear()
        bot.youtube_metadata_tasks.clear()
        bot.youtube_transcript_cache.clear()
        bot.youtube_transcript_tasks.clear()

    async def test_100_users_share_one_metadata_request(self):
        calls = 0

        async def fake_metadata(url):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return {"id": "JTrjZNspkWA", "title": "Course", "duration": 34064}

        urls = [
            (
                "https://youtu.be/JTrjZNspkWA?si=phone"
                if index % 2
                else "https://www.youtube.com/watch?v=JTrjZNspkWA&t=9s"
            )
            for index in range(100)
        ]
        with patch.object(bot, "read_youtube_metadata", new=fake_metadata):
            results = await asyncio.gather(
                *(bot.read_youtube_metadata_cached(url) for url in urls)
            )

        self.assertEqual(calls, 1)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(result["id"] == "JTrjZNspkWA" for result in results))

    async def test_100_users_share_one_transcription_build(self):
        builds = 0

        async def fake_build(url):
            nonlocal builds
            builds += 1
            await asyncio.sleep(0.05)
            return bot.YouTubeTranscriptResult("Course", "0:00     │ Hola")

        urls = [
            f"https://youtu.be/JTrjZNspkWA?si=user{index}"
            for index in range(100)
        ]
        with patch.object(bot, "build_youtube_transcription_result", new=fake_build):
            results = await asyncio.gather(
                *(bot.get_youtube_transcription_result(url) for url in urls)
            )

        self.assertEqual(builds, 1)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(result.transcript == "0:00     │ Hola" for result, _ in results))
        self.assertEqual(sum(source == "new" for _, source in results), 1)
        self.assertEqual(sum(source == "shared" for _, source in results), 99)

    async def test_100_unique_transcriptions_are_bounded_not_launched_at_once(self):
        active_builds = 0
        maximum_active_builds = 0

        async def fake_metadata(url):
            video_id = bot.youtube_video_cache_key(url).split(":", 1)[-1]
            return {
                "id": video_id,
                "title": video_id,
                "duration": 60,
                "chapters": [{"start_time": 0, "title": "Opening"}],
            }

        async def fake_captions(url, destination_dir, metadata):
            nonlocal active_builds, maximum_active_builds
            active_builds += 1
            maximum_active_builds = max(maximum_active_builds, active_builds)
            await asyncio.sleep(0.01)
            active_builds -= 1
            return [{"start": 0, "end": 1, "text": metadata["title"]}]

        urls = [f"https://youtu.be/unique{index}" for index in range(100)]
        with (
            patch.object(bot, "read_youtube_metadata_cached", new=fake_metadata),
            patch.object(bot, "download_youtube_captions", new=fake_captions),
        ):
            results = await asyncio.gather(
                *(bot.get_youtube_transcription_result(url) for url in urls)
            )

        self.assertEqual(len(results), 100)
        self.assertLessEqual(
            maximum_active_builds,
            bot.YOUTUBE_TRANSCRIPT_MAX_CONCURRENT_BUILDS,
        )

    async def test_100_result_sends_respect_delivery_concurrency(self):
        active = 0
        maximum_active = 0
        delivered = []

        async def fake_send(_bot, chat_id, transcript, title):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            delivered.append((chat_id, transcript, title))
            active -= 1

        result = bot.YouTubeTranscriptResult("Course", "0:00     │ Hola")
        with patch.object(bot, "send_youtube_transcription_file", new=fake_send):
            await asyncio.gather(
                *(
                    bot.send_youtube_transcription_file_reliably(object(), chat_id, result)
                    for chat_id in range(1000, 1100)
                )
            )

        self.assertEqual(len(delivered), 100)
        self.assertLessEqual(maximum_active, bot.YOUTUBE_RESULT_MAX_CONCURRENT_SENDS)

    async def test_100_downloaders_share_one_physical_download(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_name:
            root = Path(temp_name)
            download_calls = 0

            async def fake_run(command, timeout_seconds=None):
                nonlocal download_calls
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
                            "format": {"duration": "120", "size": "11"},
                        }
                    )
                download_calls += 1
                await asyncio.sleep(0.05)
                output_template = Path(command[command.index("-o") + 1])
                output_path = Path(str(output_template).replace("%(ext)s", "mp4"))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"valid-media")
                return ""

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
                    max_concurrent_downloads=2,
                ),
                run_command=fake_run,
                base_command=lambda: ["yt-dlp"],
                ffprobe_binary="ffprobe",
            )
            try:
                requests = [
                    service.create_request(
                        2000 + index,
                        "https://youtu.be/video123",
                        sample_metadata(),
                    )
                    for index in range(100)
                ]
                started = time.monotonic()
                results = await asyncio.gather(
                    *(service.download(request, "720") for request in requests)
                )
                elapsed = time.monotonic() - started

                self.assertEqual(download_calls, 1)
                self.assertEqual(len({record.token for record, _, _ in results}), 1)
                self.assertLess(elapsed, 5)
            finally:
                await service.stop()
