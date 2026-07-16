import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import media_platform
import test_whisper as bot
from youtube_downloader import (
    YouTubeDownloadConfig,
    YouTubeDownloadRecord,
    YouTubeDownloadService,
    download_landing_page,
    register_youtube_download_routes,
)


DEVICE_USER_AGENTS = {
    "windows_chrome": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/148.0.0.0 Safari/537.36"
    ),
    "android_chrome": (
        "Mozilla/5.0 (Linux; Android 15; Pixel 9) "
        "AppleWebKit/537.36 Chrome/148.0.0.0 Mobile Safari/537.36"
    ),
    "iphone_safari": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X) "
        "AppleWebKit/605.1.15 Version/19.0 Mobile/15E148 Safari/604.1"
    ),
    "ipad_safari": (
        "Mozilla/5.0 (iPad; CPU OS 19_0 like Mac OS X) "
        "AppleWebKit/605.1.15 Version/19.0 Mobile/15E148 Safari/604.1"
    ),
}


class ResponsiveMarkupTests(unittest.TestCase):
    def test_download_page_has_mobile_layout_and_large_touch_target(self):
        now = time.time()
        record = YouTubeDownloadRecord(
            cache_key="video:360",
            token="token",
            video_id="video",
            quality="360",
            title="Очень длинное название видео " * 8,
            file_name="Видео [360p].mp4",
            file_path=Path("video.mp4"),
            file_size=1024,
            width=576,
            height=360,
            duration=120,
            created_at=now,
            expires_at=now + 3600,
        )

        html = download_landing_page(record, "/file").text
        self.assertIn('name="viewport"', html)
        self.assertIn("width=device-width", html)
        self.assertIn("@media(max-width:480px)", html)
        self.assertIn("min-height:52px", html)
        self.assertIn("overflow-wrap:anywhere", html)
        self.assertIn("touch-action:manipulation", html)

    def test_large_upload_page_is_mobile_safe_and_resumable(self):
        html = bot.render_upload_page("device-test").text

        self.assertIn('name="viewport"', html)
        self.assertIn("width: min(560px, calc(100vw - 28px))", html)
        self.assertIn("font-size: 16px", html)
        self.assertIn("min-height: 46px", html)
        self.assertIn("file.slice(start, end)", html)
        self.assertIn("attempt <= 3", html)

    def test_media_library_page_uses_responsive_grid_and_mobile_inputs(self):
        html = media_platform.html_page("Test", "<main>ok</main>").text

        self.assertIn('name="viewport"', html)
        self.assertIn("repeat(auto-fit, minmax(260px, 1fr))", html)
        self.assertIn("input, button { font-size: 16px; }", html)
        self.assertIn("min-height: 44px", html)


class CrossDeviceDownloadHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        root = Path(self.temp_dir.name)
        media_path = root / "storage" / "video" / "360" / "token.mp4"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(bytes(range(256)) * 32)

        async def forbidden_run(command, timeout_seconds=None):
            raise AssertionError(f"Unexpected command: {command}")

        self.service = YouTubeDownloadService(
            config=YouTubeDownloadConfig(
                public_base_url="http://127.0.0.1",
                storage_dir=root / "storage",
                db_path=root / "downloads.sqlite3",
                ttl_seconds=3600,
                request_ttl_seconds=1800,
                timeout_seconds=3600,
                telegram_direct_limit_bytes=49 * 1024 * 1024,
                concurrent_fragments=8,
                max_concurrent_downloads=2,
            ),
            run_command=forbidden_run,
            base_command=lambda: ["yt-dlp"],
            ffprobe_binary="ffprobe",
        )
        now = time.time()
        self.record = YouTubeDownloadRecord(
            cache_key="video:360",
            token="device-token",
            video_id="video123",
            quality="360",
            title="Международный курс",
            file_name="Международный курс [360p].mp4",
            file_path=media_path,
            file_size=media_path.stat().st_size,
            width=576,
            height=360,
            duration=120,
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
        app = web.Application()
        register_youtube_download_routes(app, self.service)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.temp_dir.cleanup()

    async def test_landing_page_is_identical_and_safe_for_all_devices(self):
        for device, user_agent in DEVICE_USER_AGENTS.items():
            with self.subTest(device=device):
                response = await self.client.get(
                    f"/youtube-download/{self.record.token}",
                    headers={"User-Agent": user_agent},
                )
                body = await response.text()
                self.assertEqual(response.status, 200)
                self.assertIn("Скачать готовый файл", body)
                self.assertIn('name="viewport"', body)
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
                self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    async def test_head_and_range_downloads_work_for_all_devices(self):
        for device, user_agent in DEVICE_USER_AGENTS.items():
            with self.subTest(device=device):
                headers = {"User-Agent": user_agent}
                head = await self.client.head(
                    f"/youtube-download/{self.record.token}/file",
                    headers=headers,
                )
                self.assertEqual(head.status, 200)
                self.assertEqual(int(head.headers["Content-Length"]), self.record.file_size)
                self.assertEqual(head.headers["Accept-Ranges"], "bytes")
                self.assertEqual(head.headers["Content-Type"], "video/mp4")
                disposition = head.headers["Content-Disposition"]
                self.assertIn('filename="youtube_video123_360.mp4"', disposition)
                self.assertIn("filename*=UTF-8''", disposition)

                partial = await self.client.get(
                    f"/youtube-download/{self.record.token}/file",
                    headers={**headers, "Range": "bytes=1024-2047"},
                )
                data = await partial.read()
                self.assertEqual(partial.status, 206)
                self.assertEqual(len(data), 1024)
                self.assertEqual(
                    partial.headers["Content-Range"],
                    f"bytes 1024-2047/{self.record.file_size}",
                )
                self.assertEqual(data, self.record.file_path.read_bytes()[1024:2048])

    async def test_parallel_mobile_and_desktop_ranges_do_not_interfere(self):
        async def fetch(device: str, start: int):
            response = await self.client.get(
                f"/youtube-download/{self.record.token}/file",
                headers={
                    "User-Agent": DEVICE_USER_AGENTS[device],
                    "Range": f"bytes={start}-{start + 511}",
                },
            )
            return response.status, await response.read()

        results = await asyncio.gather(
            fetch("windows_chrome", 0),
            fetch("android_chrome", 512),
            fetch("iphone_safari", 1024),
            fetch("ipad_safari", 1536),
        )

        self.assertEqual([status for status, _ in results], [206, 206, 206, 206])
        self.assertTrue(all(len(data) == 512 for _, data in results))


class CrossDeviceChunkUploadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = web.Application(client_max_size=4 * 1024 * 1024)
        app.router.add_post("/upload/{upload_id}/init", bot.upload_init_handler)
        app.router.add_put("/upload/{upload_id}/chunk/{index}", bot.upload_chunk_handler)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.session_ids: list[str] = []

    async def asyncTearDown(self):
        await self.client.close()
        for upload_id in self.session_ids:
            session = bot.upload_sessions.pop(upload_id, None)
            if session is not None:
                bot.cleanup_upload_chunk_dir(session)

    async def test_interrupted_chunk_upload_resumes_on_every_device_profile(self):
        with patch.object(bot, "UPLOAD_CHUNK_BYTES", 1024):
            for device, user_agent in DEVICE_USER_AGENTS.items():
                with self.subTest(device=device):
                    upload_id = bot.create_upload_session(42)
                    self.session_ids.append(upload_id)
                    headers = {"User-Agent": user_agent}
                    payload = {"file_name": f"{device}.mp3", "file_size": 2048}

                    initialized = await self.client.post(
                        f"/upload/{upload_id}/init",
                        json=payload,
                        headers=headers,
                    )
                    init_data = await initialized.json()
                    self.assertEqual(initialized.status, 200)
                    self.assertEqual(init_data["chunk_size"], 1024)
                    self.assertEqual(init_data["total_chunks"], 2)
                    self.assertEqual(init_data["received_chunks"], [])

                    chunk = bytes(range(256)) * 4
                    uploaded = await self.client.put(
                        f"/upload/{upload_id}/chunk/0",
                        data=chunk,
                        headers={**headers, "Content-Type": "application/octet-stream"},
                    )
                    upload_data = await uploaded.json()
                    self.assertEqual(uploaded.status, 200)
                    self.assertEqual(upload_data["received_chunks"], [0])

                    resumed = await self.client.post(
                        f"/upload/{upload_id}/init",
                        json=payload,
                        headers=headers,
                    )
                    resumed_data = await resumed.json()
                    self.assertEqual(resumed.status, 200)
                    self.assertEqual(resumed_data["received_chunks"], [0])
