import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_whisper as bot
from youtube_downloader import QUALITY_AUDIO, YouTubeDownloadRequest


class FakeMessage:
    def __init__(self, chat_id=42, text=""):
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.answers = []

    async def answer(self, text, **kwargs):
        response = FakeMessage(self.chat.id)
        self.answers.append({"text": text, "kwargs": kwargs, "message": response})
        return response


class FakeCallback:
    def __init__(self, data, message):
        self.data = data
        self.message = message
        self.answers = []

    async def answer(self, text=None, **kwargs):
        self.answers.append({"text": text, "kwargs": kwargs})


class BotControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        bot.youtube_download_waiting_chats.discard(42)
        bot.youtube_download_active_chats.discard(42)
        bot.chat_transcription_modes.pop(42, None)
        for upload_id, session in list(bot.upload_sessions.items()):
            if session.chat_id == 42:
                bot.cleanup_upload_chunk_dir(session)
                bot.upload_sessions.pop(upload_id, None)

    async def test_main_menu_and_command_buttons_all_respond(self):
        start_message = FakeMessage(text="/start")
        await bot.start_handler(start_message)
        self.assertEqual(len(start_message.answers), 1)
        main_keyboard = start_message.answers[0]["kwargs"]["reply_markup"]
        buttons = [button for row in main_keyboard.inline_keyboard for button in row]
        self.assertEqual(
            [button.text for button in buttons],
            [
                "🎬 Расшифровать YouTube",
                "📥 Скачать видео с YouTube",
                "📚 Медиатека",
                "🌐 Русский",
                "📤 Загрузить большой файл",
            ],
        )
        self.assertTrue(next(button for button in buttons if button.text == "📚 Медиатека").url)
        self.assertTrue(
            next(
                button
                for button in buttons
                if button.text == "📤 Загрузить большой файл"
            ).url
        )

        youtube_message = FakeMessage()
        youtube_callback = FakeCallback("youtube:help", youtube_message)
        await bot.youtube_help_callback(youtube_callback)
        self.assertIn("Расшифровка YouTube", youtube_message.answers[0]["text"])
        self.assertIn("отправь ссылку", youtube_callback.answers[0]["text"].lower())

        download_message = FakeMessage()
        download_callback = FakeCallback("youtube_download:help", download_message)
        await bot.youtube_download_help_callback(download_callback)
        self.assertIn(42, bot.youtube_download_waiting_chats)
        self.assertIn("Скачать видео с YouTube", download_message.answers[0]["text"])

        language_message = FakeMessage()
        language_callback = FakeCallback("transcription_mode:menu", language_message)
        await bot.transcription_mode_callback(language_callback)
        language_keyboard = language_message.answers[0]["kwargs"]["reply_markup"]
        language_buttons = [
            button for row in language_keyboard.inline_keyboard for button in row
        ]
        self.assertEqual([len(row) for row in language_keyboard.inline_keyboard], [2, 2, 1])
        self.assertEqual(
            [button.callback_data for button in language_buttons],
            [
                "transcription_mode:ru",
                "transcription_mode:en",
                "transcription_mode:es",
                "transcription_mode:hy",
                "transcription_mode:auto",
            ],
        )
        for mode in ("ru", "en", "es", "hy", "auto"):
            callback = FakeCallback(f"transcription_mode:{mode}", language_message)
            await bot.transcription_mode_callback(callback)
            self.assertEqual(bot.get_chat_transcription_mode(42), mode)
            self.assertTrue(callback.answers)

        help_message = FakeMessage(text="/help")
        await bot.help_handler(help_message)
        self.assertIn("Что я умею", help_message.answers[0]["text"])
        self.assertIn("reply_markup", help_message.answers[0]["kwargs"])

        long_message = FakeMessage(text="/long")
        await bot.long_upload_handler(long_message)
        long_button = long_message.answers[0]["kwargs"]["reply_markup"].inline_keyboard[0][0]
        self.assertEqual(long_button.text, "📤 Загрузить большой файл")
        self.assertIn("/upload/", long_button.url)
        self.assertIn("Загрузка идёт частями", long_message.answers[0]["text"])
        self.assertNotIn("домашн", long_message.answers[0]["text"].lower())

        library_message = FakeMessage(text="/library")
        await bot.library_handler(library_message)
        library_button = library_message.answers[0]["kwargs"]["reply_markup"].inline_keyboard[0][0]
        self.assertEqual(library_button.text, "Открыть медиатеку")
        self.assertIn("/app?chat_id=42", library_button.url)

        search_message = FakeMessage(text="/search")
        await bot.search_handler(search_message)
        self.assertIn("Напиши запрос", search_message.answers[0]["text"])

        command_message = FakeMessage(text="/download")
        await bot.youtube_download_command_handler(command_message)
        self.assertIn(42, bot.youtube_download_waiting_chats)
        self.assertIn("YouTube-ссылку", command_message.answers[0]["text"])

        fallback_message = FakeMessage(text="неизвестное действие")
        await bot.fallback_handler(fallback_message)
        self.assertIn("голосовое", fallback_message.answers[0]["text"])

    async def test_every_youtube_quality_button_starts_the_selected_job(self):
        request = YouTubeDownloadRequest(
            request_id="request",
            chat_id=42,
            url="https://youtu.be/JTrjZNspkWA",
            video_id="JTrjZNspkWA",
            title="FULL SPANISH COURSE FREE FROM BEGINNERS TO ADVANCED",
            duration=34064,
            available_qualities=("360", "720", "1080", QUALITY_AUDIO),
            estimated_sizes={},
            created_at=0,
        )
        service = SimpleNamespace(get_request=lambda request_id, chat_id: request)
        keyboard = bot.build_youtube_download_quality_keyboard(request)
        callbacks = [button.callback_data for row in keyboard.inline_keyboard for button in row]
        self.assertEqual(
            callbacks,
            [
                "youtube_download:select:request:360",
                "youtube_download:select:request:720",
                "youtube_download:select:request:1080",
                "youtube_download:select:request:audio",
            ],
        )

        original_create_task = asyncio.create_task
        for quality in ("360", "720", "1080", QUALITY_AUDIO):
            message = FakeMessage()
            callback = FakeCallback(
                f"youtube_download:select:request:{quality}",
                message,
            )
            created_tasks = []

            def capture_task(coroutine):
                task = original_create_task(coroutine)
                created_tasks.append(task)
                return task

            process = AsyncMock()
            with (
                patch.object(bot, "youtube_download_service", service),
                patch.object(bot, "process_youtube_download_selection", process),
                patch.object(bot.asyncio, "create_task", side_effect=capture_task),
            ):
                await bot.youtube_download_quality_callback(callback, SimpleNamespace())
                await asyncio.gather(*created_tasks)

            self.assertEqual(callback.answers[0]["text"], "Начинаю скачивание")
            process.assert_awaited_once()
            self.assertEqual(process.await_args.args[3], quality)
            bot.youtube_download_active_chats.discard(42)
