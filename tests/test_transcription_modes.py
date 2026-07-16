import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import test_whisper as bot


RUSSIAN_INTERVIEW_EXCERPT = (
    "нет он нихуя мне не сказал он просто сказал спасибо за то что пришел на встречу "
    "спасибо что типа мы поговорили познакомились тебе фидбэк будет от HR"
)

ENGLISH_INTERVIEW_EXCERPT = (
    "No, he didn't say anything to me, he just said thank you for coming to the meeting, "
    "thank you for talking, getting to know each other, you will get feedback from HR."
)

ARMENIAN_TRANSLATION_EXCERPT = (
    "\u0549\u0567, \u0576\u0561 \u0578\u0579\u056b\u0576\u0579 \u056b\u0576\u0571 "
    "\u0579\u0561\u057d\u0561\u0581, \u0576\u0561 \u0578\u0582\u0572\u0572\u0561\u056f\u056b "
    "\u0561\u057d\u0561\u0581 \u0577\u0576\u0578\u0580\u0570\u0561\u056f\u0561\u056c\u0578\u0582\u0569\u0575\u0578\u0582\u0576, "
    "\u0578\u0580 \u0565\u056f\u0561\u0580 \u0570\u0561\u0576\u0564\u056b\u057a\u0574\u0561\u0576\u0568"
)


def media_message(**overrides):
    defaults = {
        "voice": None,
        "video_note": None,
        "audio": None,
        "video": None,
        "document": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeTranscription:
    text = "recognized text"


class FakeTranscriptions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeTranscription()


class FakeGroqClient:
    def __init__(self):
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())
        self.closed = False

    async def close(self):
        self.closed = True


class FakeAudioFile:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, *_args, **_kwargs):
        return b"audio"


class FakeBot:
    def __init__(self):
        self.messages = []
        self.documents = []

    async def send_message(self, chat_id, text):
        self.messages.append({"chat_id": chat_id, "text": text})

    async def send_document(self, chat_id, document, caption):
        self.documents.append({"chat_id": chat_id, "document": document, "caption": caption})


class FakeStatusMessage:
    def __init__(self):
        self.edits = []

    async def edit_text(self, text):
        self.edits.append(text)


class ModeNormalizationTests(unittest.TestCase):
    def test_supported_modes_normalize_to_expected_codes(self):
        cases = {
            "ru": "ru",
            "en": "en",
            "eng": "en",
            "english": "en",
            "es": "es",
            "esp": "es",
            "spanish": "es",
            "espanol": "es",
            "auto": "auto",
            "detect": "auto",
            "autodetect": "auto",
            "hy": "hy",
            "arm": "hy",
            "armenian": "hy",
            "hay": "hy",
            "hayeren": "hy",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(bot.normalize_transcription_mode(raw), expected)

    def test_unknown_mode_falls_back_to_russian(self):
        self.assertEqual(bot.normalize_transcription_mode("klingon"), "ru")
        self.assertEqual(bot.normalize_transcription_mode(""), "ru")

    def test_labels_are_user_facing(self):
        self.assertEqual(bot.transcription_mode_label("ru"), "Русский")
        self.assertEqual(bot.transcription_mode_label("en"), "English")
        self.assertEqual(bot.transcription_mode_label("es"), "Español")
        self.assertEqual(bot.transcription_mode_label("hy"), "Armenian")
        self.assertEqual(bot.transcription_mode_label("auto"), "Auto")

    def test_keyboard_contains_all_modes_and_callbacks(self):
        bot.chat_transcription_modes[123] = "hy"
        try:
            keyboard = bot.build_transcription_mode_keyboard(123)
        finally:
            bot.chat_transcription_modes.pop(123, None)

        buttons = [button for row in keyboard.inline_keyboard for button in row]
        labels = [button.text for button in buttons]
        callbacks = [button.callback_data for button in buttons]
        self.assertEqual([len(row) for row in keyboard.inline_keyboard], [2, 2, 1])
        self.assertEqual(labels, ["Русский", "English", "Español", "✓ Armenian", "Auto"])
        self.assertEqual(
            callbacks,
            [
                "transcription_mode:ru",
                "transcription_mode:en",
                "transcription_mode:es",
                "transcription_mode:hy",
                "transcription_mode:auto",
            ],
        )

    def test_chat_mode_defaults_to_russian(self):
        bot.chat_transcription_modes.pop(999, None)
        self.assertEqual(bot.get_chat_transcription_mode(999), "ru")

    def test_main_keyboard_has_dedicated_youtube_button(self):
        keyboard = bot.build_main_keyboard(999)
        buttons = [button for row in keyboard.inline_keyboard for button in row]
        youtube_buttons = [button for button in buttons if button.callback_data == "youtube:help"]
        self.assertEqual(len(youtube_buttons), 1)
        self.assertIn("YouTube", youtube_buttons[0].text)

        download_buttons = [
            button
            for button in buttons
            if button.callback_data == "youtube_download:help"
        ]
        self.assertEqual(len(download_buttons), 1)
        self.assertIn("Скачать", download_buttons[0].text)


class MediaRoutingTests(unittest.TestCase):
    def test_voice_and_audio_are_supported(self):
        self.assertTrue(bot.is_supported_audio_message(media_message(voice=object())))
        self.assertTrue(bot.is_supported_audio_message(media_message(audio=object())))

    def test_video_note_and_video_are_supported(self):
        self.assertTrue(bot.is_supported_audio_message(media_message(video_note=object())))
        self.assertTrue(bot.is_supported_audio_message(media_message(video=object())))

    def test_document_audio_video_mime_is_supported(self):
        audio_doc = SimpleNamespace(mime_type="audio/mpeg", file_name="blob")
        video_doc = SimpleNamespace(mime_type="video/mp4", file_name="blob")
        self.assertTrue(bot.is_supported_audio_message(media_message(document=audio_doc)))
        self.assertTrue(bot.is_supported_audio_message(media_message(document=video_doc)))

    def test_document_media_suffix_is_supported_without_mime(self):
        document = SimpleNamespace(mime_type="", file_name="meeting.mkv")
        self.assertTrue(bot.is_supported_audio_message(media_message(document=document)))

    def test_plain_text_document_is_not_supported(self):
        document = SimpleNamespace(mime_type="text/plain", file_name="notes.txt")
        self.assertFalse(bot.is_supported_audio_message(media_message(document=document)))

    def test_voice_ogg_is_normalized_with_ffmpeg(self):
        message = media_message(voice=object())
        self.assertTrue(bot.should_prepare_media_with_ffmpeg(message, "voice_123.ogg"))

    def test_common_audio_stays_direct_without_ffmpeg(self):
        message = media_message(audio=object())
        for name in ["song.mp3", "speech.wav", "note.m4a", "clip.webm", "voice.opus"]:
            with self.subTest(name=name):
                self.assertFalse(bot.should_prepare_media_with_ffmpeg(message, name))

    def test_video_and_video_note_use_ffmpeg(self):
        self.assertTrue(bot.should_prepare_media_with_ffmpeg(media_message(video=object()), "clip.mp4"))
        self.assertTrue(bot.should_prepare_media_with_ffmpeg(media_message(video_note=object()), "round.mp4"))

    def test_webm_video_uses_ffmpeg_but_webm_audio_stays_direct(self):
        self.assertTrue(bot.should_prepare_media_with_ffmpeg(media_message(video=object()), "clip.webm"))
        self.assertFalse(bot.should_prepare_media_with_ffmpeg(media_message(audio=object()), "clip.webm"))

    def test_exotic_audio_uses_ffmpeg(self):
        self.assertTrue(bot.should_prepare_media_with_ffmpeg(media_message(audio=object()), "old.wma"))


class YouTubeUrlTests(unittest.TestCase):
    def test_extracts_supported_youtube_urls(self):
        cases = [
            "https://youtu.be/abc123",
            "watch https://www.youtube.com/watch?v=abc123&t=10",
            "https://youtube.com/shorts/abc123",
            "https://m.youtube.com/live/abc123?feature=share",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(bot.extract_youtube_url(text))

    def test_ignores_non_youtube_urls(self):
        self.assertEqual(bot.extract_youtube_url("https://example.com/watch?v=abc"), "")
        self.assertFalse(bot.should_handle_youtube_text("hello world"))

    def test_does_not_handle_commands_even_with_youtube_url(self):
        self.assertFalse(bot.should_handle_youtube_text("/search https://youtu.be/abc123"))

    def test_trims_trailing_punctuation(self):
        self.assertEqual(bot.extract_youtube_url("смотри https://youtu.be/abc123."), "https://youtu.be/abc123")

    def test_youtube_commands_are_built_as_expected(self):
        url = "https://youtu.be/abc123"
        destination = Path("tmp")
        self.assertEqual(
            bot.build_youtube_metadata_command(url),
            [
                *bot.youtube_ytdlp_base_command(),
                "--dump-json",
                "--no-playlist",
                "--skip-download",
                url,
            ],
        )
        self.assertEqual(
            bot.build_youtube_download_command(url, destination),
            [
                *bot.youtube_ytdlp_base_command(),
                "--no-playlist",
                "-x",
                "--audio-format",
                bot.YOUTUBE_AUDIO_FORMAT,
                "-o",
                str(destination / "youtube_%(id)s.%(ext)s"),
                url,
            ],
        )

    def test_youtube_metadata_validation_accepts_single_video(self):
        bot.validate_youtube_metadata({"duration": 120, "title": "ok"})

    def test_youtube_metadata_validation_rejects_playlist(self):
        with self.assertRaisesRegex(ValueError, "Плейлисты"):
            bot.validate_youtube_metadata({"_type": "playlist", "duration": 120})

    def test_youtube_metadata_validation_rejects_missing_duration(self):
        with self.assertRaisesRegex(ValueError, "длительность"):
            bot.validate_youtube_metadata({"title": "missing"})

    def test_youtube_metadata_validation_rejects_too_long_video(self):
        with self.assertRaisesRegex(ValueError, "слишком длинное"):
            bot.validate_youtube_metadata({"duration": bot.YOUTUBE_MAX_DURATION_SECONDS + 1})

    def test_selects_original_automatic_caption_track(self):
        metadata = {
            "language": "en-US",
            "automatic_captions": {
                "ru": [{"name": "Russian"}],
                "en-orig": [{"name": "English (Original)"}],
            },
        }
        self.assertEqual(bot.select_youtube_caption_track(metadata), ("automatic", "en-orig"))

    def test_original_automatic_track_wins_over_exact_language_translation(self):
        metadata = {
            "language": "en",
            "automatic_captions": {
                "en": [{"name": "English"}],
                "en-orig": [{"name": "English (Original)"}],
            },
        }
        self.assertEqual(bot.select_youtube_caption_track(metadata), ("automatic", "en-orig"))

    def test_formats_youtube_transcript_vertically(self):
        transcript = bot.format_youtube_transcript(
            [
                {"start": 0, "end": 1, "text": "hello"},
                {"start": 61.9, "end": 63, "text": "world"},
                {"start": 3661, "end": 3662, "text": "long video"},
            ]
        )
        self.assertEqual(
            transcript,
            "0:00     │ hello\n1:01     │ world\n1:01:01  │ long video",
        )

    def test_groups_short_youtube_caption_fragments(self):
        transcript = bot.format_youtube_transcript(
            [
                {"start": 0, "end": 1, "text": "hola"},
                {"start": 1, "end": 5, "text": "hello guys how are you"},
                {"start": 5, "end": 8, "text": "this is the course"},
                {"start": 14, "end": 16, "text": "new phrase"},
            ]
        )
        self.assertEqual(
            transcript,
            "0:00     │ hola hello guys how are you this is the course\n"
            "0:14     │ new phrase",
        )

    def test_extracts_native_youtube_chapters(self):
        chapters = bot.extract_youtube_outline(
            {
                "duration": 7200,
                "chapters": [
                    {"start_time": 0, "title": "<Untitled Chapter 1>"},
                    {"start_time": 185, "title": "Introduce yourself"},
                    {"start_time": 820, "title": "Greetings"},
                ],
                "description": "0:00 This must not replace native chapters",
            }
        )

        self.assertEqual(
            chapters,
            [
                {"start": 185.0, "title": "Introduce yourself"},
                {"start": 820.0, "title": "Greetings"},
            ],
        )

    def test_extracts_youtube_chapters_from_description(self):
        chapters = bot.extract_youtube_outline(
            {
                "duration": 36000,
                "description": (
                    "Course links\n"
                    "0:03:05 Introduce yourself ¡Hola! Me llamo…,\n"
                    "0:13:40 - Greetings\n"
                    "9:14:05 Spanish level test advanced C1\n"
                ),
            }
        )

        self.assertEqual(chapters[0], {"start": 185.0, "title": "Introduce yourself ¡Hola! Me llamo…"})
        self.assertEqual(chapters[1], {"start": 820.0, "title": "Greetings"})
        self.assertEqual(chapters[2], {"start": 33245.0, "title": "Spanish level test advanced C1"})

    def test_formats_youtube_contents_and_transcript_in_one_document(self):
        document = bot.format_youtube_document(
            [{"start": 185, "title": "Introduce yourself"}],
            [{"start": 185, "end": 190, "text": "Hola, me llamo Ana."}],
        )

        self.assertEqual(
            document,
            "СОДЕРЖАНИЕ ВИДЕО\n\n"
            "3:05     │ Introduce yourself\n\n"
            "РАСШИФРОВКА ВИДЕО\n\n"
            "3:05     │ Hola, me llamo Ana.",
        )

    def test_parses_youtube_json3_captions(self):
        segments = bot.parse_youtube_json3_payload(
            {
                "events": [
                    {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello"}]},
                    {"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "\n"}]},
                    {"tStartMs": 1500, "dDurationMs": 1000, "segs": [{"utf8": "next line"}]},
                ]
            }
        )

        self.assertEqual([segment["text"] for segment in segments], ["hello", "next line"])
        self.assertEqual(segments[1]["start"], 1.5)


class YouTubePipelineTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_youtube_result_is_always_sent_as_txt_document(self):
        class CapturingBot(FakeBot):
            async def send_document(self, chat_id, document, caption):
                self.documents.append(
                    {
                        "chat_id": chat_id,
                        "filename": document.filename,
                        "text": Path(document.path).read_text(encoding="utf-8"),
                        "caption": caption,
                    }
                )

        fake_bot = CapturingBot()
        await bot.send_youtube_transcription_file(fake_bot, 4, "0:00\nhello", "Test / video")

        self.assertEqual(fake_bot.messages, [])
        self.assertEqual(len(fake_bot.documents), 1)
        self.assertTrue(fake_bot.documents[0]["filename"].endswith(".txt"))
        self.assertEqual(fake_bot.documents[0]["text"], "0:00\nhello")

    async def test_download_youtube_audio_uses_metadata_and_download_commands(self):
        calls = []
        fake_audio = MagicMock()
        fake_audio.name = "youtube_abc.mp3"
        fake_audio.suffix = ".mp3"
        fake_audio.is_file.return_value = True
        fake_audio.stat.return_value = SimpleNamespace(st_size=5, st_mtime=1)

        async def fake_run(command, timeout_seconds=None):
            calls.append((command, timeout_seconds))
            if "--dump-json" in command:
                return '{"duration": 120, "title": "Video"}'
            return ""

        with patch.object(Path, "mkdir"):
            with patch.object(Path, "glob", return_value=[fake_audio]):
                with patch.object(bot, "run_subprocess", new=fake_run):
                    audio_path, metadata = await bot.download_youtube_audio(
                        "https://youtu.be/abc", Path("youtube-test")
                    )

        self.assertEqual(metadata["title"], "Video")
        self.assertEqual(audio_path.name, "youtube_abc.mp3")
        self.assertEqual(len(calls), 2)
        self.assertIn("--dump-json", calls[0][0])
        self.assertIn("--audio-format", calls[1][0])

    async def test_process_youtube_url_falls_back_to_timestamped_whisper_and_cleans_temp(self):
        fake_bot = FakeBot()
        status = FakeStatusMessage()
        transcribe_calls = []
        sent_files = []

        temp_dir = MagicMock(spec=Path)
        downloaded = MagicMock(spec=Path)
        prepared = MagicMock(spec=Path)
        prepared.exists.return_value = True

        async def fake_metadata(_url):
            return {
                "title": "Fixture",
                "duration": 120,
                "chapters": [{"start_time": 2, "title": "Topic"}],
            }

        async def fake_captions(_url, _destination_dir, _metadata):
            return None

        async def fake_download(_url, _destination_dir, metadata=None):
            return downloaded, metadata

        async def fake_prepare(_path):
            return prepared

        async def fake_transcribe(path, mode="auto", progress_callback=None):
            transcribe_calls.append({"path": path, "mode": mode})
            if progress_callback is not None:
                await progress_callback(1, 2)
                await progress_callback(2, 2)
            return [{"start": 2, "end": 3, "text": "youtube text"}]

        async def fake_send(_bot, chat_id, transcript, title):
            sent_files.append((chat_id, transcript, title))

        bot.chat_transcription_modes[77] = "en"
        bot.youtube_active_jobs_by_chat[77] = 1
        try:
            with patch.object(bot, "create_temp_directory", return_value=temp_dir):
                with patch.object(bot, "read_youtube_metadata", new=fake_metadata):
                    with patch.object(bot, "download_youtube_captions", new=fake_captions):
                        with patch.object(bot, "download_youtube_audio", new=fake_download):
                            with patch.object(bot, "prepare_media_for_transcription", new=fake_prepare):
                                with patch.object(bot, "transcribe_audio_with_timestamps", new=fake_transcribe):
                                    with patch.object(bot, "send_youtube_transcription_file", new=fake_send):
                                        with patch.object(bot.shutil, "rmtree"):
                                            await bot.process_youtube_url(fake_bot, 77, "https://youtu.be/abc", status)
        finally:
            bot.chat_transcription_modes.pop(77, None)

        self.assertEqual(transcribe_calls[0]["mode"], "auto")
        self.assertEqual(
            sent_files,
            [
                (
                    77,
                    "СОДЕРЖАНИЕ ВИДЕО\n\n"
                    "0:02     │ Topic\n\n"
                    "РАСШИФРОВКА ВИДЕО\n\n"
                    "0:02     │ youtube text",
                    "Fixture",
                )
            ],
        )
        self.assertNotIn(77, bot.youtube_active_jobs_by_chat)
        prepared.unlink.assert_called_once()

    async def test_process_youtube_url_prefers_captions_without_downloading_audio(self):
        fake_bot = FakeBot()
        status = FakeStatusMessage()
        sent_files = []

        async def fake_metadata(_url):
            return {
                "title": "Captioned",
                "duration": 60,
                "chapters": [{"start_time": 0, "title": "Opening"}],
            }

        async def fake_captions(_url, _destination_dir, _metadata):
            return [{"start": 0, "end": 2, "text": "caption text"}]

        async def forbidden_audio(*_args, **_kwargs):
            raise AssertionError("audio fallback must not run when captions exist")

        async def fake_send(_bot, chat_id, transcript, title):
            sent_files.append((chat_id, transcript, title))

        temp_dir = MagicMock(spec=Path)
        bot.youtube_active_jobs_by_chat[78] = 1
        with patch.object(bot, "create_temp_directory", return_value=temp_dir):
            with patch.object(bot, "read_youtube_metadata", new=fake_metadata):
                with patch.object(bot, "download_youtube_captions", new=fake_captions):
                    with patch.object(bot, "download_youtube_audio", new=forbidden_audio):
                        with patch.object(bot, "send_youtube_transcription_file", new=fake_send):
                            with patch.object(bot.shutil, "rmtree"):
                                await bot.process_youtube_url(fake_bot, 78, "https://youtu.be/abc", status)

        self.assertEqual(
            sent_files,
            [
                (
                    78,
                    "СОДЕРЖАНИЕ ВИДЕО\n\n"
                    "0:00     │ Opening\n\n"
                    "РАСШИФРОВКА ВИДЕО\n\n"
                    "0:00     │ caption text",
                    "Captioned",
                )
            ],
        )
        self.assertNotIn(78, bot.youtube_active_jobs_by_chat)

    def test_copyright_block_is_explained_without_internal_details(self):
        message = bot.describe_processing_error(
            RuntimeError("ERROR: Video unavailable. It was blocked due to the claimed content by WMG.")
        )

        self.assertIn("Правообладатель", message)
        self.assertNotIn("WMG", message)
        self.assertNotIn("yt-dlp", message)

    async def test_process_youtube_url_reports_validation_error_and_cleans_state(self):
        fake_bot = FakeBot()
        status = FakeStatusMessage()

        async def fake_metadata(_url):
            raise ValueError("Видео слишком длинное.")

        bot.youtube_active_jobs_by_chat[88] = 1
        with patch.object(bot, "read_youtube_metadata", new=fake_metadata):
            await bot.process_youtube_url(fake_bot, 88, "https://youtu.be/too-long", status)

        self.assertNotIn(88, bot.youtube_active_jobs_by_chat)
        self.assertTrue(status.edits)
        self.assertIn("Не удалось открыть видео", status.edits[-1])

    def test_youtube_active_job_counter(self):
        bot.youtube_active_jobs_by_chat.pop(55, None)
        self.assertEqual(bot.youtube_active_jobs_for_chat(55), 0)
        bot.increment_youtube_active_jobs(55)
        bot.increment_youtube_active_jobs(55)
        self.assertEqual(bot.youtube_active_jobs_for_chat(55), 2)
        bot.decrement_youtube_active_jobs(55)
        self.assertEqual(bot.youtube_active_jobs_for_chat(55), 1)
        bot.decrement_youtube_active_jobs(55)
        self.assertEqual(bot.youtube_active_jobs_for_chat(55), 0)


class FormattingTests(unittest.TestCase):
    def test_progress_bar_clamps_values(self):
        self.assertEqual(bot.progress_bar(-10, width=4), "░░░░")
        self.assertEqual(bot.progress_bar(50, width=4), "██░░")
        self.assertEqual(bot.progress_bar(120, width=4), "████")

    def test_elapsed_format_is_compact(self):
        self.assertEqual(bot.format_elapsed(17), "17 сек")
        self.assertEqual(bot.format_elapsed(77), "1 мин 17 сек")

    def test_processing_status_contains_all_parts(self):
        status = bot.render_processing_status("Title", 70, "Stage", "Detail", 17)
        self.assertIn("Title", status)
        self.assertIn("70%", status)
        self.assertIn("Stage", status)
        self.assertIn("Detail", status)
        self.assertIn("Время: 17 сек", status)

    def test_merge_transcription_parts_appends_missing_tail(self):
        merged = bot.merge_transcription_parts(["a b c d", "b c d e f"])
        self.assertEqual(merged, "a b c d e f")

    def test_merge_transcription_parts_does_not_duplicate_contained_tail(self):
        merged = bot.merge_transcription_parts(["a b c d e f g h i", "c d e f g h i"])
        self.assertEqual(merged, "a b c d e f g h i")

    def test_split_text_for_llm_empty(self):
        self.assertEqual(bot.split_text_for_llm("   "), [])

    def test_split_text_for_llm_respects_max_chars(self):
        chunks = bot.split_text_for_llm("one. two. three. four.", max_chars=10)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))

    def test_split_text_for_llm_splits_oversized_word(self):
        chunks = bot.split_text_for_llm("abcdefghij", max_chars=3)
        self.assertEqual(chunks, ["abc", "def", "ghi", "j"])


class ArmenianFormattingTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_armenian_mode_is_unchanged(self):
        self.assertEqual(await bot.format_transcription_for_mode("hello", "en"), "hello")

    async def test_armenian_mode_adds_original_translation_and_latin(self):
        armenian = "\u0549\u0567, \u0576\u0561 \u0578\u0579\u056b\u0576\u0579 \u056b\u0576\u0571 \u0579\u056b \u0561\u057d\u0565\u056c\u0589"
        async def fake_translate(_text):
            return armenian

        with patch.object(bot, "translate_text_to_armenian", new=fake_translate):
            formatted = await bot.format_transcription_for_mode("source text", "hy")

        self.assertIn("Original:\nsource text", formatted)
        self.assertIn("Armenian:\n" + armenian, formatted)
        self.assertIn("Latin letters:\nChe, na vochinch indz chi asel.", formatted)

    async def test_send_result_uses_armenian_formatter(self):
        fake_bot = FakeBot()
        armenian = "\u0532\u0561\u0580\u0565\u0582 \u0561\u0577\u056d\u0561\u0580\u0570\u0589"
        async def fake_translate(_text):
            return armenian

        with patch.object(bot, "translate_text_to_armenian", new=fake_translate):
            await bot.send_transcription_result(fake_bot, 7, "hello world", mode="hy")

        self.assertEqual(len(fake_bot.messages), 1)
        text = fake_bot.messages[0]["text"]
        self.assertIn("Original:\nhello world", text)
        self.assertIn("Armenian:", text)
        self.assertIn("Latin letters:\nBarew ashkharh.", text)

    async def test_armenian_formatter_falls_back_when_translation_fails(self):
        async def broken_translate(_text):
            raise RuntimeError("translation failed")

        with patch.object(bot, "translate_text_to_armenian", new=broken_translate):
            with patch.object(bot.logging, "warning"):
                formatted = await bot.format_transcription_for_mode("plain fallback", "hy")

        self.assertIn("Original:\nplain fallback", formatted)
        self.assertIn("Armenian:\nplain fallback", formatted)
        self.assertIn("Latin letters:\nplain fallback", formatted)

    async def test_long_armenian_result_sends_preview_and_document(self):
        fake_bot = FakeBot()
        long_text = "x" * (bot.LONG_TRANSCRIPTION_TEXT_THRESHOLD + 50)

        async def fake_translate(_text):
            return long_text

        with patch.object(bot, "translate_text_to_armenian", new=fake_translate):
            await bot.send_transcription_result(fake_bot, 9, long_text, mode="hy")

        self.assertGreaterEqual(len(fake_bot.messages), 2)
        self.assertEqual(len(fake_bot.documents), 1)
        self.assertIn("Расшифровка длинная", fake_bot.messages[0]["text"])

    def test_armenian_transliteration_handles_common_digraphs(self):
        sample = "\u0532\u0561\u0580\u0565\u0582 \u0561\u0577\u056d\u0561\u0580\u0570\u0589 \u0548\u0582 \u0565\u057d \u0565\u0574\u0589"
        self.assertEqual(bot.transliterate_armenian_to_latin(sample), "Barew ashkharh. U yes yem.")

    def test_armenian_transliteration_preserves_non_armenian_text(self):
        self.assertEqual(bot.transliterate_armenian_to_latin("X5 HR 24"), "X5 HR 24")


class RealTextRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_russian_mode_keeps_real_russian_excerpt_unchanged(self):
        formatted = await bot.format_transcription_for_mode(RUSSIAN_INTERVIEW_EXCERPT, "ru")
        self.assertEqual(formatted, RUSSIAN_INTERVIEW_EXCERPT)

    async def test_english_mode_keeps_real_english_excerpt_unchanged(self):
        formatted = await bot.format_transcription_for_mode(ENGLISH_INTERVIEW_EXCERPT, "en")
        self.assertEqual(formatted, ENGLISH_INTERVIEW_EXCERPT)

    async def test_spanish_mode_keeps_recognized_text_unchanged(self):
        spanish = "No me dijo nada, solo dijo gracias por venir a la reunión."
        formatted = await bot.format_transcription_for_mode(spanish, "es")
        self.assertEqual(formatted, spanish)

    async def test_auto_mode_keeps_recognized_text_unchanged(self):
        mixed = "HR напишет feedback tomorrow."
        formatted = await bot.format_transcription_for_mode(mixed, "auto")
        self.assertEqual(formatted, mixed)

    async def test_armenian_mode_preserves_original_russian_excerpt(self):
        async def fake_translate(_text):
            return ARMENIAN_TRANSLATION_EXCERPT

        with patch.object(bot, "translate_text_to_armenian", new=fake_translate):
            formatted = await bot.format_transcription_for_mode(RUSSIAN_INTERVIEW_EXCERPT, "hy")

        self.assertIn("Original:\n" + RUSSIAN_INTERVIEW_EXCERPT, formatted)
        self.assertIn("Armenian:\n" + ARMENIAN_TRANSLATION_EXCERPT, formatted)
        self.assertIn("Latin letters:\nChe", formatted)
        self.assertIn("shnorhakalutyun", formatted)

    async def test_armenian_mode_preserves_original_english_excerpt(self):
        async def fake_translate(_text):
            return ARMENIAN_TRANSLATION_EXCERPT

        with patch.object(bot, "translate_text_to_armenian", new=fake_translate):
            formatted = await bot.format_transcription_for_mode(ENGLISH_INTERVIEW_EXCERPT, "hy")

        self.assertIn("Original:\n" + ENGLISH_INTERVIEW_EXCERPT, formatted)
        self.assertIn("Armenian:\n" + ARMENIAN_TRANSLATION_EXCERPT, formatted)

    def test_latin_letters_block_contains_no_armenian_letters_for_fixture(self):
        latin = bot.transliterate_armenian_to_latin(ARMENIAN_TRANSLATION_EXCERPT)
        self.assertIsNone(__import__("re").search(r"[\u0530-\u058F]", latin))
        self.assertIn("HR", bot.transliterate_armenian_to_latin("HR"))


class GroqModeParameterTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_raw_response_parser_is_awaited(self):
        expected = FakeTranscription()

        class AsyncRawResponse:
            headers = {}

            async def parse(self):
                return expected

        class RawResource:
            async def create(self, **_kwargs):
                return AsyncRawResponse()

        client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(with_raw_response=RawResource())
            )
        )

        result = await bot.create_groq_transcription(client, {"model": "whisper-large-v3"})

        self.assertIs(result, expected)

    async def transcribe_with_fake_path(self, mode):
        fake_client = FakeGroqClient()
        fake_path = SimpleNamespace(
            name="audio.ogg",
            stat=lambda: SimpleNamespace(st_size=123),
            open=lambda _mode: FakeAudioFile(),
        )
        with patch.object(bot, "get_groq_client", return_value=fake_client):
            result = await bot.transcribe_with_groq(fake_path, mode=mode)
        return result, fake_client.audio.transcriptions.calls[0], fake_client.closed

    async def test_russian_mode_forces_ru_language(self):
        result, kwargs, closed = await self.transcribe_with_fake_path("ru")
        self.assertEqual(result, "recognized text")
        self.assertEqual(kwargs["language"], "ru")
        self.assertTrue(closed)

    async def test_english_mode_forces_en_language(self):
        _, kwargs, _ = await self.transcribe_with_fake_path("en")
        self.assertEqual(kwargs["language"], "en")

    async def test_spanish_mode_forces_es_language(self):
        _, kwargs, _ = await self.transcribe_with_fake_path("es")
        self.assertEqual(kwargs["language"], "es")

    async def test_auto_mode_omits_language(self):
        _, kwargs, _ = await self.transcribe_with_fake_path("auto")
        self.assertNotIn("language", kwargs)

    async def test_armenian_mode_omits_language_for_auto_transcription(self):
        _, kwargs, _ = await self.transcribe_with_fake_path("hy")
        self.assertNotIn("language", kwargs)

    async def test_unknown_mode_falls_back_to_russian_language(self):
        _, kwargs, _ = await self.transcribe_with_fake_path("unknown")
        self.assertEqual(kwargs["language"], "ru")


class TranscriptionProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_backup_models_are_fixed_in_configuration(self):
        self.assertIn("deepgram:nova-3", bot.STT_BACKUP_MODELS)
        self.assertIn("openai:gpt-4o-transcribe", bot.STT_BACKUP_MODELS)
        self.assertIn("groq:whisper-large-v3-turbo", bot.STT_BACKUP_MODELS)

    def test_deepgram_payload_parses_text_and_timestamps(self):
        payload = {
            "results": {
                "utterances": [
                    {"start": 1.2, "end": 3.4, "transcript": "Hello world"},
                    {"start": 4.0, "end": 5.0, "transcript": "Next phrase"},
                ]
            }
        }
        segments = bot.deepgram_segments_from_payload(payload)
        self.assertEqual(segments[0], {"start": 1.2, "end": 3.4, "text": "Hello world"})
        self.assertEqual(bot.deepgram_text_from_payload(payload), "Hello world Next phrase")

    async def test_openai_provider_is_selected_by_one_setting(self):
        async def fake_request(_path, _mode, with_timestamps):
            self.assertFalse(with_timestamps)
            return {"text": "openai transcript"}

        with patch.object(bot, "STT_PROVIDER", "openai"):
            with patch.object(bot, "request_openai_transcription", new=fake_request):
                result = await bot.transcribe_with_selected_provider(Path("audio.mp3"), mode="auto")

        self.assertEqual(result, "openai transcript")

    async def test_empty_forced_language_retries_with_auto_detection(self):
        calls = []

        async def fake_transcribe(_path, mode="ru"):
            calls.append(mode)
            return "recognized automatically" if mode == "auto" else ""

        with patch.object(bot, "STT_PROVIDER", "groq"):
            with patch.object(bot, "transcribe_with_selected_provider", new=fake_transcribe):
                result = await bot.transcribe_with_empty_result_recovery(
                    Path("voice.mp3"), mode="ru"
                )

        self.assertEqual(result, "recognized automatically")
        self.assertEqual(calls, ["ru", "auto"])

    async def test_empty_auto_result_retries_with_groq_fallback_model(self):
        fallback_calls = []

        async def fake_selected(_path, mode="ru"):
            self.assertEqual(mode, "auto")
            return ""

        async def fake_groq(_path, mode="ru", model=None):
            fallback_calls.append((mode, model))
            return "recognized by turbo"

        with patch.object(bot, "STT_PROVIDER", "groq"):
            with patch.object(bot, "GROQ_WHISPER_MODEL", "whisper-large-v3"):
                with patch.object(bot, "GROQ_WHISPER_FALLBACK_MODEL", "whisper-large-v3-turbo"):
                    with patch.object(bot, "transcribe_with_selected_provider", new=fake_selected):
                        with patch.object(bot, "transcribe_with_groq", new=fake_groq):
                            result = await bot.transcribe_with_empty_result_recovery(
                                Path("voice.mp3"), mode="auto"
                            )

        self.assertEqual(result, "recognized by turbo")
        self.assertEqual(fallback_calls, [("auto", "whisper-large-v3-turbo")])

    async def test_all_empty_attempts_raise_instead_of_reporting_ready(self):
        async def fake_recovery(_path, mode="ru"):
            return ""

        async def no_tail(_path, mode="ru"):
            return ""

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "voice.mp3"
            audio_path.write_bytes(b"not-empty")
            with patch.object(bot, "transcribe_with_empty_result_recovery", new=fake_recovery):
                with patch.object(bot, "transcribe_tail_for_recovery", new=no_tail):
                    with self.assertRaises(bot.EmptyTranscriptionError):
                        await bot.transcribe_audio_safely(audio_path, mode="ru")

    def test_empty_transcription_error_has_user_safe_explanation(self):
        reason = bot.describe_processing_error(bot.EmptyTranscriptionError("internal details"))
        self.assertIn("Речь не обнаружена", reason)
        self.assertNotIn("internal details", reason)


if __name__ == "__main__":
    unittest.main()
