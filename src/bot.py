import asyncio
import logging
import os
import ssl
import traceback
import uuid
from typing import Dict, Optional

import httpx
import openai
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram import Dispatcher
from aiogram import F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from aiogram.types import CallbackQuery
from aiogram.types import InlineKeyboardButton
from aiogram.types import InlineKeyboardMarkup
from aiogram.types import Message
from dotenv import load_dotenv

from conspectum.logger import Logger
from conspectum.process import process
from conspectum.summary import guess_audio_suffix
from conspectum.summary import is_supported_audio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

user_settings: Dict[int, Dict[str, Optional[str]]] = {}


def get_help_text() -> str:
    return (
        "Send me an audio file and I will turn it into study materials.\n\n"
        "Supported formats: .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, .webm.\n"
        "Telegram voice messages are supported too.\n"
        "The bot can process files that Telegram Bot API allows it to download.\n\n"
        "What you get:\n"
        "- transcript.txt\n"
        "- result.tex\n"
        "- result.pdf\n\n"
        "Commands:\n"
        "/language - choose summary language (Russian/English/auto)\n"
        "/detail - choose summary detail level\n"
        "/status - show current settings"
    )


def build_telegram_bot() -> Bot:
    telegram_timeout = float(os.environ.get("TELEGRAM_TIMEOUT_SECONDS", "900"))
    session = AiohttpSession(timeout=telegram_timeout)
    return Bot(token=os.environ["BOT_TOKEN"], session=session)


def is_audio_message(message: Message) -> bool:
    if message.document:
        return is_supported_audio(message.document.file_name, message.document.mime_type)
    if message.audio:
        return is_supported_audio(message.audio.file_name, message.audio.mime_type)
    if message.voice:
        return is_supported_audio("voice.ogg", message.voice.mime_type or "audio/ogg")
    return False


def get_user_settings(user_id: int) -> Dict[str, Optional[str]]:
    if user_id not in user_settings:
        user_settings[user_id] = {"language": None, "detail": "standard"}
    return user_settings[user_id]


def get_language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Russian", callback_data="lang_ru"),
                InlineKeyboardButton(text="English", callback_data="lang_en"),
            ],
            [
                InlineKeyboardButton(text="Auto", callback_data="lang_auto"),
            ],
        ]
    )


def get_detail_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Quick", callback_data="detail_brief"),
                InlineKeyboardButton(text="Balanced", callback_data="detail_standard"),
                InlineKeyboardButton(text="Deep", callback_data="detail_detailed"),
            ],
        ]
    )


def get_language_name(lang: Optional[str]) -> str:
    if lang == "ru":
        return "Russian"
    if lang == "en":
        return "English"
    return "Auto"


def get_detail_name(detail: Optional[str]) -> str:
    if detail == "brief":
        return "Quick"
    if detail == "detailed":
        return "Deep"
    return "Balanced"


async def answer_plain_safely(message: Message, text: str):
    try:
        return await message.answer(text, parse_mode=None)
    except Exception as exc:
        logger.warning("Failed to send plain bot reply: %s", exc)
        return None


class TelegramLogger(Logger):
    def __init__(self, message: Message):
        log_id = uuid.uuid4()
        os.makedirs(f"logs/{log_id}", exist_ok=True)
        super().__init__(f"logs/{log_id}")
        self.message_ = message
        self.progress_message_id = None

    async def partial_result(self, text: str):
        await self._send_html_safely(text)

    async def _send_html_safely(self, text: str):
        try:
            return await self.message_.answer(text, parse_mode="HTML")
        except TelegramBadRequest as exc:
            if "can't parse entities" not in str(exc).lower():
                logger.warning("Failed to send HTML Telegram message: %s", exc)
                return None
            return await self._send_plain_safely(text)
        except Exception as exc:
            logger.warning("Failed to send Telegram message: %s", exc)
            return None

    async def _send_plain_safely(self, text: str):
        try:
            return await self.message_.answer(text, parse_mode=None)
        except Exception as exc:
            logger.warning("Failed to send plain Telegram message: %s", exc)
            return None

    async def progress(self, completed: int, total: int):
        progress_text = f"<b>Progress:</b> {round(completed / total * 100)}%"
        if self.progress_message_id:
            try:
                await self.message_.bot.edit_message_text(
                    progress_text,
                    chat_id=self.message_.chat.id,
                    message_id=self.progress_message_id,
                    parse_mode="HTML",
                )
                return
            except Exception:
                pass

        msg = await self._send_html_safely(progress_text)
        if msg is not None:
            self.progress_message_id = msg.message_id


async def main():
    required_vars = ["BOT_TOKEN", "AI_BASE_URL", "AI_API_KEY", "MODEL_NAME"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        logger.error("Missing required environment variables: %s", ", ".join(missing_vars))
        logger.error("Please check your .env file")
        return

    logger.info("Starting bot...")
    logger.info("AI Base URL: %s", os.environ["AI_BASE_URL"])
    logger.info("Model: %s", os.environ["MODEL_NAME"])

    bot = build_telegram_bot()
    dp = Dispatcher()

    allow_insecure_ssl = os.environ.get("ALLOW_INSECURE_SSL", "").lower() in {"1", "true", "yes"}
    verify: bool | ssl.SSLContext = True
    if allow_insecure_ssl:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        verify = ssl_context
        logger.warning("SSL verification is disabled because ALLOW_INSECURE_SSL is enabled")

    http_client = httpx.AsyncClient(
        verify=verify,
        timeout=300.0,
    )

    ai = openai.AsyncOpenAI(
        base_url=os.environ["AI_BASE_URL"],
        api_key=os.environ["AI_API_KEY"],
        http_client=http_client,
    )

    logger.info("OpenAI client configured")

    @dp.message(Command("start"))
    @dp.message(Command("help"))
    async def cmd_start(message: Message):
        await answer_plain_safely(message, get_help_text())

    @dp.message(Command("language"))
    async def cmd_language(message: Message):
        await message.answer(
            "Choose the language for summary generation:",
            reply_markup=get_language_keyboard(),
        )

    @dp.message(Command("detail"))
    async def cmd_detail(message: Message):
        await message.answer(
            "Choose how detailed the summary should be:",
            reply_markup=get_detail_keyboard(),
        )

    @dp.message(Command("status"))
    async def cmd_status(message: Message):
        settings = get_user_settings(message.from_user.id)
        await answer_plain_safely(
            message,
            (
                "Your settings:\n\n"
                f"Language: {get_language_name(settings['language'])}\n"
                f"Detail level: {get_detail_name(settings['detail'])}"
            ),
        )

    @dp.callback_query(F.data.startswith("lang_"))
    async def callback_language(callback: CallbackQuery):
        lang_code = callback.data.split("_", maxsplit=1)[1]
        settings = get_user_settings(callback.from_user.id)

        if lang_code == "auto":
            settings["language"] = None
        else:
            settings["language"] = lang_code

        if callback.message is not None:
            await callback.message.edit_text(
                f"Summary language set to: {get_language_name(settings['language'])}",
                reply_markup=None,
            )
        await callback.answer()

    @dp.callback_query(F.data.startswith("detail_"))
    async def callback_detail(callback: CallbackQuery):
        detail = callback.data.split("_", maxsplit=1)[1]
        settings = get_user_settings(callback.from_user.id)
        settings["detail"] = detail

        if callback.message is not None:
            await callback.message.edit_text(
                f"Detail level set to: {get_detail_name(detail)}",
                reply_markup=None,
            )
        await callback.answer()

    @dp.message(F.document | F.audio | F.voice)
    async def handle_audio(message: Message):
        if not is_audio_message(message):
            await answer_plain_safely(
                message,
                "Supported audio formats: .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, .webm.",
            )
            return

        user_id = message.from_user.id
        settings = get_user_settings(user_id)
        selected_language = settings["language"]
        selected_detail = settings["detail"] or "standard"

        tg_file = message.document or message.audio or message.voice
        file_size_mb = (tg_file.file_size or 0) / (1024 * 1024)

        try:
            logger.info("Downloading file for user %s, size: %.2f MB", user_id, file_size_mb)
            file = await bot.get_file(tg_file.file_id)
            file_bytes = await bot.download_file(file.file_path)
            audio_bytes = file_bytes.read()
            logger.info("File downloaded successfully, %s bytes", len(audio_bytes))
        except Exception as exc:
            error_msg = f"Failed to download file: {exc}"
            if "file is too big" in str(exc).lower():
                error_msg = (
                    "Failed to download file: Telegram Bot API rejected it as too large.\n\n"
                    "For ordinary bots, Telegram currently allows downloading files only up to 20 MB. "
                    "This limit comes from Telegram itself, not from the project code."
                )
            logger.error("Error downloading file for user %s:", user_id)
            logger.error(traceback.format_exc())
            await answer_plain_safely(message, error_msg)
            return

        status_line = (
            f"Audio file received. Language: {get_language_name(selected_language)}. "
            f"Detail level: {get_detail_name(selected_detail)}."
        )
        await answer_plain_safely(message, status_line)

        audio_filename = getattr(tg_file, "file_name", None)
        audio_mime_type = getattr(tg_file, "mime_type", None)
        if not audio_filename:
            audio_filename = f"input{guess_audio_suffix(mime_type=audio_mime_type)}"

        try:
            result = await process(
                audio_bytes,
                ai,
                TelegramLogger(message),
                language=selected_language,
                detail_level=selected_detail,
                audio_filename=audio_filename,
                audio_mime_type=audio_mime_type,
            )
        except Exception as exc:
            error_msg = f"Failed to generate PDF: {exc}"
            logger.error("Error processing audio for user %s:", user_id)
            logger.error(traceback.format_exc())
            await answer_plain_safely(message, error_msg)
            return

        await answer_plain_safely(
            message,
            (
                f"Title: {result.title}\n"
                f"Detected language: {get_language_name(result.language)}\n"
                f"Detail level: {get_detail_name(selected_detail)}"
            ),
        )

        transcript_file = BufferedInputFile(
            result.transcript.encode("utf-8"),
            filename="transcript.txt",
        )
        await message.answer_document(transcript_file)

        tex_file = BufferedInputFile(
            result.tex.encode("utf-8"),
            filename="result.tex",
        )
        await message.answer_document(tex_file)

        if result.pdf is not None:
            pdf_file = BufferedInputFile(result.pdf, filename="result.pdf")
            await message.answer_document(pdf_file)
            if result.pdf_warning:
                await answer_plain_safely(message, result.pdf_warning)
            await answer_plain_safely(message, "Summary is ready.")
        else:
            await answer_plain_safely(
                message,
                result.pdf_warning or "PDF compilation failed, but transcript.txt and result.tex are ready.",
            )

    logger.info("Bot started successfully! Waiting for messages...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
