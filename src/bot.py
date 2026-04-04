import asyncio
import logging
import os
import ssl
import traceback
import uuid
from typing import Dict, Optional

from aiogram import Bot
from aiogram import Dispatcher
from aiogram import F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from aiogram.types import CallbackQuery
from aiogram.types import InlineKeyboardButton
from aiogram.types import InlineKeyboardMarkup
from aiogram.types import Message
from dotenv import load_dotenv
import httpx
import openai

from conspectum.logger import Logger
from conspectum.process import process

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

HELP_TEXT = (
    "Send me a WAV file (.wav) — I will recognize the speech and return a PDF summary.\n\n"
    "Supported formats: 'File' (document) and 'Audio'.\n"
    "File size limit: 20 MB (Telegram API limit).\n"
    "For larger files, use the web interface.\n\n"
    "Commands:\n"
    "/language - choose summary language (Russian/English/auto)\n"
    "/status - show current settings"
)

# User settings storage
user_settings: Dict[int, Dict[str, Optional[str]]] = {}


def is_wav_message(message: Message) -> bool:
    # WAV usually comes as document, but can also come as audio
    if message.document:
        filename = (message.document.file_name or "").lower()
        mime = (message.document.mime_type or "").lower()
        return filename.endswith(".wav") or mime in {"audio/wav", "audio/x-wav"}
    if message.audio:
        filename = (message.audio.file_name or "").lower()
        mime = (message.audio.mime_type or "").lower()
        return filename.endswith(".wav") or mime in {"audio/wav", "audio/x-wav"}
    return False


def get_user_settings(user_id: int) -> Dict[str, Optional[str]]:
    if user_id not in user_settings:
        user_settings[user_id] = {"language": None}
    return user_settings[user_id]


def get_language_keyboard() -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇷🇺 Russian", callback_data="lang_ru"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
            ],
            [
                InlineKeyboardButton(text="🌐 Auto-detect", callback_data="lang_auto"),
            ],
        ]
    )
    return keyboard


def get_language_name(lang: Optional[str]) -> str:
    if lang == "ru":
        return "🇷🇺 Russian"
    elif lang == "en":
        return "🇬🇧 English"
    else:
        return "🌐 Auto-detect"


class TelegramLogger(Logger):
    def __init__(self, message: Message):
        id = uuid.uuid4()
        os.makedirs(f"logs/{id}")
        super().__init__(f"logs/{id}")
        self.message_ = message
        self.progress_message_id = None

    async def partial_result(self, text: str):
        await self.message_.answer(text, parse_mode="HTML")

    async def progress(self, completed: int, total: int):
        progress_text = f"📊 <b>Progress:</b> {round(completed / total * 100)}%"
        if self.progress_message_id:
            try:
                await self.message_.bot.edit_message_text(
                    progress_text,
                    chat_id=self.message_.chat.id,
                    message_id=self.progress_message_id,
                    parse_mode="HTML"
                )
            except Exception:
                # If edit fails, send new message
                msg = await self.message_.answer(progress_text, parse_mode="HTML")
                self.progress_message_id = msg.message_id
        else:
            msg = await self.message_.answer(progress_text, parse_mode="HTML")
            self.progress_message_id = msg.message_id


async def main():
    # Check required environment variables
    required_vars = ["BOT_TOKEN", "AI_BASE_URL", "AI_API_KEY", "MODEL_NAME"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        return
    
    logger.info("Starting bot...")
    logger.info(f"AI Base URL: {os.environ['AI_BASE_URL']}")
    logger.info(f"Model: {os.environ['MODEL_NAME']}")
    
    bot = Bot(token=os.environ["BOT_TOKEN"])
    dp = Dispatcher()

    # Create httpx client with SSL verification disabled for internal Yandex API
    # This is safe for internal APIs with self-signed certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    http_client = httpx.AsyncClient(
        verify=ssl_context,
        timeout=300.0  # 5 minutes timeout for long audio processing
    )
    
    ai = openai.AsyncOpenAI(
        base_url=os.environ["AI_BASE_URL"], 
        api_key=os.environ["AI_API_KEY"],
        http_client=http_client
    )
    
    logger.info("OpenAI client configured with SSL verification disabled for internal API")

    @dp.message(Command("start"))
    async def cmd_start(message: Message):
        await message.answer(HELP_TEXT)

    @dp.message(Command("language"))
    async def cmd_language(message: Message):
        await message.answer(
            "Choose the language for summary generation:",
            reply_markup=get_language_keyboard()
        )

    @dp.message(Command("status"))
    async def cmd_status(message: Message):
        settings = get_user_settings(message.from_user.id)
        lang_name = get_language_name(settings["language"])
        await message.answer(
            f"<b>Your settings:</b>\n\n"
            f"Summary language: {lang_name}",
            parse_mode="HTML"
        )

    @dp.callback_query(F.data.startswith("lang_"))
    async def callback_language(callback: CallbackQuery):
        lang_code = callback.data.split("_")[1]
        user_id = callback.from_user.id
        
        settings = get_user_settings(user_id)
        if lang_code == "auto":
            settings["language"] = None
            lang_name = "🌐 Auto-detect"
        else:
            settings["language"] = lang_code
            lang_name = get_language_name(lang_code)
        
        await callback.message.edit_text(
            f"✅ Summary language set to: {lang_name}\n\n"
            f"Now send a WAV file to process.",
            reply_markup=None
        )
        await callback.answer()

    @dp.message(F.document | F.audio)
    async def handle_wav(message: Message):
        if not is_wav_message(message):
            await message.answer("WAV file (.wav) is required.")
            return

        # Get user settings
        user_id = message.from_user.id
        settings = get_user_settings(user_id)
        selected_language = settings["language"]

        # Check file size (Telegram Bot API limit is 20 MB)
        tg_file = message.document or message.audio
        file_size_mb = (tg_file.file_size or 0) / (1024 * 1024)
        
        if file_size_mb > 20:
            await message.answer(
                f"❌ File is too large ({file_size_mb:.1f} MB).\n\n"
                f"Telegram Bot API limit is 20 MB.\n"
                f"Please compress the audio or use the web interface for larger files."
            )
            return

        # Download file to memory
        tg_file = message.document or message.audio
        file_size_mb = (tg_file.file_size or 0) / (1024 * 1024)
        try:
            logger.info(f"Downloading file for user {user_id}, size: {file_size_mb:.2f} MB")
            file = await bot.get_file(tg_file.file_id)
            file_bytes = await bot.download_file(file.file_path)
            audio_bytes = file_bytes.read()
            logger.info(f"File downloaded successfully, {len(audio_bytes)} bytes")
        except Exception as e:
            error_msg = f"❌ Failed to download file: {e}"
            logger.error(f"Error downloading file for user {user_id}:")
            logger.error(traceback.format_exc())
            await message.answer(error_msg)
            return

        if selected_language:
            lang_name = get_language_name(selected_language)
            await message.answer(f"📝 WAV file received. Generating summary in: {lang_name}")
        else:
            await message.answer("📝 WAV file received. Detecting language and generating summary...")

        try:
            tex, pdf = await process(audio_bytes, ai, TelegramLogger(message), language=selected_language)
        except Exception as e:
            error_msg = f"❌ Failed to generate PDF: {e}"
            logger.error(f"Error processing audio for user {user_id}:")
            logger.error(traceback.format_exc())
            await message.answer(error_msg)
            return

        tex_name = "result.tex"
        tex_file = BufferedInputFile(bytes(tex, encoding="utf-8"), filename=tex_name)
        await message.answer_document(tex_file)

        if pdf is not None:
            pdf_name = "result.pdf"
            pdf_file = BufferedInputFile(pdf, filename=pdf_name)
            await message.answer_document(pdf_file)
            await message.answer("✅ Summary is ready!")

    logger.info("Bot started successfully! Waiting for messages...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
