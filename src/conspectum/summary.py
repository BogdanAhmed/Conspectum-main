import base64
import dataclasses
import io
import os
import pathlib
import pydub
import tempfile

import openai
from faster_whisper import WhisperModel

from .logger import Logger

LANGUAGE_DETECTION_MS = 30_000

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
}


@dataclasses.dataclass
class Summary:
    title: str
    abstract: str


async def transcribe_audio(wav_file: bytes, logger: Logger = Logger()) -> str:
    """Transcribe audio to text using local Whisper model."""
    # Load model (will download on first use)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(wav_file)
        temp_path = temp_file.name
    
    try:
        # Transcribe
        segments, info = model.transcribe(temp_path, beam_size=5)
        
        # Collect all text
        transcript = ""
        for segment in segments:
            transcript += segment.text
        
        await logger.partial_result(f"🎤 <b>Transcription complete:</b> {len(transcript)} characters")
        await logger.file("transcript", transcript, Logger.FileType.TEXT)
        
        return transcript.strip()
    finally:
        # Clean up temp file
        os.unlink(temp_path)


async def detect_language_from_text(text: str, ai: openai.AsyncOpenAI) -> str:
    """Detect language from text using AI."""
    prompt = f"Detect the language of this text. Return only 'ru' for Russian or 'en' for English:\n\n{text[:1000]}"
    
    response = await ai.chat.completions.create(
        model="gpt-4o-mini",  # Use cheaper model for language detection
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.1
    )
    
    detected = response.choices[0].message.content.strip().lower()
    if detected not in ("ru", "en"):
        return "en"  # Default to English
    return detected


async def detect_language(wav_file: bytes, ai: openai.AsyncOpenAI) -> str:
    # First transcribe audio locally
    transcript = await transcribe_audio(wav_file)
    
    # Then detect language from text
    return await detect_language_from_text(transcript, ai)


async def make_summary(wav_file: bytes, ai: openai.AsyncOpenAI, language: str, logger: Logger = Logger()) -> Summary:
    # First transcribe the audio locally
    transcript = await transcribe_audio(wav_file, logger)
    
    # Then create summary from text
    with open(pathlib.Path(__file__).parent / "prompts/summary_prompt.txt") as prompt_file:
        prompt = prompt_file.read()
    prompt = prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])
    
    # Use cheaper model for text processing
    response = await ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Create a summary from this lecture transcript:\n\n{transcript}"},
        ],
        temperature=0.3
    )
    response = response.choices[0].message.content
    await logger.file("summary", response, Logger.FileType.TEXT)

    response = response.split("\n\n", maxsplit=1)
    title, abstract = response[0], response[1]
    if title.lower().startswith("title: "):
        title = title[len("title: ") :]

    await logger.partial_result(f"<b>The topic of the lecture:</b>\n{title}")
    await logger.partial_result(f"<b>The abstract of the lecture:</b>\n{abstract}")
    return Summary(title=title, abstract=abstract)


async def postprocess_summary(
    tex_content: str,
    ai: openai.AsyncOpenAI,
    language: str,
    logger: Logger = Logger()
) -> str:
    """
    Postprocess the LaTeX summary by checking for logical errors and adding references.
    
    Args:
        tex_content: The complete LaTeX document content
        ai: OpenAI API client (will use gpt-4.1 for postprocessing)
        language: Language of the summary ('ru' or 'en')
        logger: Logger for tracking progress
        
    Returns:
        Enhanced LaTeX content with corrections and references
    """
    with open(pathlib.Path(__file__).parent / "prompts/postprocess_prompt.txt") as prompt_file:
        prompt = prompt_file.read()
    
    prompt = prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])
    
    await logger.partial_result("🔍 <b>Starting postprocessing:</b> checking for errors and adding references...")
    
    # Use gpt-4.1 for higher quality postprocessing
    response = await ai.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Review and enhance this lecture summary:\n\n{tex_content}"
            },
        ],
        temperature=0.3,  # Lower temperature for more consistent output
    )
    
    enhanced_content = response.choices[0].message.content
    
    # Remove code block markers if present
    if enhanced_content.startswith("```"):
        lines = enhanced_content.split("\n")
        # Remove first line (```latex or similar)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        enhanced_content = "\n".join(lines)
    
    # Basic validation: check that essential LaTeX structure is preserved
    required_elements = [
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}"
    ]
    
    missing_elements = [elem for elem in required_elements if elem not in enhanced_content]
    
    if missing_elements:
        await logger.partial_result(
            f"⚠️ <b>Postprocessing validation failed:</b> missing {', '.join(missing_elements)}. "
            f"Using original version."
        )
        return tex_content
    
    # Check that document is not truncated
    if not enhanced_content.strip().endswith(r"\end{document}"):
        await logger.partial_result(
            "⚠️ <b>Postprocessing warning:</b> document appears truncated. "
            "Using original version."
        )
        return tex_content
    
    await logger.file("lecture_postprocessed", enhanced_content, Logger.FileType.TEX)
    await logger.partial_result("✅ <b>Postprocessing complete:</b> errors corrected and references added")
    
    return enhanced_content
