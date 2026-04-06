import dataclasses
import os
import pathlib
import re
import tempfile
from pathlib import Path

import openai
from faster_whisper import WhisperModel

from .logger import Logger

LANGUAGE_DETECTION_MS = 30_000

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
}

DETAIL_LEVEL_PROMPTS = {
    "brief": (
        "Keep the overview compact. Mention only the central topic, the main ideas, and the final takeaway."
    ),
    "standard": (
        "Provide a balanced overview with the main topic, the most important concepts, and the key conclusion."
    ),
    "detailed": (
        "Provide a richer overview that mentions the learning goals, major definitions, formulas, examples, and conclusions."
    ),
}

_TRANSCRIPTION_MODEL: WhisperModel | None = None

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".oga",
    ".opus",
    ".flac",
    ".aac",
    ".mp4",
    ".m4b",
    ".webm",
}

SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
    "video/webm": ".webm",
}


@dataclasses.dataclass
class Summary:
    title: str
    abstract: str


def guess_audio_suffix(filename: str | None = None, mime_type: str | None = None) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in SUPPORTED_AUDIO_EXTENSIONS:
            return suffix

    if mime_type:
        normalized_mime = mime_type.lower().strip()
        if normalized_mime in SUPPORTED_AUDIO_MIME_TYPES:
            return SUPPORTED_AUDIO_MIME_TYPES[normalized_mime]

    return ".wav"


def is_supported_audio(filename: str | None = None, mime_type: str | None = None) -> bool:
    if filename and Path(filename).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
        return True

    if mime_type and mime_type.lower().strip() in SUPPORTED_AUDIO_MIME_TYPES:
        return True

    return False


def get_transcription_model() -> WhisperModel:
    global _TRANSCRIPTION_MODEL
    if _TRANSCRIPTION_MODEL is None:
        _TRANSCRIPTION_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _TRANSCRIPTION_MODEL


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_latex_text(text: str) -> str:
    normalized = strip_markdown_fences(text)

    substitutions = [
        (r"\*\*(.+?)\*\*", r"\\textbf{\1}"),
        (r"__(.+?)__", r"\\textbf{\1}"),
    ]

    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized, flags=re.DOTALL)

    normalized = normalized.replace("–", "--")
    normalized = normalized.replace("—", "---")
    return normalized.strip()


def parse_summary_response(response_text: str) -> Summary:
    cleaned = normalize_latex_text(response_text)
    parts = [part.strip() for part in cleaned.split("\n\n", maxsplit=1)]

    title = parts[0] if parts else "Lecture Summary"
    abstract = parts[1] if len(parts) > 1 else ""

    title = re.sub(r"^(title|заголовок)\s*:\s*", "", title, flags=re.IGNORECASE)
    abstract = re.sub(r"^(abstract|summary|аннотация)\s*:\s*", "", abstract, flags=re.IGNORECASE)

    if not abstract:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 2:
            title = re.sub(r"^(title|заголовок)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
            abstract = " ".join(lines[1:])
        else:
            abstract = cleaned

    return Summary(title=title.strip(), abstract=abstract.strip())


async def transcribe_audio(
    audio_file: bytes,
    logger: Logger = Logger(),
    filename: str | None = None,
    mime_type: str | None = None,
) -> str:
    """Transcribe audio to text using local Whisper model."""
    model = get_transcription_model()
    
    suffix = guess_audio_suffix(filename=filename, mime_type=mime_type)

    # Save audio to a temp file with its original suffix so ffmpeg/Whisper can decode it correctly.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(audio_file)
        temp_path = temp_file.name
    
    try:
        await logger.stage("transcribing", 0)

        # Transcribe
        segments, info = model.transcribe(temp_path, beam_size=5)
        
        # Collect all text
        transcript = ""
        total_duration = float(getattr(info, "duration", 0) or 0)
        last_reported_progress = -1

        for segment in segments:
            transcript += segment.text
            if total_duration > 0:
                segment_end = float(getattr(segment, "end", 0) or 0)
                percent = max(0, min(100, round(segment_end / total_duration * 100)))
                if percent >= last_reported_progress + 3 or percent >= 100:
                    await logger.progress(percent, 100)
                    last_reported_progress = percent

        await logger.stage("transcript_ready", 100)
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
    
    detected = (response.choices[0].message.content or "").strip().lower()
    if detected not in ("ru", "en"):
        return "en"  # Default to English
    return detected


async def detect_language(
    audio_file: bytes,
    ai: openai.AsyncOpenAI,
    filename: str | None = None,
    mime_type: str | None = None,
) -> str:
    transcript = await transcribe_audio(audio_file, filename=filename, mime_type=mime_type)
    return await detect_language_from_text(transcript, ai)


async def make_summary_from_transcript(
    transcript: str,
    ai: openai.AsyncOpenAI,
    language: str,
    logger: Logger = Logger(),
    detail_level: str = "standard",
) -> Summary:
    with open(pathlib.Path(__file__).parent / "prompts/summary_prompt.txt", encoding="utf-8", errors="replace") as prompt_file:
        prompt = prompt_file.read()
    prompt = prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])

    await logger.stage("summary", 10)

    response = await ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "system",
                "content": (
                    f"Summary detail level: {detail_level}. "
                    f"{DETAIL_LEVEL_PROMPTS.get(detail_level, DETAIL_LEVEL_PROMPTS['standard'])}"
                ),
            },
            {"role": "user", "content": f"Create a summary from this lecture transcript:\n\n{transcript}"},
        ],
        temperature=0.3
    )
    response_text = response.choices[0].message.content or ""
    await logger.file("summary", response_text, Logger.FileType.TEXT)

    summary = parse_summary_response(response_text)

    await logger.stage("summary", 70)
    await logger.partial_result(f"<b>The topic of the lecture:</b>\n{summary.title}")
    await logger.stage("summary", 100)
    await logger.partial_result(f"<b>The abstract of the lecture:</b>\n{summary.abstract}")
    return summary


async def make_summary(
    audio_file: bytes,
    ai: openai.AsyncOpenAI,
    language: str,
    logger: Logger = Logger(),
    detail_level: str = "standard",
    filename: str | None = None,
    mime_type: str | None = None,
) -> Summary:
    transcript = await transcribe_audio(audio_file, logger, filename=filename, mime_type=mime_type)
    return await make_summary_from_transcript(transcript, ai, language, logger, detail_level=detail_level)


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
    with open(pathlib.Path(__file__).parent / "prompts/postprocess_prompt.txt", encoding="utf-8", errors="replace") as prompt_file:
        prompt = prompt_file.read()
    
    prompt = prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])

    await logger.stage("postprocess", 10)
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
    
    enhanced_content = normalize_latex_text(response.choices[0].message.content or "")
    
    # Basic validation: check that essential LaTeX structure is preserved
    required_elements = [
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}"
    ]
    
    missing_elements = [elem for elem in required_elements if elem not in enhanced_content]
    
    if missing_elements:
        await logger.stage("postprocess", 100)
        await logger.partial_result(
            f"⚠️ <b>Postprocessing validation failed:</b> missing {', '.join(missing_elements)}. "
            f"Using original version."
        )
        return tex_content
    
    # Check that document is not truncated
    if not enhanced_content.strip().endswith(r"\end{document}"):
        await logger.stage("postprocess", 100)
        await logger.partial_result(
            "⚠️ <b>Postprocessing warning:</b> document appears truncated. "
            "Using original version."
        )
        return tex_content
    
    await logger.file("lecture_postprocessed", enhanced_content, Logger.FileType.TEX)
    await logger.stage("postprocess", 100)
    await logger.partial_result("✅ <b>Postprocessing complete:</b> errors corrected and references added")
    
    return enhanced_content
