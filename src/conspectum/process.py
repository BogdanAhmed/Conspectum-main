import base64
import io
import pathlib
import typing
import uuid

import openai
import pdflatex
import pydub
import pydub.silence
import os
import shutil

from .logger import Logger
from .summary import make_summary, detect_language, postprocess_summary, LANGUAGE_NAMES, transcribe_audio

if not shutil.which("pdflatex"):
    _tex_bin = "/Library/TeX/texbin"
    if os.path.isdir(_tex_bin):
        os.environ["PATH"] = _tex_bin + os.pathsep + os.environ.get("PATH", "")

LANG_CONFIG = {
    "en": {
        "fontenc": "T1",
        "babel": "english",
        "theorem": "Theorem",
        "definition": "Definition",
        "lemma": "Lemma",
        "proposition": "Proposition",
        "corollary": "Corollary",
        "example": "Example",
        "remark": "Remark",
    },
    "ru": {
        "fontenc": "T2A",
        "babel": "russian",
        "theorem": "Теорема",
        "definition": "Определение",
        "lemma": "Лемма",
        "proposition": "Утверждение",
        "corollary": "Следствие",
        "example": "Пример",
        "remark": "Замечание",
    },
}


def localize_template(tex_template: str, language: str) -> str:
    config = LANG_CONFIG[language]
    replacements = {
        "<FONTENC>": config["fontenc"],
        "<BABEL_LANG>": config["babel"],
        "<THEOREM_NAME>": config["theorem"],
        "<DEFINITION_NAME>": config["definition"],
        "<LEMMA_NAME>": config["lemma"],
        "<PROPOSITION_NAME>": config["proposition"],
        "<COROLLARY_NAME>": config["corollary"],
        "<EXAMPLE_NAME>": config["example"],
        "<REMARK_NAME>": config["remark"],
    }
    for placeholder, value in replacements.items():
        tex_template = tex_template.replace(placeholder, value)
    return tex_template


async def split_into_chunks(transcript: str, logger: Logger = Logger()) -> typing.List[str]:
    """Split transcript into text chunks instead of audio chunks."""
    # Split by sentences/paragraphs, aim for ~1000-2000 characters per chunk
    sentences = transcript.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > 1500:  # If adding this sentence would exceed limit
            if current_chunk:  # Save current chunk if not empty
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:  # If single sentence is too long, add it anyway
                chunks.append(sentence.strip() + ". ")
        else:
            current_chunk += sentence + ". "
    
    if current_chunk:  # Add remaining chunk
        chunks.append(current_chunk.strip())
    
    # Log chunks
    for i, chunk in enumerate(chunks):
        await logger.file(f"chunk_{i + 1}_text", chunk, Logger.FileType.TEXT)
    
    return chunks


async def process_chunk(
    text_chunk: str,
    chunk_num: int,
    total_chunks: int,
    tex_template: str,
    ai: openai.AsyncOpenAI,
    language: str,
    previous_chunk_result: typing.Optional[str] = None,
):
    with open(pathlib.Path(__file__).parent / "prompts/system_prompt.txt") as prompt_file:
        system_prompt = prompt_file.read()
    system_prompt = system_prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"The template:\n\n{tex_template}"},
        {"role": "user", "content": f"This is chunk {chunk_num}/{total_chunks} of the lecture transcript:\n\n{text_chunk}"},
    ]
    
    if previous_chunk_result is not None:
        last_section_start = max(previous_chunk_result.rfind("\\section"), previous_chunk_result.rfind("\\subsection"))
        if last_section_start >= 0:
            last_section = f"Previous chunk finished with the following:\n\n{previous_chunk_result[last_section_start:]}"
            messages.append({"role": "system", "content": last_section})

    # Use cheaper model for text processing
    response = await ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content


async def process(
    wav_file: bytes, 
    ai: openai.AsyncOpenAI, 
    logger: Logger = Logger(), 
    language: typing.Optional[str] = None,
) -> typing.Tuple[str, pdflatex.PDFLaTeX]:
    if language is not None and language not in ("ru", "en"):
        raise ValueError(f"Unsupported language: {language}. Must be 'ru' or 'en'.")

    # First transcribe the entire audio to text locally
    await logger.partial_result("🎤 <b>Starting transcription...</b>")
    transcript = await transcribe_audio(wav_file, logger)

    if language is None:
        language = await detect_language(wav_file, ai)
        await logger.partial_result(f"<b>Detected language:</b> {language}")

    summary = await make_summary(wav_file, ai, language, logger)

    with open(pathlib.Path(__file__).parent / "prompts/template.tex") as tex_template_file:
        tex_template = tex_template_file.read()
    tex_template = localize_template(tex_template, language)
    tex_template = tex_template.replace("<INSERT TITLE HERE>", summary.title)
    tex_template = tex_template.replace("<INSERT ABSTRACT HERE>", summary.abstract)
    await logger.file("tex_template", tex_template, Logger.FileType.TEX)

    # Split transcript into text chunks instead of audio chunks
    chunks = await split_into_chunks(transcript, logger)
    await logger.progress(2, 2 + len(chunks))

    results = []
    for i, chunk in enumerate(chunks):
        content = await process_chunk(
            chunk, i + 1, len(chunks), tex_template, ai, language, results[-1] if len(results) > 0 else None
        )
        await logger.file(f"chunk_{i + 1}", content, Logger.FileType.TEXT)
        results.append(content)
        await logger.progress(2 + len(results), 2 + len(chunks))
    tex = tex_template.replace("%% <INSERT CONTENT HERE>", "\n".join(results))

    await logger.file("lecture_before_postprocess", tex, Logger.FileType.TEX)

    # Postprocess: check for errors and add references
    tex_postprocessed = None
    try:
        tex_postprocessed = await postprocess_summary(tex, ai, language, logger)
        await logger.file("lecture", tex_postprocessed, Logger.FileType.TEX)
    except Exception as e:
        await logger.partial_result(f"⚠️ <b>Postprocessing failed:</b> {e}\nContinuing with original version...")
        # If postprocessing fails, continue with original tex
        tex_postprocessed = tex
        await logger.file("lecture", tex, Logger.FileType.TEX)

    # Try to create PDF if pdflatex is available
    if shutil.which("pdflatex") is None:
        await logger.partial_result(
            "⚠️ PDF generation skipped: pdflatex is not installed or not found in PATH. Returning only .tex file."
        )
        return tex_postprocessed, None

    try:
        pdf_generator = pdflatex.PDFLaTeX(bytes(tex_postprocessed, encoding="utf-8"), uuid.uuid4())
        pdf = pdf_generator.create_pdf()[0]
        await logger.file("lecture", pdf, Logger.FileType.PDF)
        return tex_postprocessed, pdf
    except Exception as e:
        error_msg = f"Failed to convert LaTeX to PDF: {e}"
        
        # Try to get more detailed error information
        try:
            # Try to compile and capture the log
            import tempfile
            import subprocess
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
                f.write(tex_postprocessed)
                tex_file = f.name
            
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                # Get last 20 lines of stderr/stdout for error context
                error_lines = (result.stdout + result.stderr).split('\n')
                relevant_errors = [l for l in error_lines if 'error' in l.lower() or 'undefined' in l.lower()]
                if relevant_errors:
                    error_msg += f"\n\nLaTeX errors:\n" + "\n".join(relevant_errors[-5:])
            
            # Clean up
            import os
            for ext in ['.tex', '.log', '.aux', '.pdf']:
                try:
                    os.unlink(tex_file.replace('.tex', ext))
                except:
                    pass
        except Exception as debug_error:
            error_msg += f"\n(Could not get detailed error: {debug_error})"
        
        await logger.partial_result(error_msg)
        
        # Return tex even if PDF creation failed
        return tex_postprocessed, None
