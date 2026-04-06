import asyncio
import ipaddress
import io
import json
import os
import re
import ssl
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

import httpx
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from conspectum.logger import Logger
from conspectum.process import process
from conspectum.summary import guess_audio_suffix
from conspectum.summary import is_supported_audio


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "web_template.html")
REMOTE_AUDIO_MAX_BYTES = int(
    os.environ.get("REMOTE_AUDIO_MAX_BYTES", str(512 * 1024 * 1024))
)

VALID_DETAIL_LEVELS = {"brief", "standard", "detailed"}
VALID_LANGUAGES = {"en", "ru"}

TASK_STAGE_CONFIG = {
    "queued": {"label": "В очереди", "start": 2, "end": 4},
    "starting": {"label": "Запуск обработки", "start": 4, "end": 8},
    "prepare_url": {"label": "Подготовка URL", "start": 8, "end": 12},
    "download_audio": {"label": "Скачивание источника", "start": 12, "end": 18},
    "transcribing": {"label": "Распознавание аудио", "start": 18, "end": 38},
    "transcript_ready": {"label": "Транскрипт готов", "start": 38, "end": 42},
    "detect_language": {"label": "Определение языка", "start": 42, "end": 48},
    "summary": {"label": "Сборка конспекта", "start": 48, "end": 60},
    "sections": {"label": "Сборка разделов", "start": 60, "end": 80},
    "postprocess": {"label": "Постобработка", "start": 80, "end": 90},
    "pdf": {"label": "Сборка PDF", "start": 90, "end": 96},
    "pdf_retry": {"label": "Повторная сборка PDF", "start": 94, "end": 98},
    "pdf_problem": {"label": "Проблема с PDF", "start": 92, "end": 96},
    "tex_only": {"label": "Только TEX", "start": 97, "end": 99},
    "done": {"label": "Готово", "start": 100, "end": 100},
    "error": {"label": "Ошибка", "start": 0, "end": 0},
}
DEFAULT_STAGE_CODE = "queued"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(
    title="Conspectum Web",
    description="Beautiful web interface for audio to LaTeX/PDF summary",
)

tasks: Dict[str, dict] = {}
TASK_TTL = timedelta(hours=1)

allow_insecure_ssl = os.environ.get("ALLOW_INSECURE_SSL", "").lower() in {
    "1",
    "true",
    "yes",
}
verify: bool | ssl.SSLContext = True
if allow_insecure_ssl:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    verify = ssl_context

http_client = httpx.AsyncClient(
    verify=verify,
    timeout=300.0,
)

ai = openai.AsyncOpenAI(
    base_url=os.environ.get("AI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ["AI_API_KEY"],
    http_client=http_client,
)


class WebLogger(Logger):
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.last_progress_bucket: Optional[int] = None

        log_id = str(uuid.uuid4())
        log_dir = os.path.join(LOGS_DIR, log_id)
        os.makedirs(log_dir, exist_ok=True)

        super().__init__(log_dir)

        self.messages: List[str] = []
        tasks[self.task_id]["messages"] = self.messages
        tasks[self.task_id]["progress"] = 0
        set_task_stage(self.task_id, "starting", 0, force=True)

    async def partial_result(self, text: str):
        self.messages.append(text)
        tasks[self.task_id]["messages"] = self.messages
        inferred_stage = infer_stage_update_from_message(text)
        if inferred_stage is not None:
            stage_code, stage_progress = inferred_stage
            set_task_stage(self.task_id, stage_code, stage_progress)
        tasks[self.task_id]["updated_at"] = datetime.now(timezone.utc)

    async def progress(self, completed: int, total: int):
        if total <= 0:
            percent = 0
        else:
            percent = round(completed / total * 100)

        stage_code = tasks[self.task_id].get("stage_code") or "sections"
        set_task_stage(self.task_id, stage_code, percent)

        bucket = 100 if percent >= 100 else max(0, min(95, percent)) // 5
        if bucket != self.last_progress_bucket:
            progress_text = f"Progress: {percent}%"
            self.messages.append(progress_text)
            tasks[self.task_id]["messages"] = self.messages
            self.last_progress_bucket = bucket

        tasks[self.task_id]["updated_at"] = datetime.now(timezone.utc)

    async def stage(self, stage: str, progress: Optional[int] = None):
        set_task_stage(self.task_id, stage, progress)


def cleanup_expired_tasks() -> None:
    now = datetime.now(timezone.utc)
    expired_ids = [
        task_id
        for task_id, task in tasks.items()
        if now - task["created_at"] > TASK_TTL
    ]

    for task_id in expired_ids:
        task = tasks.pop(task_id)
        for key in ("tex_path", "pdf_path", "transcript_path"):
            file_path = task.get(key)
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass


def normalize_detail_value(detail: Optional[str]) -> str:
    normalized = (detail or "standard").strip().lower()
    if normalized not in VALID_DETAIL_LEVELS:
        raise HTTPException(
            status_code=400,
            detail="Detail must be one of: brief, standard, detailed.",
        )
    return normalized


def normalize_language_value(language: Optional[str]) -> Optional[str]:
    if language is None or not language.strip():
        return None

    normalized = language.strip().lower()
    if normalized not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail="Language must be en or ru.")
    return normalized


def create_task_state(
    task_id: str,
    detail: str,
    *,
    source_mode: str,
    source_name: Optional[str] = None,
    source_url: Optional[str] = None,
    audio_size_bytes: Optional[int] = None,
) -> dict:
    created_at = datetime.now(timezone.utc)
    return {
        "task_id": task_id,
        "status": "running",
        "messages": [],
        "progress": 0,
        "stage_code": DEFAULT_STAGE_CODE,
        "stage": TASK_STAGE_CONFIG[DEFAULT_STAGE_CODE]["label"],
        "bundle_url": None,
        "tex_url": None,
        "pdf_url": None,
        "transcript_url": None,
        "tex_path": None,
        "pdf_path": None,
        "transcript_path": None,
        "title": None,
        "language": None,
        "detail": detail,
        "error": None,
        "warning": None,
        "source_mode": source_mode,
        "source_name": source_name,
        "source_url": source_url,
        "audio_size_bytes": audio_size_bytes,
        "abstract": None,
        "transcript_preview": None,
        "transcript_words": None,
        "abstract_words": None,
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "duration_seconds": None,
    }


def make_preview(text: Optional[str], limit: int = 1800) -> Optional[str]:
    if text is None:
        return None

    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized

    shortened = normalized[:limit].rsplit(" ", 1)[0].rstrip()
    return (shortened or normalized[:limit]).rstrip() + "..."


def count_words(text: Optional[str]) -> int:
    return len((text or "").split())


def safe_download_name(text: Optional[str], fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip()).strip("-._")
    return normalized[:80] or fallback


def get_stage_config(stage_code: Optional[str]) -> dict:
    return TASK_STAGE_CONFIG.get(stage_code or DEFAULT_STAGE_CODE, TASK_STAGE_CONFIG[DEFAULT_STAGE_CODE])


def map_stage_progress(stage_code: Optional[str], stage_progress: Optional[int] = None) -> int:
    config = get_stage_config(stage_code)
    start = int(config["start"])
    end = int(config["end"])

    if stage_progress is None:
        return start

    safe_progress = max(0, min(100, int(stage_progress)))
    return start + round((end - start) * (safe_progress / 100))


def set_task_stage(
    task_id: str,
    stage_code: str,
    stage_progress: Optional[int] = None,
    *,
    force: bool = False,
) -> None:
    task = tasks.get(task_id)
    if not task:
        return

    current_stage_code = task.get("stage_code") or DEFAULT_STAGE_CODE
    current_config = get_stage_config(current_stage_code)
    new_config = get_stage_config(stage_code)

    if force or stage_code == current_stage_code or int(new_config["start"]) >= int(current_config["start"]):
        task["stage_code"] = stage_code
        task["stage"] = str(new_config["label"])
        mapped_progress = map_stage_progress(stage_code, stage_progress)
    else:
        mapped_progress = map_stage_progress(current_stage_code, stage_progress)

    if task.get("status") == "running":
        mapped_progress = min(mapped_progress, 99)

    if force:
        task["progress"] = mapped_progress
    else:
        task["progress"] = max(int(task.get("progress", 0)), mapped_progress)

    task["updated_at"] = datetime.now(timezone.utc)


def infer_stage_update_from_message(text: str) -> Optional[tuple[str, Optional[int]]]:
    normalized = text.lower()

    stage_progress = [
        ("fetching audio", ("download_audio", 10)),
        ("remote audio downloaded", ("download_audio", 100)),
        ("starting transcription", ("transcribing", 5)),
        ("transcription complete", ("transcript_ready", 100)),
        ("detected language", ("detect_language", 100)),
        ("topic of the lecture", ("summary", 70)),
        ("abstract of the lecture", ("summary", 100)),
        ("starting postprocessing", ("postprocess", 10)),
        ("postprocessing complete", ("postprocess", 100)),
        ("postprocessing validation failed", ("postprocess", 100)),
        ("postprocessing warning", ("postprocess", 100)),
        ("retrying pdf generation with a safer latex cleanup pass", ("pdf_retry", 30)),
        ("retrying pdf generation with xelatex unicode support", ("pdf_retry", 45)),
        ("retrying pdf generation with lualatex unicode support", ("pdf_retry", 55)),
        ("retrying pdf generation with a readable fallback layout", ("pdf_retry", 75)),
        ("retrying pdf generation with an ascii-safe transliteration fallback", ("pdf_retry", 88)),
        ("failed to convert latex to pdf", ("pdf_problem", 45)),
        ("pdf generation skipped", ("tex_only", 100)),
    ]

    for marker, stage_data in stage_progress:
        if marker in normalized:
            return stage_data

    return None


def validate_remote_audio_url(audio_url: str) -> None:
    parsed = urlparse(audio_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Enter a valid direct http:// or https:// link to an audio file.")

    hostname = parsed.hostname
    if not hostname:
        raise RuntimeError("The provided URL is missing a hostname.")

    if hostname.lower() == "localhost":
        raise RuntimeError("Localhost URLs are not allowed for remote audio downloads.")

    try:
        host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        return

    if (
        host_ip.is_private
        or host_ip.is_loopback
        or host_ip.is_link_local
        or host_ip.is_multicast
        or host_ip.is_reserved
        or host_ip.is_unspecified
    ):
        raise RuntimeError(
            "Private and local network URLs are not allowed for remote audio downloads."
        )


def build_remote_audio_name(audio_url: str, content_type: str | None) -> str:
    parsed = urlparse(audio_url)
    raw_name = os.path.basename(parsed.path)
    filename = unquote(raw_name) if raw_name else "remote-audio"

    if not os.path.splitext(filename)[1]:
        filename += guess_audio_suffix(filename=filename, mime_type=content_type)

    return filename


async def download_audio_from_url(
    audio_url: str,
    logger: WebLogger,
) -> tuple[bytes, str, Optional[str]]:
    validate_remote_audio_url(audio_url)

    await logger.stage("download_audio", 0)
    await logger.partial_result("Fetching audio from the provided URL...")

    timeout = httpx.Timeout(900.0, connect=30.0)
    async with http_client.stream(
        "GET",
        audio_url,
        follow_redirects=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

        content_type_header = (
            (response.headers.get("content-type") or "")
            .split(";")[0]
            .strip()
            .lower()
        )
        filename = build_remote_audio_name(str(response.url), content_type_header)

        if not is_supported_audio(filename, content_type_header):
            raise RuntimeError(
                "The URL does not look like a supported audio file. "
                "Use a direct link to .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, or .webm."
            )

        content_length = response.headers.get("content-length")
        expected_bytes = int(content_length) if content_length and content_length.isdigit() else None

        if expected_bytes and expected_bytes > REMOTE_AUDIO_MAX_BYTES:
            max_mb = REMOTE_AUDIO_MAX_BYTES // (1024 * 1024)
            raise RuntimeError(
                f"Remote audio is too large for the web worker limit ({max_mb} MB)."
            )

        total_bytes = 0
        chunks: list[bytes] = []

        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > REMOTE_AUDIO_MAX_BYTES:
                max_mb = REMOTE_AUDIO_MAX_BYTES // (1024 * 1024)
                raise RuntimeError(
                    f"Remote audio is too large for the web worker limit ({max_mb} MB)."
                )
            chunks.append(chunk)
            if expected_bytes:
                await logger.progress(total_bytes, expected_bytes)

    await logger.partial_result(
        f"Remote audio downloaded: {round(total_bytes / (1024 * 1024), 1)} MB"
    )
    return b"".join(chunks), filename, content_type_header or None


def build_task_bundle_bytes(task_id: str) -> bytes:
    task = tasks[task_id]
    archive_bytes = io.BytesIO()

    metadata = {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "stage": task.get("stage"),
        "title": task.get("title"),
        "language": task.get("language"),
        "detail": task.get("detail"),
        "source_mode": task.get("source_mode"),
        "source_name": task.get("source_name"),
        "source_url": task.get("source_url"),
        "audio_size_bytes": task.get("audio_size_bytes"),
        "warning": task.get("warning"),
        "error": task.get("error"),
        "created_at": task.get("created_at").isoformat() if task.get("created_at") else None,
        "updated_at": task.get("updated_at").isoformat() if task.get("updated_at") else None,
        "completed_at": task.get("completed_at").isoformat() if task.get("completed_at") else None,
        "duration_seconds": task.get("duration_seconds"),
        "transcript_words": task.get("transcript_words"),
        "abstract_words": task.get("abstract_words"),
    }

    summary_text = task.get("abstract") or ""
    files_to_attach = [
        ("transcript.txt", task.get("transcript_path")),
        ("result.tex", task.get("tex_path")),
        ("result.pdf", task.get("pdf_path")),
    ]

    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        attached_any = False

        for archive_name, path in files_to_attach:
            if path and os.path.exists(path):
                archive.write(path, arcname=archive_name)
                attached_any = True

        if summary_text:
            archive.writestr("summary.txt", summary_text)
            attached_any = True

        archive.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )

        if not attached_any:
            raise HTTPException(
                status_code=409,
                detail="This task does not have any downloadable artifacts yet.",
            )

    archive_bytes.seek(0)
    return archive_bytes.getvalue()


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(TEMPLATE_PATH, encoding="utf-8", errors="replace") as template_file:
        return template_file.read()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    detail: Optional[str] = Form("standard"),
):
    cleanup_expired_tasks()

    normalized_language = normalize_language_value(language)
    normalized_detail = normalize_detail_value(detail)

    if not is_supported_audio(file.filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported audio formats: .wav, .mp3, .m4a, .ogg, .opus, "
                ".flac, .aac, .mp4, .m4b, .webm"
            ),
        )

    audio_bytes = await file.read()

    task_id = str(uuid.uuid4())
    tasks[task_id] = create_task_state(
        task_id,
        normalized_detail,
        source_mode="file",
        source_name=file.filename,
        audio_size_bytes=len(audio_bytes),
    )

    logger = WebLogger(task_id)

    asyncio.create_task(
        run_processing(
            task_id=task_id,
            audio_bytes=audio_bytes,
            language=normalized_language,
            detail_level=normalized_detail,
            audio_filename=file.filename,
            audio_mime_type=file.content_type,
            source_name=file.filename,
            logger=logger,
        )
    )

    return {"task_id": task_id}


@app.post("/upload-url")
async def upload_url(
    audio_url: str = Form(...),
    language: Optional[str] = Form(None),
    detail: Optional[str] = Form("standard"),
):
    cleanup_expired_tasks()

    normalized_url = audio_url.strip()
    if not normalized_url:
        raise HTTPException(status_code=400, detail="Audio URL is required.")

    normalized_language = normalize_language_value(language)
    normalized_detail = normalize_detail_value(detail)

    try:
        validate_remote_audio_url(normalized_url)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = str(uuid.uuid4())
    tasks[task_id] = create_task_state(
        task_id,
        normalized_detail,
        source_mode="url",
        source_name=normalized_url,
        source_url=normalized_url,
    )

    logger = WebLogger(task_id)

    asyncio.create_task(
        run_processing(
            task_id=task_id,
            audio_bytes=None,
            language=normalized_language,
            detail_level=normalized_detail,
            audio_filename=None,
            audio_mime_type=None,
            source_url=normalized_url,
            source_name=normalized_url,
            logger=logger,
        )
    )

    return {"task_id": task_id}


async def run_processing(
    task_id: str,
    audio_bytes: Optional[bytes],
    language: Optional[str],
    detail_level: str,
    audio_filename: Optional[str],
    audio_mime_type: Optional[str],
    logger: WebLogger,
    source_url: Optional[str] = None,
    source_name: Optional[str] = None,
):
    started_at = tasks.get(task_id, {}).get("created_at", datetime.now(timezone.utc))

    try:
        set_task_stage(task_id, "starting", 20)
        resolved_audio_bytes = audio_bytes
        resolved_audio_filename = audio_filename
        resolved_audio_mime_type = audio_mime_type
        resolved_source_name = source_name or audio_filename or "audio"

        if source_url:
            set_task_stage(task_id, "prepare_url", 30)
            (
                resolved_audio_bytes,
                resolved_audio_filename,
                resolved_audio_mime_type,
            ) = await download_audio_from_url(source_url, logger)
            resolved_source_name = resolved_audio_filename or source_url
            tasks[task_id].update(
                {
                    "source_name": resolved_source_name,
                    "audio_size_bytes": len(resolved_audio_bytes),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        elif resolved_audio_bytes is not None:
            tasks[task_id].update(
                {
                    "source_name": resolved_source_name,
                    "audio_size_bytes": len(resolved_audio_bytes),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        if resolved_audio_bytes is None:
            raise RuntimeError("No audio data was provided.")

        result = await process(
            resolved_audio_bytes,
            ai,
            logger,
            language=language,
            detail_level=detail_level,
            audio_filename=resolved_audio_filename,
            audio_mime_type=resolved_audio_mime_type,
        )

        transcript_filename = f"transcript_{uuid.uuid4()}.txt"
        transcript_path = os.path.join(STATIC_DIR, transcript_filename)

        with open(transcript_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(result.transcript)

        tex_filename = f"result_{uuid.uuid4()}.tex"
        tex_path = os.path.join(STATIC_DIR, tex_filename)

        with open(tex_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(result.tex)

        pdf_filename = None
        pdf_path = None

        if result.pdf:
            pdf_filename = f"result_{uuid.uuid4()}.pdf"
            pdf_path = os.path.join(STATIC_DIR, pdf_filename)

            with open(pdf_path, "wb") as file_handle:
                file_handle.write(result.pdf)

        warning_message = result.pdf_warning
        if not pdf_filename and warning_message is None:
            pdf_error = next(
                (
                    message
                    for message in reversed(logger.messages)
                    if message.startswith("Failed to convert LaTeX to PDF:")
                    or message.startswith("PDF generation skipped:")
                ),
                None,
            )
            if pdf_error:
                warning_message = pdf_error
            else:
                warning_message = (
                    "PDF was not generated. The TEX file was created successfully, "
                    "but the exact reason was not captured."
                )

        completed_at = datetime.now(timezone.utc)
        tasks[task_id].update(
            {
                "status": "done",
                "progress": 100,
                "stage_code": "done",
                "stage": TASK_STAGE_CONFIG["done"]["label"],
                "title": result.title,
                "language": result.language,
                "detail": detail_level,
                "bundle_url": f"/bundle/{task_id}",
                "transcript_url": f"/static/{transcript_filename}",
                "tex_url": f"/static/{tex_filename}",
                "pdf_url": f"/static/{pdf_filename}" if pdf_filename else None,
                "transcript_path": transcript_path,
                "tex_path": tex_path,
                "pdf_path": pdf_path if pdf_filename else None,
                "warning": warning_message,
                "source_name": resolved_source_name,
                "audio_size_bytes": len(resolved_audio_bytes),
                "abstract": result.abstract,
                "transcript_preview": make_preview(result.transcript),
                "transcript_words": count_words(result.transcript),
                "abstract_words": count_words(result.abstract),
                "updated_at": completed_at,
                "completed_at": completed_at,
                "duration_seconds": max(
                    0,
                    round((completed_at - started_at).total_seconds()),
                ),
            }
        )
    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        tasks[task_id].update(
            {
                "status": "error",
                "stage_code": "error",
                "stage": TASK_STAGE_CONFIG["error"]["label"],
                "error": str(exc),
                "updated_at": completed_at,
                "completed_at": completed_at,
                "duration_seconds": max(
                    0,
                    round((completed_at - started_at).total_seconds()),
                ),
            }
        )


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks[task_id]


@app.get("/bundle/{task_id}")
async def get_task_bundle(task_id: str):
    cleanup_expired_tasks()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    bundle_bytes = build_task_bundle_bytes(task_id)
    archive_name = safe_download_name(tasks[task_id].get("title"), f"conspectum-{task_id[:8]}")
    filename = f"{archive_name}.zip"

    return StreamingResponse(
        io.BytesIO(bundle_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/static/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
