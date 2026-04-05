import asyncio
import os
import ssl
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from conspectum.logger import Logger
from conspectum.process import process
from conspectum.summary import is_supported_audio


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(
    title="Conspectum Web",
    description="Beautiful web interface for audio to LaTeX/PDF summary",
)

tasks: Dict[str, dict] = {}
TASK_TTL = timedelta(hours=1)

allow_insecure_ssl = os.environ.get("ALLOW_INSECURE_SSL", "").lower() in {"1", "true", "yes"}
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

        log_id = str(uuid.uuid4())
        log_dir = os.path.join(LOGS_DIR, log_id)
        os.makedirs(log_dir, exist_ok=True)

        super().__init__(log_dir)

        self.messages: List[str] = []
        tasks[self.task_id]["messages"] = self.messages
        tasks[self.task_id]["progress"] = 0

    async def partial_result(self, text: str):
        self.messages.append(text)
        tasks[self.task_id]["messages"] = self.messages

    async def progress(self, completed: int, total: int):
        if total <= 0:
            percent = 0
        else:
            percent = round(completed / total * 100)

        progress_text = f"Progress: {percent}%"
        self.messages.append(progress_text)
        tasks[self.task_id]["messages"] = self.messages
        tasks[self.task_id]["progress"] = percent


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


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conspectum Web</title>
    <style>
        :root {
            --bg-1: #0f172a;
            --bg-2: #111827;
            --card: rgba(17, 24, 39, 0.78);
            --card-border: rgba(255, 255, 255, 0.08);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #60a5fa;
            --accent-2: #a78bfa;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --white: #ffffff;
            --shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Trebuchet MS", "Segoe UI Variable Text", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(251, 191, 36, 0.12), transparent 28%),
                linear-gradient(135deg, #07111f, #101826 46%, #172033);
            position: relative;
            overflow-x: hidden;
        }

        body::before,
        body::after {
            content: "";
            position: fixed;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            filter: blur(70px);
            opacity: 0.22;
            pointer-events: none;
            z-index: 0;
        }

        body::before {
            top: -90px;
            left: -60px;
            background: #38bdf8;
        }

        body::after {
            right: -90px;
            bottom: 10%;
            background: #f59e0b;
        }

        .page {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 980px;
            margin: 0 auto;
            padding: 40px 20px 60px;
        }

        .hero {
            text-align: center;
            margin-bottom: 28px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--card-border);
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 18px;
            backdrop-filter: blur(10px);
        }

        h1 {
            margin: 0 0 10px;
            font-size: clamp(32px, 5vw, 54px);
            line-height: 1.05;
            color: var(--white);
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: 0.02em;
        }

        .subtitle {
            margin: 0 auto;
            max-width: 700px;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.6;
        }

        .hero-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 20px;
        }

        .hero-tile {
            padding: 14px 16px;
            border-radius: 18px;
            border: 1px solid var(--card-border);
            background: rgba(255, 255, 255, 0.05);
            text-align: left;
        }

        .hero-tile strong {
            display: block;
            color: var(--white);
            margin-bottom: 4px;
            font-size: 14px;
        }

        .hero-tile span {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }

        .upload-card {
            padding: 24px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 18px;
            margin-top: 26px;
        }

        .dropzone {
            position: relative;
            border: 1.5px dashed rgba(255, 255, 255, 0.18);
            border-radius: 22px;
            padding: 26px;
            background: rgba(255, 255, 255, 0.03);
            transition: 0.2s ease;
        }

        .dropzone:hover {
            border-color: rgba(96, 165, 250, 0.55);
            background: rgba(255, 255, 255, 0.05);
        }

        .dropzone.dragging {
            border-color: rgba(251, 191, 36, 0.9);
            background: rgba(251, 191, 36, 0.08);
            box-shadow: inset 0 0 0 1px rgba(251, 191, 36, 0.22);
        }

        .dropzone h2 {
            margin: 0 0 8px;
            font-size: 22px;
            color: var(--white);
        }

        .dropzone p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .file-row {
            margin-top: 18px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }

        input[type="file"] {
            display: none;
        }

        .file-label,
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            padding: 0 18px;
            border-radius: 14px;
            border: 1px solid transparent;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s ease;
            text-decoration: none;
        }

        .file-label {
            color: var(--white);
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--card-border);
        }

        .file-label:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        .btn {
            color: var(--white);
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            box-shadow: 0 10px 24px rgba(96, 165, 250, 0.24);
        }

        .btn:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
        }

        .btn:disabled {
            cursor: not-allowed;
            opacity: 0.65;
            transform: none;
            filter: none;
        }

        .side-panel {
            padding: 22px;
        }

        .panel-title {
            margin: 0 0 12px;
            font-size: 18px;
            color: var(--white);
        }

        .field {
            margin-bottom: 16px;
        }

        .field label {
            display: block;
            margin-bottom: 8px;
            font-size: 14px;
            color: var(--muted);
        }

        select {
            width: 100%;
            min-height: 48px;
            padding: 0 14px;
            border-radius: 14px;
            border: 1px solid var(--card-border);
            background: rgba(255, 255, 255, 0.05);
            color: var(--white);
            font-size: 15px;
            outline: none;
        }

        select option {
            color: #111827;
        }

        .hint {
            margin-top: 10px;
            font-size: 13px;
            color: var(--muted);
            line-height: 1.5;
        }

        .meta {
            margin-top: 12px;
            padding: 14px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--muted);
            font-size: 14px;
            line-height: 1.6;
        }

        .dashboard {
            display: none;
            margin-top: 22px;
            gap: 18px;
        }

        .status-card,
        .logs-card,
        .result-card {
            padding: 22px;
        }

        .status-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 14px;
        }

        .status-title {
            margin: 0;
            font-size: 20px;
            color: var(--white);
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 34px;
            padding: 0 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .chip.running {
            color: #bfdbfe;
            background: rgba(96, 165, 250, 0.12);
            border-color: rgba(96, 165, 250, 0.22);
        }

        .chip.done {
            color: #a7f3d0;
            background: rgba(52, 211, 153, 0.12);
            border-color: rgba(52, 211, 153, 0.22);
        }

        .chip.error {
            color: #fecaca;
            background: rgba(248, 113, 113, 0.12);
            border-color: rgba(248, 113, 113, 0.22);
        }

        .progress-wrap {
            margin-top: 8px;
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }

        .meta-pill {
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
        }

        .meta-pill span {
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 4px;
        }

        .meta-pill strong {
            color: var(--white);
            font-size: 14px;
        }

        .progress-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
            color: var(--muted);
            font-size: 14px;
        }

        .progress-bar {
            width: 100%;
            height: 14px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--card-border);
        }

        .progress-fill {
            width: 0%;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            transition: width 0.25s ease;
        }

        .file-name {
            margin-top: 10px;
            color: var(--text);
            font-size: 14px;
        }

        .logs-card h3,
        .result-card h3 {
            margin: 0 0 14px;
            font-size: 18px;
            color: var(--white);
        }

        #logsContainer {
            max-height: 320px;
            overflow: auto;
            padding-right: 6px;
        }

        .log-list {
            margin: 0;
            padding-left: 18px;
            color: var(--text);
        }

        .log-list li {
            margin-bottom: 10px;
            line-height: 1.5;
        }

        .empty {
            color: var(--muted);
            line-height: 1.6;
        }

        .result-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }

        .result-summary {
            margin-bottom: 14px;
            padding: 14px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--muted);
            line-height: 1.6;
        }

        .result-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            padding: 0 18px;
            border-radius: 14px;
            text-decoration: none;
            font-weight: 700;
            color: var(--white);
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--card-border);
            transition: 0.2s ease;
        }

        .result-link:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateY(-1px);
        }

        .result-link.primary {
            background: linear-gradient(135deg, var(--success), #10b981);
            border-color: transparent;
            box-shadow: 0 10px 24px rgba(16, 185, 129, 0.24);
        }

        .warning-box,
        .error-box {
            margin-top: 14px;
            padding: 14px 16px;
            border-radius: 16px;
            line-height: 1.55;
            font-size: 14px;
        }

        .warning-box {
            color: #fde68a;
            background: rgba(251, 191, 36, 0.12);
            border: 1px solid rgba(251, 191, 36, 0.22);
        }

        .error-box {
            color: #fecaca;
            background: rgba(248, 113, 113, 0.12);
            border: 1px solid rgba(248, 113, 113, 0.22);
        }

        .footer-note {
            margin-top: 14px;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
        }

        @media (max-width: 840px) {
            .grid {
                grid-template-columns: 1fr;
            }

             .hero-strip,
             .meta-grid {
                grid-template-columns: 1fr;
            }

            .page {
                padding-top: 24px;
            }

            .upload-card,
            .side-panel,
            .status-card,
            .logs-card,
            .result-card {
                padding: 18px;
            }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="hero">
            <div class="badge">AI-powered lecture summary • LaTeX • PDF</div>
            <h1>Conspectum</h1>
            <p class="subtitle">
                Turn lecture audio into a cleaner study pack with transcript export, selectable depth,
                and LaTeX-first output that is ready for coursework demos.
            </p>
            <div class="hero-strip">
                <div class="hero-tile">
                    <strong>Transcript Export</strong>
                    <span>Download the recognized lecture text as a plain `.txt` file.</span>
                </div>
                <div class="hero-tile">
                    <strong>Detail Presets</strong>
                    <span>Switch between quick, balanced, and deep summaries before processing.</span>
                </div>
                <div class="hero-tile">
                    <strong>Live Debug View</strong>
                    <span>Watch the pipeline logs in real time and keep the TEX file even if PDF fails.</span>
                </div>
            </div>
        </div>

        <div class="card upload-card">
            <div class="grid">
                <div class="dropzone" id="dropzone">
                    <h2>Upload lecture audio</h2>
                    <p>
                        The web version accepts <strong>.wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, and .webm</strong>.
                        After upload, processing starts in the background and you can watch progress live.
                    </p>

                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="file-row">
                            <label class="file-label" for="fileInput">Choose audio file</label>
                            <input type="file" id="fileInput" accept="audio/*,.wav,.mp3,.m4a,.ogg,.oga,.opus,.flac,.aac,.mp4,.webm" required>
                            <button class="btn" id="submitBtn" type="submit">Process audio</button>
                        </div>

                        <div class="file-name" id="fileName">No file selected</div>
                    </form>

                    <div class="meta">
                        <strong>What you get:</strong><br>
                        1. Structured .tex summary<br>
                        2. PDF file if pdflatex is installed correctly<br>
                        3. Live progress updates during processing
                    </div>
                </div>

                <div class="card side-panel">
                    <h3 class="panel-title">Processing settings</h3>

                    <div class="field">
                        <label for="language">Summary language</label>
                        <select id="language">
                            <option value="">Auto-detect</option>
                            <option value="en">English</option>
                            <option value="ru">Russian</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="detail">Summary depth</label>
                        <select id="detail">
                            <option value="standard">Balanced</option>
                            <option value="brief">Quick</option>
                            <option value="detailed">Deep</option>
                        </select>
                    </div>

                    <div class="hint">
                        Auto-detect tries to identify the lecture language automatically.
                        You can also force Russian or English.
                    </div>

                    <div class="footer-note">
                        Tip: if .tex is created but .pdf is missing, check whether <strong>pdflatex</strong>
                        is installed and available in your system PATH.
                    </div>
                </div>
            </div>

            <div class="dashboard" id="dashboard">
                <div class="card status-card">
                    <div class="status-head">
                        <h3 class="status-title">Task status</h3>
                        <div class="chip running" id="statusChip">Running</div>
                    </div>

                    <div class="progress-wrap">
                        <div class="progress-top">
                            <span id="progressText">Waiting to start...</span>
                            <span id="progressPercent">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill"></div>
                        </div>
                    </div>

                    <div class="meta-grid">
                        <div class="meta-pill">
                            <span>Title</span>
                            <strong id="taskTitle">Pending</strong>
                        </div>
                        <div class="meta-pill">
                            <span>Language</span>
                            <strong id="taskLanguage">Pending</strong>
                        </div>
                        <div class="meta-pill">
                            <span>Detail</span>
                            <strong id="taskDetail">Balanced</strong>
                        </div>
                    </div>

                    <div class="warning-box" id="warningBox" style="display: none;"></div>
                    <div class="error-box" id="errorBox" style="display: none;"></div>
                </div>

                <div class="card logs-card">
                    <h3>Processing log</h3>
                    <div id="logsContainer" class="empty">Logs will appear here during processing.</div>
                </div>

                <div class="card result-card">
                    <h3>Result files</h3>
                    <div class="result-summary" id="resultSummary">Task metadata will appear here after upload.</div>
                    <div class="result-actions" id="resultActions">
                        <span class="empty">No files yet.</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const dropzone = document.getElementById('dropzone');
        const uploadForm = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const dashboard = document.getElementById('dashboard');
        const progressText = document.getElementById('progressText');
        const progressPercent = document.getElementById('progressPercent');
        const progressFill = document.getElementById('progressFill');
        const logsContainer = document.getElementById('logsContainer');
        const resultActions = document.getElementById('resultActions');
        const resultSummary = document.getElementById('resultSummary');
        const statusChip = document.getElementById('statusChip');
        const warningBox = document.getElementById('warningBox');
        const errorBox = document.getElementById('errorBox');
        const taskTitle = document.getElementById('taskTitle');
        const taskLanguage = document.getElementById('taskLanguage');
        const taskDetail = document.getElementById('taskDetail');
        let renderedMessagesCount = 0;

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }

        function setStatus(status) {
            statusChip.className = 'chip ' + status;

            if (status === 'running') {
                statusChip.innerText = 'Running';
            } else if (status === 'done') {
                statusChip.innerText = 'Done';
            } else if (status === 'error') {
                statusChip.innerText = 'Error';
            } else {
                statusChip.innerText = status;
            }
        }

        function updateProgress(progress) {
            const safeProgress = Math.max(0, Math.min(100, Number(progress || 0)));
            progressText.innerText = safeProgress >= 100 ? 'Processing complete' : 'Processing...';
            progressPercent.innerText = `${safeProgress}%`;
            progressFill.style.width = `${safeProgress}%`;
        }

        function renderMessages(messages) {
            if (!messages || messages.length === 0) {
                logsContainer.className = 'empty';
                logsContainer.innerHTML = 'No log messages yet.';
                renderedMessagesCount = 0;
                return;
            }

            const shouldStickToBottom =
                logsContainer.scrollHeight - logsContainer.scrollTop - logsContainer.clientHeight < 24;

            logsContainer.className = '';

            let list = logsContainer.querySelector('.log-list');
            if (!list || messages.length < renderedMessagesCount) {
                logsContainer.innerHTML = '<ol class="log-list"></ol>';
                list = logsContainer.querySelector('.log-list');
                renderedMessagesCount = 0;
            }

            messages.slice(renderedMessagesCount).forEach((msg) => {
                const item = document.createElement('li');
                item.innerHTML = escapeHtml(msg);
                list.appendChild(item);
            });

            renderedMessagesCount = messages.length;

            if (shouldStickToBottom || renderedMessagesCount <= 2) {
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }
        }

        function renderMetadata(data) {
            taskTitle.innerText = data.title || 'Pending';
            if (data.language === 'ru') {
                taskLanguage.innerText = 'Russian';
            } else if (data.language === 'en') {
                taskLanguage.innerText = 'English';
            } else {
                taskLanguage.innerText = data.language || 'Pending';
            }

            if (data.detail === 'brief') {
                taskDetail.innerText = 'Quick';
            } else if (data.detail === 'detailed') {
                taskDetail.innerText = 'Deep';
            } else {
                taskDetail.innerText = 'Balanced';
            }

            if (!data.title && !data.language) {
                resultSummary.innerHTML = 'Task metadata will appear here after upload.';
                return;
            }

            resultSummary.innerHTML = `
                <strong>${escapeHtml(data.title || 'Untitled summary')}</strong><br>
                Language: ${escapeHtml(taskLanguage.innerText)}<br>
                Detail: ${escapeHtml(taskDetail.innerText)}
            `;
        }

        function renderResults(data) {
            const actions = [];

            if (data.transcript_url) {
                actions.push(`
                    <a class="result-link" href="${data.transcript_url}" download="transcript.txt">
                        Download Transcript
                    </a>
                `);
            }

            if (data.tex_url) {
                actions.push(`
                    <a class="result-link" href="${data.tex_url}" download="result.tex">
                        Download TEX
                    </a>
                `);
            }

            if (data.pdf_url) {
                actions.push(`
                    <a class="result-link primary" href="${data.pdf_url}" download="result.pdf">
                        Download PDF
                    </a>
                `);
            }

            if (actions.length === 0) {
                resultActions.innerHTML = '<span class="empty">No files yet.</span>';
                return;
            }

            resultActions.innerHTML = actions.join('');
        }

        async function pollTask(taskId) {
            const response = await fetch(`/status/${taskId}`);

            if (!response.ok) {
                throw new Error('Unable to fetch task status');
            }

            const data = await response.json();

            renderMessages(data.messages || []);
            updateProgress(data.progress ?? 0);
            renderMetadata(data);
            renderResults(data);

            if (data.warning) {
                warningBox.style.display = 'block';
                warningBox.innerText = data.warning;
            } else {
                warningBox.style.display = 'none';
                warningBox.innerText = '';
            }

            if (data.status === 'done') {
                setStatus('done');
                progressText.innerText = 'Processing complete';
                submitBtn.disabled = false;
                return;
            }

            if (data.status === 'error') {
                setStatus('error');
                errorBox.style.display = 'block';
                errorBox.innerText = data.error || 'Unknown error';
                submitBtn.disabled = false;
                return;
            }

            setStatus('running');
            setTimeout(() => pollTask(taskId), 1500);
        }

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files[0]) {
                fileName.innerText = `Selected file: ${fileInput.files[0].name}`;
            } else {
                fileName.innerText = 'No file selected';
            }
        });

        ['dragenter', 'dragover'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.add('dragging');
            });
        });

        ['dragleave', 'dragend', 'drop'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.remove('dragging');
            });
        });

        dropzone.addEventListener('drop', (event) => {
            const droppedFiles = event.dataTransfer.files;
            if (!droppedFiles || droppedFiles.length === 0) {
                return;
            }

            const transfer = new DataTransfer();
            transfer.items.add(droppedFiles[0]);
            fileInput.files = transfer.files;
            fileName.innerText = `Selected file: ${droppedFiles[0].name}`;
        });

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const file = fileInput.files[0];
            const language = document.getElementById('language').value;
            const detail = document.getElementById('detail').value;

            if (!file) {
                fileName.innerText = 'Please choose an audio file first';
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            if (language) {
                formData.append('language', language);
            }

            if (detail) {
                formData.append('detail', detail);
            }

            dashboard.style.display = 'grid';
            submitBtn.disabled = true;
            errorBox.style.display = 'none';
            errorBox.innerText = '';
            warningBox.style.display = 'none';
            warningBox.innerText = '';
            renderedMessagesCount = 0;
            taskTitle.innerText = 'Pending';
            taskLanguage.innerText = language || 'Auto';
            taskDetail.innerText = detail === 'brief' ? 'Quick' : detail === 'detailed' ? 'Deep' : 'Balanced';
            resultSummary.innerHTML = 'Upload accepted. Waiting for transcript and summary metadata...';
            resultActions.innerHTML = '<span class="empty">No files yet.</span>';
            logsContainer.className = 'empty';
            logsContainer.innerHTML = 'Upload started...';
            setStatus('running');
            updateProgress(0);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    setStatus('error');
                    errorBox.style.display = 'block';
                    errorBox.innerText = errorText;
                    submitBtn.disabled = false;
                    return;
                }

                const result = await response.json();
                logsContainer.className = 'empty';
                logsContainer.innerHTML = 'Task created. Waiting for processing...';

                pollTask(result.task_id);
            } catch (error) {
                setStatus('error');
                errorBox.style.display = 'block';
                errorBox.innerText = error.message || 'Unknown upload error';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    detail: Optional[str] = Form("standard"),
):
    cleanup_expired_tasks()

    if not is_supported_audio(file.filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="Supported audio formats: .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, .webm",
        )

    audio_bytes = await file.read()

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "messages": [],
        "progress": 0,
        "tex_url": None,
        "pdf_url": None,
        "transcript_url": None,
        "tex_path": None,
        "pdf_path": None,
        "transcript_path": None,
        "title": None,
        "language": None,
        "detail": detail or "standard",
        "error": None,
        "warning": None,
        "created_at": datetime.now(timezone.utc),
    }

    logger = WebLogger(task_id)

    asyncio.create_task(
        run_processing(
            task_id=task_id,
            audio_bytes=audio_bytes,
            language=language if language else None,
            detail_level=detail if detail else "standard",
            audio_filename=file.filename,
            audio_mime_type=file.content_type,
            logger=logger,
        )
    )

    return {"task_id": task_id}


async def run_processing(
    task_id: str,
    audio_bytes: bytes,
    language: Optional[str],
    detail_level: str,
    audio_filename: Optional[str],
    audio_mime_type: Optional[str],
    logger: WebLogger,
):
    try:
        result = await process(
            audio_bytes,
            ai,
            logger,
            language=language,
            detail_level=detail_level,
            audio_filename=audio_filename,
            audio_mime_type=audio_mime_type,
        )

        transcript_filename = f"transcript_{uuid.uuid4()}.txt"
        transcript_path = os.path.join(STATIC_DIR, transcript_filename)

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result.transcript)

        tex_filename = f"result_{uuid.uuid4()}.tex"
        tex_path = os.path.join(STATIC_DIR, tex_filename)

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(result.tex)

        pdf_filename = None
        pdf_path = None

        if result.pdf:
            pdf_filename = f"result_{uuid.uuid4()}.pdf"
            pdf_path = os.path.join(STATIC_DIR, pdf_filename)

            with open(pdf_path, "wb") as f:
                f.write(result.pdf)

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

        tasks[task_id].update(
            {
                "status": "done",
                "progress": 100,
                "title": result.title,
                "language": result.language,
                "detail": detail_level,
                "transcript_url": f"/static/{transcript_filename}",
                "tex_url": f"/static/{tex_filename}",
                "pdf_url": f"/static/{pdf_filename}" if pdf_filename else None,
                "transcript_path": transcript_path,
                "tex_path": tex_path,
                "pdf_path": pdf_path if pdf_filename else None,
                "warning": warning_message,
            }
        )
    except Exception as e:
        tasks[task_id].update(
            {
                "status": "error",
                "error": str(e),
            }
        )


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks[task_id]


@app.get("/static/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
