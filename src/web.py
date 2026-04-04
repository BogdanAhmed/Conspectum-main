import asyncio
import os
import uuid
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import openai
import httpx
import ssl
from dotenv import load_dotenv

from conspectum.logger import Logger
from conspectum.process import process

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Conspectum Web", description="Web interface for audio to PDF summary")

tasks: Dict[str, dict] = {}

# Setup AI client
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

http_client = httpx.AsyncClient(
    verify=ssl_context,
    timeout=300.0
)

ai = openai.AsyncOpenAI(
    base_url=os.environ.get("AI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ["AI_API_KEY"],
    http_client=http_client
)

class WebLogger(Logger):
    def __init__(self, task_id: str):
        self.task_id = task_id
        id = uuid.uuid4()
        os.makedirs(f"logs/{id}", exist_ok=True)
        super().__init__(f"logs/{id}")
        self.messages: List[str] = []
        tasks[self.task_id]["messages"] = self.messages
        tasks[self.task_id]["progress"] = 0

    async def partial_result(self, text: str):
        self.messages.append(text)
        tasks[self.task_id]["messages"] = self.messages

    async def progress(self, completed: int, total: int):
        percent = round(completed / total * 100)
        progress_text = f"Progress: {percent}%"
        self.messages.append(progress_text)
        tasks[self.task_id]["messages"] = self.messages
        tasks[self.task_id]["progress"] = percent

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conspectum Web</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; }
            .progress { margin-top: 20px; }
            .messages { margin-top: 20px; background: #f9f9f9; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Conspectum - Audio to PDF Summary</h1>
        <div class="upload-form">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" accept=".wav" required>
                <br><br>
                <select id="language">
                    <option value="">Auto-detect</option>
                    <option value="en">English</option>
                    <option value="ru">Russian</option>
                </select>
                <br><br>
                <button type="submit">Process Audio</button>
            </form>
        </div>
        <div class="progress" id="progress" style="display: none;">
            <div id="progressText">Processing...</div>
            <div class="progress-bar" style="background: #eee; border-radius: 10px; margin-top: 8px; overflow: hidden; height: 18px;">
                <div id="progressFill" style="width: 0%; height: 100%; background: #4a90e2; transition: width 0.2s ease;"></div>
            </div>
        </div>
        <div class="messages" id="messages" style="display: none;"></div>
        <script>
            const statusUrl = (taskId) => `/status/${taskId}`;

            function renderMessages(messages) {
                return messages.map(msg => `<p>${msg}</p>`).join('');
            }

            function updateProgress(progress) {
                document.getElementById('progressText').innerText = progress >= 0 ? `Processing... ${progress}%` : 'Processing...';
                document.getElementById('progressFill').style.width = `${progress}%`;
            }

            async function pollTask(taskId) {
                const response = await fetch(statusUrl(taskId));
                if (!response.ok) {
                    throw new Error('Unable to fetch task status');
                }
                const data = await response.json();
                document.getElementById('messages').innerHTML = renderMessages(data.messages || []);
                updateProgress(data.progress ?? 0);

                if (data.status === 'done') {
                    document.getElementById('progressText').innerText = 'Processing complete!';
                    if (data.tex_url) {
                        const texLink = document.createElement('a');
                        texLink.href = data.tex_url;
                        texLink.download = 'result.tex';
                        texLink.innerText = 'Download TEX';
                        texLink.style.display = 'block';
                        document.getElementById('messages').appendChild(texLink);
                    }
                    if (data.pdf_url) {
                        const pdfLink = document.createElement('a');
                        pdfLink.href = data.pdf_url;
                        pdfLink.download = 'result.pdf';
                        pdfLink.innerText = 'Download PDF';
                        pdfLink.style.display = 'block';
                        document.getElementById('messages').appendChild(pdfLink);
                    }
                    return;
                }

                if (data.status === 'error') {
                    document.getElementById('progressText').innerText = 'Error: ' + data.error;
                    return;
                }

                setTimeout(() => pollTask(taskId), 1500);
            }

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const language = document.getElementById('language').value;
                const file = fileInput.files[0];

                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);
                if (language) formData.append('language', language);

                document.getElementById('progress').style.display = 'block';
                document.getElementById('messages').style.display = 'block';
                document.getElementById('messages').innerHTML = '<p>Upload started...</p>';
                updateProgress(0);

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    if (!response.ok) {
                        const error = await response.text();
                        document.getElementById('progressText').innerText = 'Error: ' + error;
                        return;
                    }
                    const result = await response.json();
                    document.getElementById('messages').innerHTML = '<p>Task created, waiting for processing...</p>';
                    pollTask(result.task_id);
                } catch (error) {
                    document.getElementById('progressText').innerText = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), language: str = None):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    audio_bytes = await file.read()
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "running",
        "messages": [],
        "progress": 0,
        "tex_url": None,
        "pdf_url": None,
        "error": None,
    }

    logger = WebLogger(task_id)
    asyncio.create_task(run_processing(task_id, audio_bytes, language if language else None, logger))
    return {"task_id": task_id}

async def run_processing(task_id: str, audio_bytes: bytes, language: str, logger: WebLogger):
    try:
        tex, pdf = await process(audio_bytes, ai, logger, language=language)
        tex_filename = f"result_{uuid.uuid4()}.tex"
        pdf_filename = f"result_{uuid.uuid4()}.pdf" if pdf else None
        os.makedirs("static", exist_ok=True)
        with open(f"static/{tex_filename}", "w", encoding="utf-8") as f:
            f.write(tex)
        if pdf:
            with open(f"static/{pdf_filename}", "wb") as f:
                f.write(pdf)
        tasks[task_id].update({
            "status": "done",
            "progress": 100,
            "tex_url": f"/static/{tex_filename}",
            "pdf_url": f"/static/{pdf_filename}" if pdf_filename else None,
        })
    except Exception as e:
        tasks[task_id].update({"status": "error", "error": str(e)})

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