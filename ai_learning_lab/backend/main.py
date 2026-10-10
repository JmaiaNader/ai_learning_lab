from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.services.chat_service import answer_question
from backend.rag.pdf_indexer import index_pdf_directory, index_uploaded_pdf
from backend.services.lesson_service import TRACKS
from config.settings import settings

app = FastAPI(title="AI Learning Lab")

class ChatRequest(BaseModel):
    message: str
    track: str = "rag"
    provider: str = "ollama"
    model: str | None = None
    use_rag: bool = True

@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}

@app.get("/tracks")
def tracks():
    return TRACKS

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return answer_question(request.message, request.track, request.provider, request.model, request.use_rag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/index")
def index_pdfs():
    try:
        return index_pdf_directory()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    directory = Path(settings.pdf_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(file.filename).name
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        return {"filename": path.name, **index_uploaded_pdf(str(path))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
