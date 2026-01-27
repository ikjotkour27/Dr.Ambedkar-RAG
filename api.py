from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import answer_question
import uuid
import os
import pyttsx3   # simple offline TTS (can replace later with better TTS)

print("API loaded")
load_dotenv()

app = FastAPI(
    title="Dr. Ambedkar RAG API",
    description="RAG-based QA system powered by Qdrant + Gemini",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend to call backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder to store generated audio
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Serve audio files
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


class Query(BaseModel):
    question: str


def text_to_speech(text: str, file_path: str):
    engine = pyttsx3.init()
    engine.save_to_file(text, file_path)
    engine.runAndWait()


@app.post("/ask")
def ask_question(query: Query):
    # 1. Get RAG answer
    answer = answer_question(query.question)

    # 2. Create unique audio file
    audio_filename = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)

    # 3. Convert answer to speech
    text_to_speech(answer, audio_path)

    # 4. Return response
    return {
        "question": query.question,
        "answer": answer,
        "audio_url": f"http://localhost:8000/audio/{audio_filename}"
    }
