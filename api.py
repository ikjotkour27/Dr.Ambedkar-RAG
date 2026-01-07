from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

from rag import answer_question

app = FastAPI(
    title="Dr. Ambedkar RAG API",
    description="RAG-based QA system powered by FAISS + Gemini",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    answer = answer_question(query.question)
    return {
        "question": query.question,
        "answer": answer
    }
