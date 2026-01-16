from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import answer_question

load_dotenv()

app = FastAPI(
    title="Dr. Ambedkar RAG API",
    description="RAG-based QA system powered by Qdrant + Gemini",
    version="1.0"
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
