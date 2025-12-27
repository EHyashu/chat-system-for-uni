from fastapi import FastAPI

from app.config import Settings, get_settings
from app.models import ChatRequest, ChatResponse
from app.rag.pipeline import RAGPipeline


app = FastAPI(title="University RAG Assistant", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    settings: Settings = get_settings()
    app.state.settings = settings
    app.state.rag_pipeline = RAGPipeline.from_settings(settings)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pipeline: RAGPipeline = app.state.rag_pipeline
    result = await pipeline.answer_query(request.query)
    return ChatResponse(**result)
