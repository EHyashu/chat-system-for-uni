from typing import List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class SourceChunk(BaseModel):
    text: str
    document_name: str
    score: float
    rank: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    confidence: float = 0.0
    reasoning: str = ""
