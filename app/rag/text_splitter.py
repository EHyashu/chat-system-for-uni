from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    text: str
    source: str


def split_text(text: str, source: str, max_chars: int = 800, overlap: int = 100) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(TextChunk(text=chunk_text, source=source))
        if end == length:
            break
        start = end - overlap

    return chunks


def split_documents(raw_docs: List["RawDocument"], max_chars: int = 800, overlap: int = 100) -> List[TextChunk]:
    from app.rag.document_loader import RawDocument

    chunks: List[TextChunk] = []
    for doc in raw_docs:
        chunks.extend(split_text(doc.text, doc.source, max_chars=max_chars, overlap=overlap))
    return chunks
