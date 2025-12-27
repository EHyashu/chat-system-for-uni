from typing import Any, Dict, List

from app.config import Settings
from app.models import SourceChunk
from app.rag.document_loader import load_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.llm import BaseLLM, get_llm
from app.rag.question_classifier import classify_question, get_fallback_response, should_use_retrieval
from app.rag.retriever import Retriever
from app.rag.text_splitter import split_documents
from app.rag.vector_store import FaissVectorStore


SYSTEM_PROMPT = (
    "You are a university helpdesk assistant with expertise in academic policies and procedures.\n"
    "\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Answer questions ONLY using the provided university documents context.\n"
    "2. If the context does not contain sufficient information, you MUST respond exactly: "
    '"I could not find this information in the university documents."\n'
    "3. Use step-by-step reasoning when needed.\n"
    "4. Cite specific document names when providing information.\n"
    "5. Use formal but simple language.\n"
    "6. Prefer bullet points for structured information (rules, criteria, lists).\n"
    "7. Never make assumptions or add information not present in the context.\n"
    "8. If asked a multi-part question, address each part systematically.\n"
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful general-purpose AI assistant. "
    "Answer using your broad knowledge, not limited to the provided documents. "
    "Use clear, formal but simple language, and bullet points for lists. "
    "Do NOT claim to have access to this specific university's internal records. "
    "If the question appears to be about official university rules, policies, or syllabus details, "
    "clearly state that exact policies may vary by institution and recommend consulting official documents."
)


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: Retriever, llm: BaseLLM) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        embedder = EmbeddingModel(settings.embedding_model_name)
        store = FaissVectorStore.load(settings.index_dir)
        retriever = Retriever(settings, embedder, store)
        llm = get_llm(settings)
        return cls(settings=settings, retriever=retriever, llm=llm)

    @staticmethod
    def build_index(settings: Settings) -> None:
        raw_docs = load_documents(settings.documents_dir)
        chunks = split_documents(raw_docs)
        texts = [c.text for c in chunks]
        sources = [c.source for c in chunks]

        embedder = EmbeddingModel(settings.embedding_model_name)
        embeddings = embedder.embed_documents(texts)

        store = FaissVectorStore.from_embeddings(embeddings, texts, sources)
        store.save(settings.index_dir)

    async def answer_query(self, query: str) -> Dict[str, Any]:
        # Step 1: Classify the question type
        question_type = classify_question(query)

        # Step 2: Handle greetings or simple chit-chat
        if question_type == "general_greeting":
            return get_fallback_response(question_type, query)

        # Step 3: General-knowledge mode (no RAG) for non-university questions
        if question_type == "general_question":
            user_prompt = (
                "USER QUESTION:\n"
                f"{query}\n\n"
                "INSTRUCTIONS:\n"
                "1. Answer as a general-purpose assistant using your own knowledge.\n"
                "2. Use formal but simple language.\n"
                "3. Use bullet points for lists.\n"
                "4. If the question appears to ask about a specific university's official rules, "
                "say that exact policies may vary and recommend checking official documents.\n"
            )
            answer = self.llm.generate(GENERAL_SYSTEM_PROMPT, user_prompt)
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.9,
                "reasoning": "General-knowledge question handled without university document retrieval.",
            }

        # Step 4: Retrieve with advanced retrieval and confidence scoring for university questions
        retrieved, confidence = self.retriever.retrieve(query, use_mmr=True, use_query_expansion=True)
        
        # Confidence-based early exit
        if not retrieved or confidence < 0.25:
            return {
                "answer": "I could not find this information in the university documents.",
                "sources": [],
                "confidence": confidence,
                "reasoning": "No sufficiently relevant context found in the document database.",
            }

        # Build context with length limit and structured format
        context_lines: List[str] = []
        total_chars = 0
        seen_docs = set()
        
        for r in retrieved:
            snippet = r.text.replace("\n", " ").strip()
            header = f"[Document: {r.document_name} | Relevance: {r.score:.3f} | Rank: {r.rank+1}]"
            line = f"{header}\n{snippet}"
            if total_chars + len(line) > self.settings.max_context_chars:
                break
            context_lines.append(line)
            total_chars += len(line)
            seen_docs.add(r.document_name)

        context = "\n\n".join(context_lines)

        # Enhanced user prompt with reasoning requirement
        user_prompt = (
            "You are answering a student query using university documents.\n\n"
            "CONTEXT FROM UNIVERSITY DOCUMENTS:\n"
            f"{context}\n\n"
            f"STUDENT QUESTION: {query}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the context carefully.\n"
            "2. If the context contains the answer, provide a clear, structured response.\n"
            "3. Cite the specific document name(s) you used.\n"
            "4. Use bullet points for lists or multi-part answers.\n"
            "5. If the context does NOT contain enough information, respond with EXACTLY: "
            "'I could not find this information in the university documents.'\n"
            "6. Do NOT make assumptions or add external knowledge.\n\n"
            "YOUR ANSWER:"
        )

        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # Build reasoning for transparency
        reasoning_parts = [
            f"Retrieved {len(retrieved)} relevant document chunks.",
            f"Documents consulted: {', '.join(sorted(seen_docs))}.",
            f"Overall confidence: {confidence:.2f} (based on retrieval scores and chunk consistency).",
        ]
        reasoning = " ".join(reasoning_parts)

        sources = [
            SourceChunk(text=r.text, document_name=r.document_name, score=r.score, rank=r.rank).dict()
            for r in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "reasoning": reasoning,
        }
