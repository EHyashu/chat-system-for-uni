from dataclasses import dataclass
from typing import List

import numpy as np

from app.config import Settings
from app.rag.advanced_retrieval import calculate_chunk_agreement, maximal_marginal_relevance
from app.rag.embeddings import EmbeddingModel
from app.rag.query_expansion import expand_query
from app.rag.vector_store import FaissVectorStore, StoredChunk


@dataclass
class RetrievedChunk:
    text: str
    document_name: str
    score: float
    rank: int = 0  # Position in retrieval results


class Retriever:
    def __init__(self, settings: Settings, embedder: EmbeddingModel, store: FaissVectorStore) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, use_mmr: bool = True, use_query_expansion: bool = True) -> tuple[List[RetrievedChunk], float]:
        """
        Retrieve relevant chunks with advanced retrieval and confidence scoring.
        
        Args:
            query: User query
            use_mmr: Whether to use Maximal Marginal Relevance
            use_query_expansion: Whether to expand query with variations
        
        Returns:
            Tuple of (retrieved_chunks, confidence_score)
        """
        # Query expansion: generate variations to improve recall
        queries_to_search = [query]
        if use_query_expansion:
            query_variations = expand_query(query)
            queries_to_search = query_variations[:3]  # Use top 3 variations
        
        # Search with all query variations and aggregate results
        all_chunks = []
        all_scores = []
        all_indices = []
        seen_indices = set()
        
        for q in queries_to_search:
            query_emb = self.embedder.embed_query(q)
            
            # Initial retrieval
            initial_k = self.settings.top_k * 3 if use_mmr else self.settings.top_k
            chunks, scores, indices = self.store.search(query_emb, initial_k)
            
            # Add unique results
            for chunk, score, idx in zip(chunks, scores, indices):
                if idx not in seen_indices:
                    all_chunks.append(chunk)
                    all_scores.append(score)
                    all_indices.append(idx)
                    seen_indices.add(idx)
        
        if not all_chunks:
            return [], 0.0
        
        # Sort by score (descending)
        sorted_items = sorted(
            zip(all_chunks, all_scores, all_indices),
            key=lambda x: x[1],
            reverse=True
        )
        chunks = [x[0] for x in sorted_items]
        scores = [x[1] for x in sorted_items]
        indices = [x[2] for x in sorted_items]
        
        # Apply MMR for diversity (on aggregated results)
        if use_mmr and len(chunks) > self.settings.top_k:
            # Get original query embedding for MMR
            query_emb = self.embedder.embed_query(query)
            candidate_embeddings = self.store.get_embeddings_by_indices(indices[:self.settings.top_k * 2])
            selected_indices = maximal_marginal_relevance(
                query_embedding=query_emb[0],
                candidate_embeddings=candidate_embeddings,
                candidate_indices=list(range(len(candidate_embeddings))),
                lambda_mult=0.7,
                k=self.settings.top_k,
            )
            # Reorder
            chunks = [chunks[i] for i in selected_indices]
            scores = [scores[i] for i in selected_indices]
            indices = [indices[i] for i in selected_indices]
        else:
            # Just take top-K
            chunks = chunks[:self.settings.top_k]
            scores = scores[:self.settings.top_k]
            indices = indices[:self.settings.top_k]
        
        # Filter by similarity threshold
        retrieved: List[RetrievedChunk] = []
        valid_scores = []
        for rank, (chunk, score) in enumerate(zip(chunks, scores)):
            if score < self.settings.similarity_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=chunk.text,
                    document_name=chunk.document_name,
                    score=float(score),
                    rank=rank,
                )
            )
            valid_scores.append(score)
        
        if not retrieved:
            return [], 0.0
        
        # Calculate confidence score
        confidence = self._calculate_confidence(retrieved, valid_scores)
        
        return retrieved, confidence
    
    def _calculate_confidence(self, chunks: List[RetrievedChunk], scores: List[float]) -> float:
        """
        Calculate overall confidence based on:
        - Top retrieval score
        - Average score
        - Chunk agreement (consistency)
        """
        if not chunks:
            return 0.0
        
        # Component 1: Top score (best match quality)
        top_score = max(scores)
        
        # Component 2: Average score (overall relevance)
        avg_score = float(np.mean(scores))
        
        # Component 3: Chunk agreement (consistency)
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self.embedder.embed_documents(chunk_texts)
        agreement = calculate_chunk_agreement(chunk_texts, chunk_embeddings)
        
        # Weighted combination
        confidence = 0.5 * top_score + 0.3 * avg_score + 0.2 * agreement
        
        return float(np.clip(confidence, 0.0, 1.0))
