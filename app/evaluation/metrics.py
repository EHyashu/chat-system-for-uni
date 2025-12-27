"""
Evaluation metrics for RAG system.

This module provides metrics to measure:
1. Retrieval Quality (Precision@K, Recall@K, MRR)
2. Answer Quality (Semantic Similarity, Faithfulness)
3. Hallucination Detection
"""
from typing import Dict, List, Set

import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Precision@K: What fraction of top-K retrieved documents are relevant?
        
        Formula: (# relevant docs in top-K) / K
        
        Args:
            retrieved_docs: List of retrieved document names (in rank order)
            relevant_docs: Set of ground-truth relevant document names
            k: Number of top results to consider
            
        Returns:
            Precision score [0, 1]
        """
        if k == 0 or not retrieved_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Recall@K: What fraction of all relevant documents are in top-K?
        
        Formula: (# relevant docs in top-K) / (total # relevant docs)
        
        Args:
            retrieved_docs: List of retrieved document names (in rank order)
            relevant_docs: Set of ground-truth relevant document names
            k: Number of top results to consider
            
        Returns:
            Recall score [0, 1]
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_count / len(relevant_docs)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs_list: List[List[str]], relevant_docs_list: List[Set[str]]) -> float:
        """
        MRR: Average of reciprocal ranks of first relevant document.
        
        MRR measures how quickly the system finds the first relevant result.
        Higher is better (1.0 = first result always relevant).
        
        Args:
            retrieved_docs_list: List of retrieval results for multiple queries
            relevant_docs_list: List of relevant doc sets for each query
            
        Returns:
            MRR score [0, 1]
        """
        if not retrieved_docs_list:
            return 0.0
        
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            for rank, doc in enumerate(retrieved, start=1):
                if doc in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return float(np.mean(reciprocal_ranks))
    
    @staticmethod
    def f1_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K.
        
        Args:
            retrieved_docs: List of retrieved document names
            relevant_docs: Set of relevant document names
            k: Number of top results
            
        Returns:
            F1 score [0, 1]
        """
        precision = RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


class AnswerQualityMetrics:
    """Metrics for evaluating generated answer quality."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def semantic_similarity(self, generated_answer: str, reference_answer: str) -> float:
        """
        Semantic Similarity: Cosine similarity between generated and reference answer embeddings.
        
        Measures how close the generated answer is to the expected answer in meaning.
        
        Args:
            generated_answer: Answer from the RAG system
            reference_answer: Ground-truth expected answer
            
        Returns:
            Similarity score [0, 1]
        """
        if not generated_answer.strip() or not reference_answer.strip():
            return 0.0
        
        embeddings = self.model.encode([generated_answer, reference_answer], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))
    
    def faithfulness_score(self, answer: str, context_chunks: List[str]) -> float:
        """
        Faithfulness: How well is the answer grounded in the retrieved context?
        
        Method: Check if answer embedding is similar to at least one context chunk.
        
        Args:
            answer: Generated answer
            context_chunks: Retrieved context chunks
            
        Returns:
            Faithfulness score [0, 1]
        """
        if not answer.strip() or not context_chunks:
            return 0.0
        
        # Embed answer and all context chunks
        texts = [answer] + context_chunks
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        answer_emb = embeddings[0]
        context_embs = embeddings[1:]
        
        # Max similarity to any context chunk
        similarities = np.dot(context_embs, answer_emb)
        max_similarity = float(np.max(similarities))
        
        return max(0.0, min(1.0, max_similarity))
    
    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance: How relevant is the answer to the original query?
        
        Args:
            query: User question
            answer: Generated answer
            
        Returns:
            Relevance score [0, 1]
        """
        if not query.strip() or not answer.strip():
            return 0.0
        
        embeddings = self.model.encode([query, answer], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))
    
    def exact_match(self, generated_answer: str, reference_answer: str, normalize: bool = True) -> float:
        """
        Exact Match: Binary score for exact string match (after normalization).
        
        Args:
            generated_answer: Generated answer
            reference_answer: Reference answer
            normalize: Whether to normalize (lowercase, strip) before comparing
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if normalize:
            generated = generated_answer.lower().strip()
            reference = reference_answer.lower().strip()
        else:
            generated = generated_answer
            reference = reference_answer
        
        return 1.0 if generated == reference else 0.0


class HallucinationDetector:
    """Detect potential hallucinations in generated answers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def detect_hallucination(
        self,
        answer: str,
        context_chunks: List[str],
        threshold: float = 0.3,
    ) -> Dict[str, any]:
        """
        Detect if answer contains information not supported by context.
        
        Method:
        1. Split answer into sentences
        2. Check each sentence's similarity to context
        3. Flag sentences with low similarity as potential hallucinations
        
        Args:
            answer: Generated answer
            context_chunks: Retrieved context
            threshold: Minimum similarity to be considered grounded
            
        Returns:
            Dict with hallucination analysis
        """
        if not answer.strip() or not context_chunks:
            return {
                "is_hallucinated": True,
                "hallucination_score": 1.0,
                "unsupported_sentences": [],
            }
        
        # Split answer into sentences (simple split by period)
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        
        if not sentences:
            return {
                "is_hallucinated": False,
                "hallucination_score": 0.0,
                "unsupported_sentences": [],
            }
        
        # Embed sentences and context
        all_texts = sentences + context_chunks
        embeddings = self.model.encode(all_texts, normalize_embeddings=True)
        
        sentence_embs = embeddings[:len(sentences)]
        context_embs = embeddings[len(sentences):]
        
        # Check each sentence's grounding
        unsupported_sentences = []
        sentence_scores = []
        
        for i, sent_emb in enumerate(sentence_embs):
            # Max similarity to any context chunk
            similarities = np.dot(context_embs, sent_emb)
            max_sim = float(np.max(similarities))
            sentence_scores.append(max_sim)
            
            if max_sim < threshold:
                unsupported_sentences.append({
                    "text": sentences[i],
                    "similarity": max_sim,
                })
        
        hallucination_score = 1.0 - float(np.mean(sentence_scores))
        is_hallucinated = len(unsupported_sentences) > 0
        
        return {
            "is_hallucinated": is_hallucinated,
            "hallucination_score": hallucination_score,
            "unsupported_sentences": unsupported_sentences,
            "sentence_grounding_scores": sentence_scores,
        }


def calculate_aggregate_score(
    retrieval_precision: float,
    retrieval_recall: float,
    semantic_similarity: float,
    faithfulness: float,
    hallucination_score: float,
) -> float:
    """
    Calculate overall RAG system quality score.
    
    Combines retrieval and generation metrics into single score.
    
    Args:
        retrieval_precision: Precision@K
        retrieval_recall: Recall@K
        semantic_similarity: Answer semantic similarity
        faithfulness: Faithfulness to context
        hallucination_score: Hallucination penalty (lower is better)
        
    Returns:
        Aggregate quality score [0, 1]
    """
    retrieval_f1 = (
        2 * (retrieval_precision * retrieval_recall) / (retrieval_precision + retrieval_recall)
        if (retrieval_precision + retrieval_recall) > 0
        else 0.0
    )
    
    # Weighted combination
    score = (
        0.25 * retrieval_f1 +
        0.25 * semantic_similarity +
        0.30 * faithfulness +
        0.20 * (1.0 - hallucination_score)  # Penalize hallucinations
    )
    
    return float(np.clip(score, 0.0, 1.0))
