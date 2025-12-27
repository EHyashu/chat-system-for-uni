"""
Advanced retrieval utilities including MMR and re-ranking.
"""
from typing import List, Tuple

import numpy as np


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_indices: List[int],
    lambda_mult: float = 0.5,
    k: int = 5,
) -> List[int]:
    """
    Select diverse results using Maximal Marginal Relevance.
    
    Args:
        query_embedding: Query vector (1D)
        candidate_embeddings: All candidate vectors (2D: n_candidates x dim)
        candidate_indices: Original indices of candidates
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        k: Number of results to return
        
    Returns:
        Indices of selected diverse candidates
    """
    if len(candidate_indices) == 0:
        return []
    
    if len(candidate_indices) <= k:
        return candidate_indices
    
    # Normalize embeddings
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    candidate_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute similarity to query
    query_similarity = np.dot(candidate_norms, query_norm)
    
    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidate_indices)))
    
    # Select first item (most similar to query)
    best_idx = int(np.argmax(query_similarity))
    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)
    
    # Iteratively select diverse items
    while len(selected_indices) < k and remaining_indices:
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance to query
            relevance = query_similarity[idx]
            
            # Diversity: minimum similarity to already selected
            selected_embeddings = candidate_norms[selected_indices]
            similarities_to_selected = np.dot(selected_embeddings, candidate_norms[idx])
            max_similarity = np.max(similarities_to_selected)
            diversity = 1.0 - max_similarity
            
            # MMR score
            mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    # Map back to original indices
    return [candidate_indices[i] for i in selected_indices]


def calculate_chunk_agreement(chunks: List[str], embeddings: np.ndarray) -> float:
    """
    Calculate how similar retrieved chunks are to each other.
    High agreement = consistent information.
    
    Args:
        chunks: List of text chunks
        embeddings: Embeddings of chunks (2D array)
        
    Returns:
        Agreement score (0-1), higher = more consistent
    """
    if len(chunks) <= 1:
        return 1.0
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    
    # Compute pairwise similarities
    similarity_matrix = np.dot(normalized, normalized.T)
    
    # Get upper triangle (exclude diagonal)
    n = len(chunks)
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(similarity_matrix[i, j])
    
    if not upper_triangle:
        return 1.0
    
    # Average similarity
    return float(np.mean(upper_triangle))
