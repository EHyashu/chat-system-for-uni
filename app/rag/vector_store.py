import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


@dataclass
class StoredChunk:
    text: str
    document_name: str


class FaissVectorStore:
    def __init__(self, index: faiss.Index, id_to_chunk: Dict[int, StoredChunk]) -> None:
        self.index = index
        self.id_to_chunk = id_to_chunk

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        texts: List[str],
        sources: List[str],
    ) -> "FaissVectorStore":
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        id_to_chunk: Dict[int, StoredChunk] = {}
        for i, (text, src) in enumerate(zip(texts, sources)):
            id_to_chunk[i] = StoredChunk(text=text, document_name=src)
        return cls(index=index, id_to_chunk=id_to_chunk)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with (index_dir / "metadata.pkl").open("wb") as f:
            pickle.dump(self.id_to_chunk, f)

    @classmethod
    def load(cls, index_dir: Path) -> "FaissVectorStore":
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.pkl"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("FAISS index or metadata not found. Please run the ingestion step first.")

        index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            id_to_chunk: Dict[int, StoredChunk] = pickle.load(f)
        return cls(index=index, id_to_chunk=id_to_chunk)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[List[StoredChunk], List[float], List[int]]:
        """Search and return chunks, scores, and indices."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        scores_list: List[float] = scores[0].tolist()
        chunks: List[StoredChunk] = []
        valid_indices: List[int] = []
        for idx in indices[0]:
            if int(idx) == -1:
                continue
            chunks.append(self.id_to_chunk[int(idx)])
            valid_indices.append(int(idx))
        return chunks, scores_list, valid_indices

    def get_embeddings_by_indices(self, indices: List[int]) -> np.ndarray:
        """Retrieve embeddings for given indices."""
        embeddings_list = []
        for idx in indices:
            # FAISS stores vectors internally; we reconstruct them
            vector = self.index.reconstruct(int(idx))
            embeddings_list.append(vector)
        return np.array(embeddings_list, dtype="float32")
