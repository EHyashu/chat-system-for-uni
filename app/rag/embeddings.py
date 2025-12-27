from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self._model.encode(list(texts), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype("float32")
