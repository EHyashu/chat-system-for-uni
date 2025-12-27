from app.config import get_settings
from app.rag.pipeline import RAGPipeline


def main() -> None:
    settings = get_settings()
    print(f"Building FAISS index from documents in: {settings.documents_dir}")
    RAGPipeline.build_index(settings)
    print(f"Index saved to: {settings.index_dir}")


if __name__ == "__main__":
    main()
