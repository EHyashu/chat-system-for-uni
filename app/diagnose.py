"""
Diagnostic tool to debug retrieval issues.

Use this to understand why a query fails to retrieve expected documents.
"""
import asyncio
from pathlib import Path

from app.config import get_settings
from app.rag.embeddings import EmbeddingModel
from app.rag.query_expansion import expand_query
from app.rag.vector_store import FaissVectorStore


async def diagnose_query(query: str) -> None:
    """
    Diagnose why a query might be failing.
    
    Args:
        query: The query that's not working
    """
    print("=" * 80)
    print(f"DIAGNOSING QUERY: {query}")
    print("=" * 80)
    
    settings = get_settings()
    
    # Load vector store
    try:
        store = FaissVectorStore.load(settings.index_dir)
        print(f"\n✓ Loaded FAISS index with {len(store.id_to_chunk)} chunks")
    except Exception as e:
        print(f"\n✗ Failed to load index: {e}")
        return
    
    # Initialize embedder
    embedder = EmbeddingModel(settings.embedding_model_name)
    
    # 1. Show query expansions
    print(f"\n📝 QUERY EXPANSIONS:")
    variations = expand_query(query)
    for i, var in enumerate(variations, 1):
        print(f"  {i}. {var}")
    
    # 2. Search with original query
    print(f"\n🔍 SEARCHING WITH ORIGINAL QUERY:")
    print(f"   Threshold: {settings.similarity_threshold}")
    print(f"   Top-K: {settings.top_k}")
    
    query_emb = embedder.embed_query(query)
    chunks, scores, indices = store.search(query_emb, k=10)  # Get top 10
    
    if not chunks:
        print("   ⚠️  NO RESULTS FOUND!")
    else:
        print(f"\n   Found {len(chunks)} results:\n")
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            status = "✓" if score >= settings.similarity_threshold else "✗"
            print(f"   {status} Rank {i}: {chunk.document_name}")
            print(f"      Score: {score:.4f}")
            print(f"      Preview: {chunk.text[:100]}...")
            print()
    
    # 3. Search with expanded queries
    print(f"\n🔍 SEARCHING WITH EXPANDED QUERIES:")
    all_results = {}
    
    for var in variations[:3]:
        if var == query:
            continue
        
        print(f"\n   Query: '{var}'")
        var_emb = embedder.embed_query(var)
        var_chunks, var_scores, var_indices = store.search(var_emb, k=5)
        
        if var_chunks:
            for chunk, score in zip(var_chunks, var_scores):
                doc_name = chunk.document_name
                if doc_name not in all_results or score > all_results[doc_name]:
                    all_results[doc_name] = score
        
        print(f"   Found {len(var_chunks)} results")
    
    # 4. Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Documents found across all variations:")
    if all_results:
        for doc, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
            status = "✓" if score >= settings.similarity_threshold else "✗"
            print(f"     {status} {doc}: {score:.4f}")
    else:
        print("     ⚠️  NO DOCUMENTS FOUND!")
    
    # 5. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not chunks:
        print("   1. ⚠️  No results found at all!")
        print("      → Check if documents are properly ingested")
        print("      → Run: python -m app.ingest")
    elif max(scores) < settings.similarity_threshold:
        print(f"   1. Best match score ({max(scores):.4f}) is below threshold ({settings.similarity_threshold})")
        print("      → Lower the threshold in app/config.py")
        print(f"      → Or set: export UNI_RAG_SIMILARITY_THRESHOLD=0.15")
    
    if len(variations) <= 1:
        print("   2. Query expansion didn't generate variations")
        print("      → Try rephrasing the query")
        print("      → Use synonyms (e.g., 'courses' instead of 'subjects')")
    
    # 6. Check if specific semester documents exist
    if "semester" in query.lower() or "sem" in query.lower():
        print(f"\n📚 SEMESTER-RELATED DOCUMENTS IN INDEX:")
        semester_docs = set()
        for chunk in store.id_to_chunk.values():
            doc_lower = chunk.document_name.lower()
            if any(keyword in doc_lower for keyword in ["semester", "sem", "syllabus", "course", "curriculum"]):
                semester_docs.add(chunk.document_name)
        
        if semester_docs:
            for doc in sorted(semester_docs):
                print(f"   - {doc}")
        else:
            print("   ⚠️  No semester/syllabus documents found in index!")
            print("   → Check data/documents/ folder")
    
    print("\n" + "=" * 80)


async def main():
    """Interactive diagnostic tool."""
    query = input("Enter your query to diagnose: ").strip()
    if not query:
        query = "What are the subjects in 5 semesters?"
    
    await diagnose_query(query)


if __name__ == "__main__":
    asyncio.run(main())
