from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import get_embedding_model, embed_chunks
from src.vector_store import build_faiss_index
from src.search import search_faiss

PDF_PATH = "data/dp.pdf"

if __name__ == "__main__":
    print("\n=== DAY 2: Query → Embedding → FAISS Search ===\n")

    # Step 1: Load and chunk PDF
    pages = load_pdf(PDF_PATH)
    chunks = chunk_text(pages)

    # Step 2: Embeddings
    model = get_embedding_model()
    texts, vectors = embed_chunks(model, chunks)

    # Step 3: Build the FAISS index
    index = build_faiss_index(vectors)

    # Step 4: Ask user query
    query = input("\nEnter your question: ")

    # Step 5: Search similar chunks
    results = search_faiss(query, model, index, texts, k=3)

    print("\n🔍 Top Matches:\n")
    for r in results:
        print(f"Rank: {r['rank']}")
        print(f"Distance: {r['distance']:.4f}")
        print(f"Text:\n{r['text']}\n")
        print("-" * 50)

    print("\n🎉 DAY 2 COMPLETED — Retrieval working!\n")
