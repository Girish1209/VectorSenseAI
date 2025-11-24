from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import get_embedding_model, embed_chunks
from src.vector_store import build_faiss_index
from src.search import search_faiss
from src.extractor import extract_answer

PDF_PATH = "data/dp.pdf"

if __name__ == "__main__":
    print("\n=== DAY 3 (NO LLM): Minimal RAG ===\n")

    # Load + chunk
    pages = load_pdf(PDF_PATH)
    chunks = chunk_text(pages)

    # Embeddings
    model = get_embedding_model()
    texts, vectors = embed_chunks(model, chunks)

    # Build FAISS index
    index = build_faiss_index(vectors)

    # User input
    query = input("\nAsk question: ")

    # Retrieve top K chunks
    results = search_faiss(query, model, index, texts, k=4)

    # Combine context
    combined_context = "\n".join([r['text'] for r in results])

    # Extract final answer
    answer = extract_answer(combined_context, query)

    print("\n=== FINAL ANSWER ===\n")
    print(answer)

    print("\n=== DAY 3 SUCCESS (NO LLM) ===\n")
