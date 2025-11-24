import streamlit as st
import os
import pickle
import faiss
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import get_embedding_model, embed_chunks
from src.vector_store import build_faiss_index
from src.search import search_faiss
from src.bm25 import build_bm25, bm25_search
from src.extractor import extract_best_sentence, highlight_answer

# -------------------------------------------------------
#   PAGE CONFIG + GLOBAL CSS
# -------------------------------------------------------
st.set_page_config(page_title="VectorSense AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(to bottom right, #f8fbff, #e6f4ff);
}
.topbar {
    background:#0047ab;
    color:white;
    padding:12px;
    font-size:22px;
    font-weight:700;
    border-radius:6px;
    text-align:center;
}
.chunk-box {
    padding:12px;
    background:#e3f2fd;
    border-radius:6px;
    border:1px solid #90caf9;
    color:#000;
}
.answer-box {
    padding:14px;
    font-size:18px;
    background:#d0ffc8;
    border-left:6px solid #007f0e;
    border-radius:5px;
    color:#000;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="topbar">⚡ VectorSense AI — Smart PDF Q&A (Hybrid RAG)</div>', unsafe_allow_html=True)
st.write(" ")

# -------------------------------------------------------
#   SIDEBAR SETTINGS
# -------------------------------------------------------
st.sidebar.header("📘 About")
st.sidebar.write("VectorSense AI is an offline hybrid RAG engine (FAISS + BM25) with exact answer extraction.")

st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 4)

# -------------------------------------------------------
#   PDF UPLOAD (MULTI-PDF)
# -------------------------------------------------------
st.subheader("📂 Upload Your PDFs")
pdf_files = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True)

if pdf_files:

    pages = []

    # Save & load all PDFs
    for pf in pdf_files:
        with open(pf.name, "wb") as f:
            f.write(pf.read())
        pages.extend(load_pdf(pf.name))

    st.success(f"{len(pdf_files)} PDF(s) uploaded!")

    # -------------------------------------------------------
    #   LOAD EMBEDDING MODEL FIRST (Fast!)
    # -------------------------------------------------------
    embed_model = get_embedding_model()   # 💥 FIXED (MUST be outside cache!)

    # -------------------------------------------------------
    #   PROCESS PDF + CACHING
    # -------------------------------------------------------
    CACHE_TEXTS = "cache_texts.pkl"
    CACHE_VECTORS = "cache_vectors.pkl"
    CACHE_FAISS = "cache_faiss.index"

    with st.spinner("Processing PDF(s)..."):

        chunks = chunk_text(pages)

        # ---- USE CACHE IF AVAILABLE ----
        if (
            os.path.exists(CACHE_TEXTS)
            and os.path.exists(CACHE_VECTORS)
            and os.path.exists(CACHE_FAISS)
        ):
            st.success("Loaded cached embeddings and FAISS index ⚡")

            texts = pickle.load(open(CACHE_TEXTS, "rb"))
            vectors = pickle.load(open(CACHE_VECTORS, "rb"))
            faiss_index = faiss.read_index(CACHE_FAISS)

            bm25 = build_bm25(texts)

        else:
            st.warning("No cache found — embedding PDF(s)...")

            # Compute embeddings ONCE
            texts, vectors = embed_chunks(embed_model, chunks)

            # Create FAISS index
            faiss_index = build_faiss_index(vectors)

            # BM25
            bm25 = build_bm25(texts)

            # Save Cache
            pickle.dump(texts, open(CACHE_TEXTS, "wb"))
            pickle.dump(vectors, open(CACHE_VECTORS, "wb"))
            faiss.write_index(faiss_index, CACHE_FAISS)

            st.success("Cache saved for faster future loads! 🚀")

    # Sidebar Stats
    st.sidebar.header("📄 PDF Stats")
    st.sidebar.write(f"Total Pages: **{len(pages)}**")
    st.sidebar.write(f"Total Chunks: **{len(chunks)}**")
    st.sidebar.write(f"Total Embeddings: **{len(vectors)}**")

    # -------------------------------------------------------
    #   QUESTION SECTION
    # -------------------------------------------------------
    st.subheader("🤖 Ask a Question")
    query = st.text_input("What do you want to know?")

    if query:

        with st.spinner("Searching document..."):

            # ----- FAISS -----
            faiss_results = search_faiss(query, embed_model, faiss_index, texts, k=top_k)

            # ----- BM25 -----
            bm25_results = bm25_search(query, bm25, texts, k=top_k)
            bm25_formatted = [
                {"rank": r["rank"], "text": r["text"], "distance": -r["score"]}
                for r in bm25_results
            ]

            # Weighted hybrid scoring
            for r in faiss_results:
                r["final_score"] = (1 / (1 + r["distance"])) * 0.7

            for r in bm25_formatted:
                r["final_score"] = (1 / (1 + abs(r["distance"]))) * 0.3

            combined = sorted(
                faiss_results + bm25_formatted,
                key=lambda x: x["final_score"],
                reverse=True,
            )

            # Remove duplicates
            seen = set()
            final_results = []
            for r in combined:
                if r["text"] not in seen:
                    final_results.append(r)
                    seen.add(r["text"])

            final_results = final_results[:4]
            context = "\n".join([r["text"] for r in final_results])

            best_sentence, confidence = extract_best_sentence(context, query)

        # -------------------------------------------------------
        #   DISPLAY FINAL ANSWER
        # -------------------------------------------------------
        st.markdown("### ✅ Extracted Answer")
        st.markdown(f'<div class="answer-box">{best_sentence}</div>', unsafe_allow_html=True)

        st.markdown("### 🔢 Confidence Level")
        st.progress(float(min(confidence, 1.0)))

        # -------------------------------------------------------
        #   SHOW RETRIEVED CHUNKS
        # -------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔍 Retrieved Chunks (Highlighted)")

        for r in final_results:
            highlighted = highlight_answer(r["text"], best_sentence)

            with st.expander(f"Chunk Rank {r['rank']} — Score {r['distance']:.4f}"):
                st.markdown(f'<div class="chunk-box">{highlighted}</div>', unsafe_allow_html=True)
