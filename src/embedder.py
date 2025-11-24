from sentence_transformers import SentenceTransformer
import numpy as np


def get_embedding_model():
    print("[INFO] Loading FREE embedding model...")
    model = SentenceTransformer("all-MiniLM-L12-v2")
    return model

def embed_chunks(model, chunks):
    texts = [c.page_content for c in chunks]
    print("[INFO] Embedding chunks...")
    vectors = model.encode(texts, show_progress_bar=True)
    return texts, vectors

# from sentence_transformers import SentenceTransformer

def embed_texts(model_name="all-MiniLM-L6-v2", texts=None, batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

