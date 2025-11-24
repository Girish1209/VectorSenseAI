import faiss
import numpy as np

def build_faiss_index(vectors):
    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]
    
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16

    
    print(f"[INFO] FAISS index created with {index.ntotal} vectors")
    return index
