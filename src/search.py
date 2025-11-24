import numpy as np

def search_faiss(query, model, index, texts, k=3):
    # Embed user query
    query_vec = model.encode([query]).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "rank": i + 1,
                "text": texts[idx],
                "distance": float(distances[0][i])
            })
    
    return results
