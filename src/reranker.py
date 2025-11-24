from sentence_transformers import CrossEncoder

# load once
def get_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)

# input: query string, list of texts (candidates)
# returns list of (score, text) sorted desc
def rerank(reranker, query, candidates):
    pairs = [[query, t] for t in candidates]
    scores = reranker.predict(pairs)  # smaller models use predict()
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return scored
