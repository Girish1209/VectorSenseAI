
from rank_bm25 import BM25Okapi

def build_bm25(texts):
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25

def bm25_search(query, bm25, texts, k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_results = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    results = []
    for rank, (idx, score) in enumerate(scored_results, start=1):
        results.append({
            "rank": rank,
            "index": idx,
            "score": float(score),
            "text": texts[idx]
        })
    return results
