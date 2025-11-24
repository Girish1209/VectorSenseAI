import re
from nltk.stem import PorterStemmer
ps = PorterStemmer()


def extract_best_sentence(context, query):
    """
    Returns:
    - best_sentence: selected answer
    - confidence: score between 0 and 1
    """
    q_words = [w.lower() for w in query.split() if len(w) > 2]

    sentences = re.split(r'[.?\n]', context)

    scored = []

    for s in sentences:
        sent = s.strip()
        if not sent:
            continue

        # Score = number of matched query words
        score = sum(1 for w in q_words if ps.stem(w) in ps.stem(sent.lower()))


        if score > 0:
            scored.append((score, sent))

    if not scored:
        return "Not in PDF", 0.0

    # Sort by highest matched words
    scored.sort(reverse=True, key=lambda x: x[0])

    best_sentence = scored[0][1]
    confidence = scored[0][0] / len(q_words)

    return best_sentence, confidence


def highlight_answer(chunk_text, best_sentence):
    """
    Highlights the best_sentence inside the chunk.
    """

    if best_sentence == "Not in PDF":
        return chunk_text

    escaped = re.escape(best_sentence)

    highlighted = re.sub(
        escaped,
        rf"<mark style='background-color:#ffeb3b; color:black; font-weight:600; padding:2px; border-radius:3px;'>{best_sentence}</mark>",
        chunk_text,
        flags=re.IGNORECASE
    )

    return highlighted
