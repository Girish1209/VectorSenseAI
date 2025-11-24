from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", ".", "?", "!", "\n", " ", ""]
)

    chunks = splitter.split_documents(pages)
    print(f"[INFO] Created {len(chunks)} text chunks")
    return chunks
    return [
    {"text": c.page_content, "page": c.metadata["page"], "chunk_id": i}
    for i, c in enumerate(chunks)
           ]

