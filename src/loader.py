from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"[INFO] Loaded {len(pages)} pages from {path}")
    return pages
