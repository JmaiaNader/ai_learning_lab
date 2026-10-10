from pathlib import Path
import hashlib
import chromadb
from pypdf import PdfReader
from config.settings import settings

COLLECTION = "learning_lab_documents"

def _client():
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_path)

def _chunks(text, size=900, overlap=150):
    text = " ".join(text.split())
    result, start = [], 0
    while start < len(text):
        if text[start:start+size].strip():
            result.append(text[start:start+size])
        start += max(1, size-overlap)
    return result

def index_uploaded_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    collection = _client().get_or_create_collection(COLLECTION)
    ids, docs, metadata = [], [], []
    for page_no, page in enumerate(reader.pages, 1):
        for part_no, chunk in enumerate(_chunks(page.extract_text() or "")):
            key = hashlib.sha1(f"{Path(pdf_path).name}:{page_no}:{part_no}:{chunk}".encode()).hexdigest()
            ids.append(key); docs.append(chunk)
            metadata.append({"source": Path(pdf_path).name, "page": page_no})
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadata)
    return {"indexed_chunks": len(ids), "pages_read": len(reader.pages), "source": Path(pdf_path).name}

def index_pdf_directory():
    folder = Path(settings.pdf_directory)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(folder.glob("*.pdf"))
    return {"pdf_count": len(pdfs), "results": [index_uploaded_pdf(str(p)) for p in pdfs],
            "message": "Indexing complete." if pdfs else "No PDFs found."}

def retrieve(query, count=4):
    try:
        collection = _client().get_collection(COLLECTION)
        total = collection.count()
        if not total:
            return []
        result = collection.query(query_texts=[query], n_results=min(count, total))
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        return [{"text": d, "source": m.get("source", "?"), "page": m.get("page", "?")} for d, m in zip(docs, metas)]
    except Exception:
        return []
