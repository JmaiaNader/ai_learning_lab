from pathlib import Path
import hashlib
import re
import chromadb
from pypdf import PdfReader
from config.settings import settings

COLLECTION = "learning_lab_documents"
# Resolve relative paths from the project root, not the shell's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _client():
    db_path = _project_path(settings.chroma_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _chunks(text, size=900, overlap=150):
    text = " ".join((text or "").split())
    result, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            result.append(chunk)
        start += max(1, size - overlap)
    return result


def index_uploaded_pdf(pdf_path):
    source_path = Path(pdf_path).expanduser().resolve()
    reader = PdfReader(str(source_path))
    collection = _client().get_or_create_collection(COLLECTION)
    ids, docs, metadata = [], [], []
    for page_no, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        for part_no, chunk in enumerate(_chunks(page_text)):
            key = hashlib.sha1(
                f"{source_path.name}:{page_no}:{part_no}:{chunk}".encode("utf-8")
            ).hexdigest()
            ids.append(key)
            docs.append(chunk)
            metadata.append({"source": source_path.name, "page": page_no})
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadata)
    return {
        "indexed_chunks": len(ids),
        "pages_read": len(reader.pages),
        "pages_with_text": sum(bool((p.extract_text() or "").strip()) for p in reader.pages),
        "source": source_path.name,
        "collection": COLLECTION,
        "chroma_path": str(_project_path(settings.chroma_path)),
    }


def index_pdf_directory():
    folder = _project_path(settings.pdf_directory)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(folder.glob("*.pdf"))
    results = [index_uploaded_pdf(str(p)) for p in pdfs]
    return {
        "pdf_count": len(pdfs),
        "results": results,
        "collection": COLLECTION,
        "chroma_path": str(_project_path(settings.chroma_path)),
        "message": "Indexing complete." if pdfs else f"No PDF files found in {folder}.",
    }


def _query_variants(query):
    """Create a generic lexical variant without assuming any document subject."""
    original = " ".join((query or "").split()).strip()
    terms = re.findall(r"[\w'-]+", original.lower())
    stop_words = {
        "what", "when", "where", "which", "who", "why", "how", "does", "do",
        "did", "is", "are", "was", "were", "the", "a", "an", "and", "or",
        "in", "on", "at", "to", "of", "for", "from", "with", "can", "could",
        "would", "should", "please", "tell", "me", "about", "explain", "find",
    }
    keywords = [term for term in terms if term not in stop_words]
    variant = " ".join(keywords)
    return list(dict.fromkeys(q for q in (original, variant) if q))


def retrieve(query, count=4):
    """Retrieve relevant chunks from the shared collection for any document domain.

    An empty collection/no matching collection returns an empty list. Unexpected
    ChromaDB failures raise an exception so the caller can distinguish errors from
    a successful search that found no usable passages.
    """
    client = _client()
    collection_names = {item if isinstance(item, str) else item.name for item in client.list_collections()}
    if COLLECTION not in collection_names:
        return []
    collection = client.get_collection(COLLECTION)
    total = collection.count()
    if total == 0:
        return []

    queries = _query_variants(query)
    if not queries:
        return []
    result = collection.query(
        query_texts=queries,
        n_results=min(max(1, count), total),
        include=["documents", "metadatas", "distances"],
    )
    found = {}
    documents_by_query = result.get("documents") or []
    metadatas_by_query = result.get("metadatas") or []
    ids_by_query = result.get("ids") or []
    distances_by_query = result.get("distances") or []
    for qi, docs in enumerate(documents_by_query):
        metas = metadatas_by_query[qi] if qi < len(metadatas_by_query) else []
        ids = ids_by_query[qi] if qi < len(ids_by_query) else []
        distances = distances_by_query[qi] if qi < len(distances_by_query) else []
        for i, doc in enumerate(docs or []):
            if not doc or not doc.strip():
                continue
            meta = metas[i] if i < len(metas) and metas[i] else {}
            doc_id = ids[i] if i < len(ids) else hashlib.sha1(doc.encode("utf-8")).hexdigest()
            distance = distances[i] if i < len(distances) else None
            candidate = {
                "text": doc,
                "source": meta.get("source", "Unknown source"),
                "page": meta.get("page", "Unknown page"),
                "distance": distance,
            }
            previous = found.get(doc_id)
            if previous is None or (distance is not None and (previous["distance"] is None or distance < previous["distance"])):
                found[doc_id] = candidate

    # Chroma distance is lower-is-better. Merge query variants, then return the best.
    ranked = sorted(found.values(), key=lambda item: item["distance"] if item["distance"] is not None else float("inf"))
    return ranked[:count]
