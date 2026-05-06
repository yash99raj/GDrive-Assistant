import os
import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model — lazy-loaded on first use, cached afterwards.
# Avoids blocking server startup and makes testing without HF possible.
# ---------------------------------------------------------------------------
_embedder: "HuggingFaceEmbeddings | None" = None


def _get_embedder() -> "HuggingFaceEmbeddings":
    global _embedder
    if _embedder is None:
        logger.info("Loading HuggingFace embedding model ...")
        _embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedder


DB_PATH = "local_faiss_index"

# ---------------------------------------------------------------------------
# In-memory cache — avoids deserializing the FAISS index on every /ask call
# ---------------------------------------------------------------------------
_index_cache: "FAISS | None" = None


def invalidate_cache() -> None:
    """Clears the in-memory FAISS cache. Call this after every sync."""
    global _index_cache
    _index_cache = None
    logger.info("FAISS in-memory cache invalidated.")


def _load_index() -> FAISS | None:
    """
    Returns the cached FAISS index, loading it from disk only when necessary.
    Returns None if no index exists yet (i.e., before the first sync).
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    if not os.path.exists(DB_PATH):
        return None
    logger.info("Loading FAISS index from disk ...")
    _index_cache = FAISS.load_local(
        DB_PATH, _get_embedder(), allow_dangerous_deserialization=True
    )
    return _index_cache


def save_chunks_to_index(processed_chunks: List[dict]) -> None:
    """
    Embeds processed text chunks into vectors and persists them to disk.
    Appends to the existing index if one already exists.
    """
    texts = [chunk["text"] for chunk in processed_chunks]
    metadatas = [chunk["metadata"] for chunk in processed_chunks]

    existing = _load_index()

    if existing is not None:
        existing.add_texts(texts=texts, metadatas=metadatas)
        existing.save_local(DB_PATH)
        # Update cache with the freshly extended index
        global _index_cache
        _index_cache = existing
    else:
        new_index = FAISS.from_texts(texts=texts, embedding=_get_embedder(), metadatas=metadatas)
        new_index.save_local(DB_PATH)
        _index_cache = new_index

    logger.info("Saved %d chunks to FAISS index.", len(texts))


def retrieve_top_k(query: str, k: int = 3) -> List[dict]:
    """
    Embeds the user's query, performs a cosine similarity search in FAISS,
    and returns the top-k most relevant text chunks with their metadata.
    """
    index = _load_index()
    if index is None:
        logger.warning("FAISS index not found — run /sync-drive first.")
        return []

    docs = index.similarity_search(query, k=k)

    return [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]