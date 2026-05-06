import os
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def process_file(file_path: str, doc_id: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Loads a PDF or Text Document, extracts text, and splits it into semantically meaningful chunks.
    Attaches tracking metadata to each chunk for vector database storage.

    Args:
        file_path: Absolute path to the locally downloaded file.
        doc_id:    Google Drive file ID (used as the unique document identifier).
        file_name: Original filename from Drive (shown as the source in answers).

    Returns:
        A list of chunk dicts with keys: 'text' and 'metadata'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the file at: {file_path}")

    # 1. Load the document dynamically based on file extension
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        logger.warning("Unsupported file type for chunking: %s", file_name)
        return []

    pages = loader.load()

    if not pages:
        logger.warning("PDF '%s' yielded no pages — skipping.", file_name)
        return []

    # 2. Configure the splitter with a 150-char overlap to preserve sentence context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # 3. Split into chunks
    langchain_chunks = text_splitter.split_documents(pages)

    # 4. Attach tracking metadata to every chunk
    processed_chunks = [
        {
            "text": chunk.page_content,
            "metadata": {
                "doc_id": doc_id,
                "file_name": file_name,
                "source": "gdrive",
                "page": chunk.metadata.get("page", 0),
            },
        }
        for chunk in langchain_chunks
        if chunk.page_content.strip()  # Skip empty/whitespace-only chunks
    ]

    logger.info(
        "Processed '%s': %d pages → %d chunks", file_name, len(pages), len(processed_chunks)
    )
    return processed_chunks