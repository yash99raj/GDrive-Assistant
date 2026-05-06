import os
import logging
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Boot-time configuration — load .env BEFORE any project imports
# so that env vars are available when connectors/sync.py is imported.
# ---------------------------------------------------------------------------
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from connectors.sync import run_incremental_sync

from api.schemas import AskRequest, AskResponse, SyncResponse
from search.vector_store import retrieve_top_k

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Validate required environment variables at startup — fail fast
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with GROQ_API_KEY=gsk_..."
    )

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Highwatch AI - RAG API",
    description="A Retrieval-Augmented Generation API over Google Drive",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LLM — Groq (Llama 3 8B instant for maximum speed)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    temperature=0,  # Deterministic, factual answers only
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    max_tokens=1024,
)

# ---------------------------------------------------------------------------
# Strict RAG prompt — halluciations guardrail baked in
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI assistant for Highwatch AI.
Answer the user's question based on the context chunks provided below.
You may synthesize and summarize information across the chunks.
If the answer truly cannot be found or inferred from the context, say "I don't know based on the provided documents."
Do not invent facts or use knowledge outside the provided context.

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """
    Validates that the API is running and all required env vars are present.
    """
    import os as _os
    return {
        "status": "ok",
        "groq_key_set": bool(_os.getenv("GROQ_API_KEY")),
        "gdrive_folder_set": bool(_os.getenv("GDRIVE_FOLDER_ID")),
    }


# ---------------------------------------------------------------------------
# Debug — Drive access check
# ---------------------------------------------------------------------------
@app.get("/debug-drive")
async def debug_drive():
    """
    Directly lists files in the configured Drive folder.
    Use this to verify the service account has folder access.
    """
    from connectors.gdrive import list_drive_files
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    try:
        files = list_drive_files(folder_id)
        return {
            "folder_id": folder_id,
            "files_found": len(files),
            "files": [
                {"name": f["name"], "mimeType": f["mimeType"], "id": f["id"]}
                for f in files
            ],
        }
    except Exception as e:
        return {"error": str(e), "folder_id": folder_id}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/sync-drive", response_model=SyncResponse)
async def sync_google_drive():
    """
    Scans Google Drive, downloads new/updated files, chunks them, and updates FAISS.
    Only new or modified files are processed (incremental sync via SQLite).
    """
    try:
        stats = run_incremental_sync()
        return SyncResponse(
            status="success",
            new_files_processed=stats["new"],
            updated_files_processed=stats["updated"],
            files_skipped=stats["skipped"],
            errors=stats.get("errors", []),
        )
    except Exception as e:
        logger.exception("Sync failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Takes a query, retrieves top-k chunks from FAISS, and generates a grounded answer via Groq.
    """
    try:
        # Step 1: Retrieve context
        retrieved_chunks = retrieve_top_k(request.query, k=3)

        if not retrieved_chunks:
            return AskResponse(
                answer="I couldn't find any synced documents to answer this question. Please run /sync-drive first.",
                sources=[],
            )

        # Step 2: Format context and deduplicate sources
        context_text = "\n\n---\n\n".join([chunk["text"] for chunk in retrieved_chunks])

        # Use a set to avoid listing the same file multiple times in sources
        unique_sources = list(
            set(chunk["metadata"].get("file_name", "Unknown Source") for chunk in retrieved_chunks)
        )

        # Step 3: Generate the answer
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context_text,
            "question": request.query,
        })

        logger.info("Query answered. Sources: %s", unique_sources)

        return AskResponse(
            answer=response.content,
            sources=unique_sources,
        )

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")