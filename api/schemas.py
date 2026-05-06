from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

class SyncResponse(BaseModel):
    status: str
    new_files_processed: int
    updated_files_processed: int
    files_skipped: int
    errors: Optional[List[str]] = []