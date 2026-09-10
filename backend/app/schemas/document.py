from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """
    Response schema for a document.
    """

    id: int
    filename: str
    file_type: str
    document_type: str | None = None
    file_path: str
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    """
    Response schema for a list of documents.
    """

    documents: list[DocumentResponse]