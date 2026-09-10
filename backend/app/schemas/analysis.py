from datetime import datetime

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    """
    Response schema for a document analysis.
    """

    id: int
    document_id: int
    analysis_type: str
    result: str
    created_at: datetime


class AnalysisCreateResponse(BaseModel):
    """
    Response schema returned after creating an analysis.
    """

    analysis_id: int
    document_id: int
    analysis_type: str
    result: str


class AnalysisListResponse(BaseModel):
    """
    Response schema for all analyses of a document.
    """

    document_id: int
    analyses: list[AnalysisResponse]