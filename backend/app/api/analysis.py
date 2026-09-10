from fastapi import (
    APIRouter,
    Depends,
)

from psycopg import Connection

from app.core.exceptions import (
    DocumentNotFoundException,
)

from app.database.database import database

from app.schemas.analysis import (
    AnalysisCreateResponse,
    AnalysisListResponse,
)

from app.services.analysis_service import (
    analysis_service,
)


router = APIRouter(
    prefix="/documents",
    tags=["Analysis"],
)


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

@router.post(
    "/{document_id}/analyze",
    response_model=AnalysisCreateResponse,
)
def analyze_document(
    document_id: int,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Generate an AI summary for a document.
    """

    document = (
        analysis_service
        .get_document(
            connection=connection,
            document_id=document_id,
        )
    )

    if document is None:
        raise DocumentNotFoundException(
            document_id
        )

    return (
        analysis_service
        .summarize_document(
            connection=connection,
            document_id=document_id,
        )
    )


# ============================================================
# GET DOCUMENT ANALYSES
# ============================================================

@router.get(
    "/{document_id}/analysis",
    response_model=AnalysisListResponse,
)
def get_document_analysis(
    document_id: int,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Get all AI analyses for a document.
    """

    document = (
        analysis_service
        .get_document(
            connection=connection,
            document_id=document_id,
        )
    )

    if document is None:
        raise DocumentNotFoundException(
            document_id
        )

    analyses = (
        analysis_service
        .get_document_analyses(
            connection=connection,
            document_id=document_id,
        )
    )

    return {
        "document_id": document_id,
        "analyses": analyses,
    }