
from fastapi import (
    APIRouter,
    Depends,
)

from psycopg import Connection

from app.database.database import database

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from app.services.chat_service import (
    chat_service,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat_with_document(
    request: ChatRequest,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Ask a question about a document.
    """

    answer = chat_service.answer_question(
        connection=connection,
        document_id=request.document_id,
        question=request.question,
    )

    return ChatResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
    )
