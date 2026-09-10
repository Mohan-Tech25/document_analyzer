from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    Request schema for asking a question
    about a document.
    """

    document_id: int
    question: str


class ChatResponse(BaseModel):
    """
    Response schema for a document chat request.
    """

    document_id: int
    question: str
    answer: str