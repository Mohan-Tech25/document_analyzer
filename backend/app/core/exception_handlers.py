from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DocumentAnalyzerException,
    DocumentNotFoundException,
    UnsupportedFileException,
)


def document_not_found_handler(
    request: Request,
    exc: DocumentNotFoundException,
):
    """
    Convert document-not-found errors
    into HTTP 404 responses.
    """

    return JSONResponse(
        status_code=404,
        content={
            "detail": exc.message,
        },
    )


def unsupported_file_handler(
    request: Request,
    exc: UnsupportedFileException,
):
    """
    Convert unsupported-file errors
    into HTTP 400 responses.
    """

    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.message,
        },
    )


def document_analyzer_handler(
    request: Request,
    exc: DocumentAnalyzerException,
):
    """
    Handle all other application-specific errors.
    """

    return JSONResponse(
        status_code=500,
        content={
            "detail": exc.message,
        },
    )