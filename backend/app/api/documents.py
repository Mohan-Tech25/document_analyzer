from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from psycopg import Connection

from app.core.config import settings

from app.core.exceptions import (
    DocumentAnalyzerException,
    DocumentNotFoundException,
    UnsupportedFileException,
)

from app.database.database import database

from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
)

from app.services.document_service import (
    document_service,
)

from app.services.file_service import (
    file_service,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
}


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Upload and process a document.
    """

    # -----------------------------------------
    # Validate filename
    # -----------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # -----------------------------------------
    # Validate extension
    # -----------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise UnsupportedFileException(
            extension
        )

    # -----------------------------------------
    # Create upload directory
    # -----------------------------------------

    upload_dir = Path(
        settings.UPLOAD_DIR
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Generate unique filename
    # -----------------------------------------

    unique_filename = (
        f"{uuid4()}{extension}"
    )

    file_path = (
        upload_dir / unique_filename
    )

    # -----------------------------------------
    # Save uploaded file
    # -----------------------------------------

    try:

        file_content = await file.read()

        file_service.save_file(
            file_path=str(file_path),
            file_content=file_content,
        )

    except Exception:

        # File saving failed before
        # document processing started.

        file_service.delete_file(
            str(file_path)
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        )

    # -----------------------------------------
    # Process document
    # -----------------------------------------

    try:

        result = (
            document_service
            .process_document(
                connection=connection,
                filename=file.filename,
                file_type=extension.lstrip("."),
                file_path=str(file_path),
            )
        )

    except DocumentAnalyzerException as e:

        # DocumentService already:
        # - rolled back the database
        # - deleted the uploaded file

        raise HTTPException(
            status_code=500,
            detail=e.message,
        )

    except Exception:

        # Unexpected error.
        # DocumentService already handles
        # rollback and file cleanup.

        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        )

    return {
        "message": (
            "Document processed successfully."
        ),
        "filename": file.filename,
        "file_type": extension.lstrip("."),
        **result,
    }


# ============================================================
# LIST DOCUMENTS
# ============================================================

@router.get(
    "/",
    response_model=DocumentListResponse,
)
def list_documents(
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Return all documents.
    """

    return {
        "documents": (
            document_service
            .get_all_documents(
                connection=connection
            )
        )
    }


# ============================================================
# GET DOCUMENT
# ============================================================

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def retrieve_document(
    document_id: int,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Return a single document.
    """

    document = (
        document_service
        .get_document_by_id(
            connection=connection,
            document_id=document_id,
        )
    )

    if document is None:

        raise DocumentNotFoundException(
            document_id
        )

    return document

    # ============================================================
# DELETE DOCUMENT
# ============================================================

@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Delete a document and its physical file.
    """

    document = (
        document_service
        .delete_document(
            connection=connection,
            document_id=document_id,
        )
    )

    if document is None:

        raise DocumentNotFoundException(
            document_id
        )

    return {
        "message": "Document deleted successfully.",
        **document,
    }