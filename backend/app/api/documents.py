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

from app.services.voter_record_service import (
    voter_record_service,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

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

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # --------------------------------------------------------
    # Get extension
    # --------------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileException(
            extension
        )

    # --------------------------------------------------------
    # Create upload directory
    # --------------------------------------------------------

    upload_dir = Path(
        settings.UPLOAD_DIR
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    unique_filename = (
        f"{uuid4()}{extension}"
    )

    file_path = (
        upload_dir / unique_filename
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    try:

        file_content = await file.read()

        file_service.save_file(
            file_path=str(file_path),
            file_content=file_content,
        )

    except ValueError as e:

        print(
            "\n================ FILE SIZE ERROR ================\n"
        )

        print(
            repr(e)
        )

        print(
            "\n==================================================\n"
        )

        file_service.delete_file(
            str(file_path)
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        print(
            "\n================ FILE SAVE ERROR ================\n"
        )

        print(
            "Exception:",
            repr(e),
        )

        print(
            "\n==================================================\n"
        )

        file_service.delete_file(
            str(file_path)
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        )

    # --------------------------------------------------------
    # Process document
    # --------------------------------------------------------

    try:

        result = document_service.process_document(
            connection=connection,
            filename=file.filename,
            file_type=extension.lstrip("."),
            file_path=str(file_path),
        )

    # --------------------------------------------------------
    # Known application exception
    # --------------------------------------------------------

    except DocumentAnalyzerException as e:

        print(
            "\n================ DOCUMENT ERROR ================\n"
        )

        print(
            "Exception:",
            repr(e),
        )

        print(
            "Message:",
            getattr(
                e,
                "message",
                None,
            ),
        )

        print(
            "Cause:",
            repr(
                e.__cause__
            ),
        )

        print(
            "Context:",
            repr(
                e.__context__
            ),
        )

        print(
            "\n=================================================\n"
        )

        raise HTTPException(
            status_code=500,
            detail=getattr(
                e,
                "message",
                str(e),
            ),
        )

    # --------------------------------------------------------
    # Unexpected exception
    # --------------------------------------------------------

    except Exception as e:

        print(
            "\n================ UPLOAD ERROR ================\n"
        )

        print(
            "Exception:",
            repr(e),
        )

        print(
            "Type:",
            type(e).__name__,
        )

        print(
            "Cause:",
            repr(
                e.__cause__
            ),
        )

        print(
            "Context:",
            repr(
                e.__context__
            ),
        )

        print(
            "\n================================================\n"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    # --------------------------------------------------------
    # Success response
    # --------------------------------------------------------

    return {
        "message": "Document processed successfully.",
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
    Return all uploaded documents.
    """

    return {
        "documents": (
            document_service.get_all_documents(
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
        document_service.get_document_by_id(
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
# GET STRUCTURED VOTER RECORDS
# ============================================================

@router.get(
    "/{document_id}/voter-records"
)
def get_voter_records(
    document_id: int,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Return structured voter records
    for a voter-list document.
    """

    # --------------------------------------------------------
    # Get document
    # --------------------------------------------------------

    document = (
        document_service.get_document_by_id(
            connection=connection,
            document_id=document_id,
        )
    )

    if document is None:

        raise DocumentNotFoundException(
            document_id
        )

    # --------------------------------------------------------
    # Validate document type
    # --------------------------------------------------------

    if document["document_type"] != "voter_list":

        raise HTTPException(
            status_code=400,
            detail=(
                "Document is not classified "
                "as a voter list."
            ),
        )

    # --------------------------------------------------------
    # Get document chunks
    # --------------------------------------------------------

    rows = (
        document_service.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )
    )

    # --------------------------------------------------------
    # Convert chunks into structured records
    # --------------------------------------------------------

    records = (
        voter_record_service.parse_chunk_rows(
            rows
        )
    )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {
        "document_id": document_id,
        "document_type": document["document_type"],
        "total_records": len(records),
        "records": records,
    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@router.delete(
    "/{document_id}"
)
def delete_document(
    document_id: int,
    connection: Connection = Depends(
        database.get_connection
    ),
):
    """
    Delete a document and its associated file.
    """

    document = (
        document_service.delete_document(
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