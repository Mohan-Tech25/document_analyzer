import os

from psycopg import Connection

from app.core.exceptions import (
    AnalysisException,
)

from app.repositories.document_repository import (
    DocumentRepository,
)

from app.repositories.analysis_repository import (
    AnalysisRepository,
)

from app.repositories.chunk_repository import (
    ChunkRepository,
)

from app.services.ollama_service import (
    OllamaService,
)


class AnalysisService:
    """
    Service responsible for document analysis.

    Responsibilities:
    - Retrieve document information
    - Retrieve previous analyses
    - Generate document summaries
    - Analyze images using the vision model
    - Manage analysis transactions
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        analysis_repository: AnalysisRepository,
        chunk_repository: ChunkRepository,
        ollama_service: OllamaService,
    ):
        self.document_repository = document_repository
        self.analysis_repository = analysis_repository
        self.chunk_repository = chunk_repository
        self.ollama_service = ollama_service

    # ========================================================
    # GET DOCUMENT
    # ========================================================

    def get_document(
        self,
        connection: Connection,
        document_id: int,
    ):
        return self.document_repository.get_by_id(
            connection=connection,
            document_id=document_id,
        )

    # ========================================================
    # GET DOCUMENT ANALYSES
    # ========================================================

    def get_document_analyses(
        self,
        connection: Connection,
        document_id: int,
    ):
        rows = self.analysis_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        return [
            {
                "id": row[0],
                "document_id": row[1],
                "analysis_type": row[2],
                "result": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    def _is_image(
        self,
        file_type: str,
        file_path: str,
    ) -> bool:
        """
        Determine whether the uploaded document is an image.
        """

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".gif",
            ".tiff",
            ".tif",
        }

        if file_type:

            file_type_lower = file_type.lower()

            if file_type_lower.startswith("image/"):
                return True

            if file_type_lower in {
                "jpg",
                "jpeg",
                "png",
                "webp",
                "bmp",
                "gif",
                "tiff",
                "tif",
            }:
                return True

        if file_path:

            extension = os.path.splitext(
                file_path
            )[1].lower()

            if extension in image_extensions:
                return True

        return False

    # ========================================================
    # SUMMARIZE DOCUMENT
    # ========================================================

    def summarize_document(
        self,
        connection: Connection,
        document_id: int,
    ) -> dict:
        """
        Generate an AI analysis for a document.

        Text/PDF documents use the normal Ollama model.

        Images use the vision-language model. If OCR text
        exists, it is also provided to the vision model.
        """

        # -----------------------------------------
        # Retrieve document
        # -----------------------------------------

        document = self.document_repository.get_by_id(
            connection=connection,
            document_id=document_id,
        )

        if document is None:
            raise AnalysisException(
                "Document not found."
            )

        # Database structure:
        #
        # row[0] = id
        # row[1] = filename
        # row[2] = file_type
        # row[3] = document_type
        # row[4] = file_path
        # row[5] = status
        # row[6] = created_at

        file_type = document[2]
        file_path = document[4]

        # -----------------------------------------
        # IMAGE ANALYSIS
        # -----------------------------------------

        if self._is_image(
            file_type=file_type,
            file_path=file_path,
        ):
            return self._analyze_image(
                connection=connection,
                document_id=document_id,
                file_path=file_path,
            )

        # -----------------------------------------
        # TEXT/PDF ANALYSIS
        # -----------------------------------------

        chunks = self.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        if not chunks:
            raise AnalysisException(
                "No content found for this document."
            )

        # -----------------------------------------
        # Build document text
        # -----------------------------------------

        document_text = "\n\n".join(
            chunk[2]
            for chunk in chunks
            if chunk[2]
        )

        if not document_text.strip():
            raise AnalysisException(
                "Document contains no text."
            )

        # -----------------------------------------
        # Build AI prompt
        # -----------------------------------------

        prompt = f"""
You are a document analysis assistant.

Analyze the following document and provide
a clear summary.

Rules:

- Identify what type of document it appears to be.
- Explain the main purpose of the document.
- Mention the most important information.
- Do not invent information.
- Keep the response concise and easy to understand.

DOCUMENT:

{document_text}
"""

        # -----------------------------------------
        # Generate summary and save analysis
        # -----------------------------------------

        try:

            summary = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

            analysis_id = (
                self.analysis_repository.create(
                    connection=connection,
                    document_id=document_id,
                    analysis_type="summary",
                    result=summary,
                )
            )

            connection.commit()

            return {
                "analysis_id": analysis_id,
                "document_id": document_id,
                "analysis_type": "summary",
                "result": summary,
            }

        except AnalysisException:

            connection.rollback()

            raise

        except Exception as e:

            connection.rollback()

            raise AnalysisException(
                "Document analysis failed."
            ) from e

    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    def _analyze_image(
        self,
        connection: Connection,
        document_id: int,
        file_path: str,
    ) -> dict:
        """
        Analyze an image using the Ollama vision model.

        OCR text is included when available, but the actual
        image is always sent to the vision model.
        """

        if not file_path:
            raise AnalysisException(
                "Image file path is missing."
            )

        if not os.path.exists(file_path):
            raise AnalysisException(
                "Image file could not be found."
            )

        # -----------------------------------------
        # Get OCR text if available
        # -----------------------------------------

        chunks = self.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        ocr_text = ""

        if chunks:

            ocr_text = "\n\n".join(
                chunk[2]
                for chunk in chunks
                if chunk[2]
            ).strip()

        # -----------------------------------------
        # Build vision prompt
        # -----------------------------------------

        if ocr_text:

            prompt = f"""
You are an intelligent document and image analysis assistant.

Analyze the provided image carefully.

Use BOTH:

1. The actual visual information in the image.
2. The OCR text extracted from the image.

Describe what the image contains and explain the
important information visible in it.

Rules:

- Identify the main subject or purpose of the image.
- Describe important visual elements.
- Read and explain visible text when useful.
- Use the OCR text as supporting information.
- Do not invent information that is not visible.
- If something is uncertain, say that it is uncertain.
- Keep the response clear and easy to understand.

OCR TEXT:

{ocr_text}
"""

        else:

            prompt = """
You are an intelligent image analysis assistant.

Analyze the provided image carefully and describe
what is visible.

Rules:

- Identify the main subject of the image.
- Describe the important objects, people, animals,
  places, or scenes that are visible.
- Explain what appears to be happening in the image.
- If there is visible text, read and explain it.
- Do not invent information.
- If something is uncertain, clearly say so.
- Keep the response clear and easy to understand.

The image may contain no text. In that case,
focus entirely on the visual content.
"""

        # -----------------------------------------
        # Generate vision response
        # -----------------------------------------

        try:

            result = (
                self.ollama_service
                .generate_vision_response(
                    image_path=file_path,
                    prompt=prompt,
                )
            )

            analysis_id = (
                self.analysis_repository.create(
                    connection=connection,
                    document_id=document_id,
                    analysis_type="vision",
                    result=result,
                )
            )

            connection.commit()

            return {
                "analysis_id": analysis_id,
                "document_id": document_id,
                "analysis_type": "vision",
                "result": result,
            }

        except AnalysisException:

            connection.rollback()

            raise

        except Exception as e:

            connection.rollback()

            raise AnalysisException(
                f"Image analysis failed: {str(e)}"
            ) from e


analysis_service = AnalysisService(
    document_repository=DocumentRepository(),
    analysis_repository=AnalysisRepository(),
    chunk_repository=ChunkRepository(),
    ollama_service=OllamaService(),
)