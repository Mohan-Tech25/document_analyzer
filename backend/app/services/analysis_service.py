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
    # SUMMARIZE DOCUMENT
    # ========================================================

    def summarize_document(
        self,
        connection: Connection,
        document_id: int,
    ) -> dict:
        """
        Generate an AI summary for a document.

        The database transaction is controlled
        by this service.
        """

        # -----------------------------------------
        # Retrieve document chunks
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


analysis_service = AnalysisService(
    document_repository=DocumentRepository(),
    analysis_repository=AnalysisRepository(),
    chunk_repository=ChunkRepository(),
    ollama_service=OllamaService(),
)