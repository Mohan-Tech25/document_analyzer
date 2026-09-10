from psycopg import Connection

from app.repositories.document_repository import (
    DocumentRepository,
)

from app.repositories.chunk_repository import (
    ChunkRepository,
)

from app.services.extraction_service import (
    ExtractionService,
)

from app.services.chunking_service import (
    ChunkingService,
)

from app.services.classification_service import (
    ClassificationService,
)

from app.services.embedding_service import (
    EmbeddingService,
)

from app.services.file_service import (
    FileService,
)


class DocumentService:
    """
    Handles the complete document processing workflow.

    Responsibilities:
        - Process the document
        - Manage the database transaction
        - Commit successful processing
        - Roll back failed processing
        - Remove the uploaded file when processing fails
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        extraction_service: ExtractionService,
        chunking_service: ChunkingService,
        classification_service: ClassificationService,
        embedding_service: EmbeddingService,
        file_service: FileService,
    ):
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.extraction_service = extraction_service
        self.chunking_service = chunking_service
        self.classification_service = classification_service
        self.embedding_service = embedding_service
        self.file_service = file_service

    def get_all_documents(
        self,
        connection: Connection,
    ):
        rows = self.document_repository.get_all(
            connection=connection
        )

        return [
            {
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "document_type": row[3],
                "file_path": row[4],
                "status": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    def get_document_by_id(
        self,
        connection: Connection,
        document_id: int,
    ):
        row = self.document_repository.get_by_id(
            connection=connection,
            document_id=document_id,
        )

        if row is None:
            return None

        return {
            "id": row[0],
            "filename": row[1],
            "file_type": row[2],
            "document_type": row[3],
            "file_path": row[4],
            "status": row[5],
            "created_at": row[6],
        }

    def delete_document(
        self,
        connection: Connection,
        document_id: int,
    ):
        document = self.document_repository.get_by_id(
            connection=connection,
            document_id=document_id,
        )

        if document is None:
            return None

        file_path = document[4]

        deleted = self.document_repository.delete(
            connection=connection,
            document_id=document_id,
        )

        if not deleted:
            return None

        connection.commit()

        self.file_service.delete_file(file_path)

        return {
            "document_id": document_id,
            "status": "deleted",
        }

    def process_document(
        self,
        connection: Connection,
        filename: str,
        file_type: str,
        file_path: str,
    ):
        """
        Process a document inside one database transaction.

        Success:
            Database changes are committed.

        Failure:
            Database changes are rolled back.
            Uploaded physical file is deleted.
            Original exception is re-raised.
        """

        try:

            # =========================================
            # CREATE DOCUMENT
            # =========================================

            document_id = (
                self.document_repository.create(
                    connection=connection,
                    filename=filename,
                    file_type=file_type,
                    file_path=file_path,
                )
            )

            # =========================================
            # EXTRACT DOCUMENT CONTENT
            # =========================================

            pages = (
                self.extraction_service
                .extract_document_content(
                    file_path
                )
            )

            if not pages:

                self.document_repository.update_status(
                    connection=connection,
                    document_id=document_id,
                    status="no_text_found",
                )

                connection.commit()

                return {
                    "document_id": document_id,
                    "status": "no_text_found",
                    "chunks_created": 0,
                }

            # =========================================
            # COMBINE TEXT
            # =========================================

            full_text = "\n\n".join(
                page["text"]
                for page in pages
                if page["text"]
            )

            # =========================================
            # CLASSIFY DOCUMENT
            # =========================================

            document_type = (
                self.classification_service
                .classify_document(
                    full_text
                )
            )

            self.document_repository.update_document_type(
                connection=connection,
                document_id=document_id,
                document_type=document_type,
            )

            # =========================================
            # CREATE CHUNKS + EMBEDDINGS
            # =========================================

            chunks_created = 0
            chunk_ids = []

            for page in pages:

                text = page["text"]
                page_number = page["page_number"]

                if not text or not text.strip():
                    continue

                chunks = (
                    self.chunking_service
                    .chunk_text(text)
                )

                for chunk in chunks:

                    chunk_id = (
                        self.chunk_repository.create(
                            connection=connection,
                            document_id=document_id,
                            chunk_index=chunks_created,
                            content=chunk,
                            page_number=page_number,
                        )
                    )

                    embedding = (
                        self.embedding_service
                        .create_embedding(chunk)
                    )

                    self.chunk_repository.update_embedding(
                        connection=connection,
                        chunk_id=chunk_id,
                        embedding=embedding,
                    )

                    chunk_ids.append(chunk_id)

                    chunks_created += 1

            # =========================================
            # NO TEXT / NO CHUNKS
            # =========================================

            if chunks_created == 0:

                self.document_repository.update_status(
                    connection=connection,
                    document_id=document_id,
                    status="no_text_found",
                )

                connection.commit()

                return {
                    "document_id": document_id,
                    "document_type": document_type,
                    "status": "no_text_found",
                    "chunks_created": 0,
                }

            # =========================================
            # MARK AS PROCESSED
            # =========================================

            self.document_repository.update_status(
                connection=connection,
                document_id=document_id,
                status="processed",
            )

            # =========================================
            # COMMIT TRANSACTION
            # =========================================

            connection.commit()

            return {
                "document_id": document_id,
                "document_type": document_type,
                "status": "processed",
                "chunks_created": chunks_created,
                "chunk_ids": chunk_ids,
            }

        except Exception:

            # =========================================
            # ROLLBACK DATABASE
            # =========================================

            connection.rollback()

            # =========================================
            # DELETE PHYSICAL FILE
            # =========================================

            self.file_service.delete_file(
                file_path
            )

            # =========================================
            # RE-RAISE ORIGINAL ERROR
            # =========================================

            raise


document_service = DocumentService(
    document_repository=DocumentRepository(),
    chunk_repository=ChunkRepository(),
    extraction_service=ExtractionService(),
    chunking_service=ChunkingService(),
    classification_service=ClassificationService(),
    embedding_service=EmbeddingService(),
    file_service=FileService(),
)