from psycopg import Connection

from app.core.exceptions import ChatException

from app.repositories.chunk_repository import (
    ChunkRepository,
)

from app.services.embedding_service import (
    EmbeddingService,
)


class RetrievalService:
    """
    Service responsible for retrieving
    relevant document chunks.
    """

    RELEVANCE_THRESHOLD = 0.60

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        embedding_service: EmbeddingService,
    ):
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service

    # =========================================================
    # GET ALL DOCUMENT CHUNKS
    # =========================================================

    def get_document_chunks(
        self,
        connection: Connection,
        document_id: int,
    ) -> list[dict]:
        """
        Get all chunks belonging to a document.

        Used for broad document-level questions
        and fallback retrieval.
        """

        rows = self.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        return [
            {
                "id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "page_number": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    # =========================================================
    # RETRIEVE RELEVANT CHUNKS
    # =========================================================

    def retrieve_relevant_chunks(
        self,
        connection: Connection,
        document_id: int,
        question: str,
        top_k: int = 3,
        document_type: str | None = None,
    ) -> list[dict]:
        """
        Retrieve semantically relevant chunks
        using pgvector similarity search.

        Lower cosine distance means higher similarity.

        For voter_list documents:

            1 voter = 1 chunk

        Therefore the complete matching chunk is
        returned without modifying or splitting it.
        """

        if not question or not question.strip():

            raise ChatException(
                "Question cannot be empty."
            )

        # -----------------------------------------------------
        # Create embedding for user question
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_service
            .create_embedding(
                question
            )
        )

        # -----------------------------------------------------
        # Search pgvector
        # -----------------------------------------------------

        rows = (
            self.chunk_repository
            .similarity_search(
                connection=connection,
                document_id=document_id,
                query_embedding=query_embedding,
                top_k=top_k,
            )
        )

        if not rows:
            return []

        # -----------------------------------------------------
        # Best similarity check
        # -----------------------------------------------------

        best_distance = rows[0][5]

        if best_distance >= self.RELEVANCE_THRESHOLD:
            return []

        # -----------------------------------------------------
        # Build results
        # -----------------------------------------------------

        results = []

        for row in rows:

            content = row[2]

            if not content:
                continue

            results.append(
                {
                    "id": row[0],
                    "chunk_index": row[1],

                    # IMPORTANT:
                    # Keep the complete voter record.
                    "content": content,

                    "page_number": row[3],
                    "created_at": row[4],
                    "distance": row[5],
                }
            )

        # -----------------------------------------------------
        # Voter-list handling
        # -----------------------------------------------------

        if document_type == "voter_list":

            # Each result already represents one voter.
            #
            # Do NOT merge multiple chunks.
            # Do NOT split the content.
            #
            # Example:
            #
            # Chunk 15 → Person 15
            # Chunk 16 → Person 16
            # Chunk 17 → Person 17

            return results

        # -----------------------------------------------------
        # Normal documents
        # -----------------------------------------------------

        return results


retrieval_service = RetrievalService(
    chunk_repository=ChunkRepository(),
    embedding_service=EmbeddingService(),
)