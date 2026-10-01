from psycopg import Connection

from app.core.exceptions import ChatException
from app.repositories.chunk_repository import ChunkRepository
from app.services.embedding_service import EmbeddingService


class RetrievalService:
    """
    Service responsible for retrieving relevant document chunks.

    Retrieval strategy:

    1. Generate an embedding for the user question.
    2. Search document chunks using pgvector.
    3. Return the closest chunks.
    4. Do not discard all results merely because the similarity
       threshold is weak.

    Document-level/header retrieval is handled separately by
    ChatService using the actual document chunk order.
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

        Chunks are returned in document chunk order.
        """

        rows = self.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        if not rows:
            return []

        results = []

        for row in rows:

            if not row:
                continue

            if len(row) < 4:
                continue

            content = row[2]

            if not content:
                continue

            results.append(
                {
                    "id": row[0],
                    "chunk_index": row[1],
                    "content": content,
                    "page_number": row[3],
                    "created_at": (
                        row[4]
                        if len(row) > 4
                        else None
                    ),
                }
            )

        results.sort(
            key=lambda chunk: (
                int(chunk["chunk_index"])
                if chunk.get("chunk_index") is not None
                else 999999999
            )
        )

        return results

    # =========================================================
    # SEMANTIC RETRIEVAL
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
        Retrieve semantically relevant document chunks
        using pgvector similarity search.

        Lower cosine distance means higher similarity.

        Important:

        The relevance threshold is treated as a preference,
        not a hard failure.

        This is important because:

        English question
                +
        Tamil document

        can produce a weaker embedding similarity even when
        the requested information actually exists in the document.
        """

        if not question or not question.strip():

            raise ChatException(
                "Question cannot be empty."
            )

        # =====================================================
        # CREATE QUESTION EMBEDDING
        # =====================================================

        query_embedding = (
            self.embedding_service
            .create_embedding(
                question.strip()
            )
        )

        if not query_embedding:
            return []

        # =====================================================
        # VECTOR SEARCH
        # =====================================================

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

        # =====================================================
        # BUILD RESULTS
        # =====================================================

        results = []

        for row in rows:

            if not row:
                continue

            if len(row) < 6:
                continue

            content = row[2]

            if not content:
                continue

            distance = row[5]

            result = {
                "id": row[0],
                "chunk_index": row[1],

                # Keep complete source chunk.
                "content": content,

                "page_number": row[3],
                "created_at": row[4],
                "distance": distance,
            }

            results.append(
                result
            )

        if not results:
            return []

        # =====================================================
        # RELEVANCE PREFERENCE
        # =====================================================
        #
        # Do NOT return [] simply because the best distance is
        # above the threshold.
        #
        # The caller can combine these results with:
        #
        # - document header
        # - direct chunk retrieval
        # - other evidence
        #
        # This is especially important for multilingual documents.
        # =====================================================

        relevant_results = [
            result
            for result in results
            if (
                result.get("distance") is not None
                and result["distance"]
                < self.RELEVANCE_THRESHOLD
            )
        ]

        if relevant_results:
            return relevant_results

        # -----------------------------------------------------
        # No result passed threshold.
        #
        # Return the best available semantic result instead
        # of throwing away potentially useful evidence.
        # -----------------------------------------------------

        return results

    # =========================================================
    # OPTIONAL: GET FIRST DOCUMENT CHUNKS
    # =========================================================

    def get_first_document_chunks(
        self,
        connection: Connection,
        document_id: int,
        limit: int = 3,
    ) -> list[dict]:
        """
        Get the first N chunks of a document.

        Useful for document-level information such as:

        - title
        - heading
        - date
        - organization
        - location
        - constituency
        - document type

        This retrieval is positional, not semantic.
        """

        if limit <= 0:
            return []

        chunks = self.get_document_chunks(
            connection=connection,
            document_id=document_id,
        )

        return chunks[:limit]


# =============================================================
# SERVICE INITIALIZATION
# =============================================================

retrieval_service = RetrievalService(
    chunk_repository=ChunkRepository(),
    embedding_service=EmbeddingService(),
)