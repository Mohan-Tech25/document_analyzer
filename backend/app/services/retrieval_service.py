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

    def get_document_chunks(
        self,
        connection: Connection,
        document_id: int,
    ) -> list[dict]:
        """
        Get all chunks belonging to a document.
        Used for broad document-level questions.
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

    def retrieve_relevant_chunks(
        self,
        connection: Connection,
        document_id: int,
        question: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve semantically relevant chunks
        using pgvector similarity search.

        A lower cosine distance means the chunk
        is more similar to the question.

        Chunks whose best distance is >= the
        relevance threshold are rejected.
        """

        if not question or not question.strip():
            raise ChatException(
                "Question cannot be empty."
            )

        query_embedding = (
            self.embedding_service.create_embedding(
                question
            )
        )

        rows = self.chunk_repository.similarity_search(
            connection=connection,
            document_id=document_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not rows:
            return []

        best_distance = rows[0][5]

        if best_distance >= self.RELEVANCE_THRESHOLD:
            return []

        return [
            {
                "id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "page_number": row[3],
                "created_at": row[4],
                "distance": row[5],
            }
            for row in rows
        ]


retrieval_service = RetrievalService(
    chunk_repository=ChunkRepository(),
    embedding_service=EmbeddingService(),
)