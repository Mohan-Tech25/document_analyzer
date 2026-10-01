from psycopg import Connection
from pgvector import Vector  # type: ignore


class ChunkRepository:
    """
    Handles all database operations related
    to document chunks.
    """

    # ============================================================
    # CREATE CHUNK
    # ============================================================

    def create(
        self,
        connection: Connection,
        document_id: int,
        chunk_index: int,
        content: str,
        page_number: int | None = None,
    ) -> int:
        """
        Create a document chunk.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO document_chunks
                (
                    document_id,
                    chunk_index,
                    content,
                    page_number
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    document_id,
                    chunk_index,
                    content,
                    page_number,
                ),
            )

            chunk_id = cursor.fetchone()[0]

        return chunk_id

    # ============================================================
    # GET ALL CHUNKS FOR A DOCUMENT
    # ============================================================

    def get_by_document(
        self,
        connection: Connection,
        document_id: int,
    ) -> list[tuple]:
        """
        Return all chunks belonging to a document
        in their original document order.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    chunk_index,
                    content,
                    page_number,
                    created_at
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index;
                """,
                (document_id,),
            )

            return cursor.fetchall()

    # ============================================================
    # GET FIRST CHUNKS
    # ============================================================

    def get_first_chunks(
        self,
        connection: Connection,
        document_id: int,
        limit: int = 6,
    ) -> list[tuple]:
        """
        Return the first chunks of a document.

        These chunks are important because document titles,
        headings, metadata and identifying information commonly
        appear at the beginning of a document.

        This method does NOT use vector similarity.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    chunk_index,
                    content,
                    page_number,
                    created_at
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                LIMIT %s;
                """,
                (
                    document_id,
                    limit,
                ),
            )

            return cursor.fetchall()

    # ============================================================
    # UPDATE EMBEDDING
    # ============================================================

    def update_embedding(
        self,
        connection: Connection,
        chunk_id: int,
        embedding: list[float],
    ) -> None:
        """
        Store the embedding vector
        for a document chunk.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                UPDATE document_chunks
                SET embedding = %s
                WHERE id = %s;
                """,
                (
                    Vector(embedding),
                    chunk_id,
                ),
            )

    # ============================================================
    # SIMILARITY SEARCH
    # ============================================================

    def similarity_search(
        self,
        connection: Connection,
        document_id: int,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[tuple]:
        """
        Find the most semantically relevant
        chunks using pgvector cosine distance.
        """

        query_vector = Vector(query_embedding)

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    chunk_index,
                    content,
                    page_number,
                    created_at,
                    embedding <=> %s AS distance
                FROM document_chunks
                WHERE document_id = %s
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_vector,
                    document_id,
                    query_vector,
                    top_k,
                ),
            )

            return cursor.fetchall()


# ================================================================
# SINGLETON INSTANCE
# ================================================================

chunk_repository = ChunkRepository()