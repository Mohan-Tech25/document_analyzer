from psycopg import Connection
from pgvector import Vector # type: ignore


class ChunkRepository:
    """
    Handles all database operations related
    to document chunks.
    """

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

    def get_by_document(
        self,
        connection: Connection,
        document_id: int,
    ):
        """
        Return all chunks belonging
        to a document.
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

    def update_embedding(
        self,
        connection: Connection,
        chunk_id: int,
        embedding: list[float],
    ):
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

    def similarity_search(
        self,
        connection: Connection,
        document_id: int,
        query_embedding: list[float],
        top_k: int = 3,
    ):
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


chunk_repository = ChunkRepository()