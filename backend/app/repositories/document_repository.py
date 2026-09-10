from psycopg import Connection


class DocumentRepository:
    """
    Handles all database operations related to documents.
    """

    def create(
        self,
        connection: Connection,
        filename: str,
        file_type: str,
        file_path: str,
    ) -> int:
        """
        Insert a new document and return its ID.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO documents
                (
                    filename,
                    file_type,
                    file_path,
                    status
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    filename,
                    file_type,
                    file_path,
                    "uploaded",
                ),
            )

            document_id = cursor.fetchone()[0]

        return document_id

    def update_status(
        self,
        connection: Connection,
        document_id: int,
        status: str,
    ):
        """
        Update document processing status.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                UPDATE documents
                SET status = %s
                WHERE id = %s;
                """,
                (
                    status,
                    document_id,
                ),
            )

    def update_document_type(
        self,
        connection: Connection,
        document_id: int,
        document_type: str,
    ):
        """
        Store the detected document type.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                UPDATE documents
                SET document_type = %s
                WHERE id = %s;
                """,
                (
                    document_type,
                    document_id,
                ),
            )

    def get_all(
        self,
        connection: Connection,
    ):
        """
        Retrieve all documents.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    filename,
                    file_type,
                    document_type,
                    file_path,
                    status,
                    created_at
                FROM documents
                ORDER BY created_at DESC;
                """
            )

            return cursor.fetchall()

    def get_by_id(
        self,
        connection: Connection,
        document_id: int,
    ):
        """
        Retrieve one document by ID.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    filename,
                    file_type,
                    document_type,
                    file_path,
                    status,
                    created_at
                FROM documents
                WHERE id = %s;
                """,
                (document_id,),
            )

            return cursor.fetchone()

    def delete(
        self,
        connection: Connection,
        document_id: int,
    ) -> bool:
        """
        Delete a document by ID.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                DELETE FROM documents
                WHERE id = %s
                RETURNING id;
                """,
                (document_id,),
            )

            deleted_row = cursor.fetchone()

        return deleted_row is not None


document_repository = DocumentRepository()