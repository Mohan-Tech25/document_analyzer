from psycopg import Connection


class AnalysisRepository:
    """
    Handles all database operations related
    to document analysis.
    """

    def create(
        self,
        connection: Connection,
        document_id: int,
        analysis_type: str,
        result: str,
    ) -> int:
        """
        Insert a document analysis result
        and return its ID.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO document_analysis
                (
                    document_id,
                    analysis_type,
                    result
                )
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (
                    document_id,
                    analysis_type,
                    result,
                ),
            )

            analysis_id = cursor.fetchone()[0]

        return analysis_id

    def get_by_document(
        self,
        connection: Connection,
        document_id: int,
    ):
        """
        Retrieve all analysis results
        belonging to a document.
        """

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    document_id,
                    analysis_type,
                    result,
                    created_at
                FROM document_analysis
                WHERE document_id = %s
                ORDER BY created_at DESC;
                """,
                (document_id,),
            )

            return cursor.fetchall()


analysis_repository = AnalysisRepository()