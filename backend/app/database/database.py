from app.database.pool import pool

from app.database.models import (
    DOCUMENTS_TABLE,
    DOCUMENT_CHUNKS_TABLE,
    DOCUMENT_ANALYSIS_TABLE,
)


class Database:
    """
    Handles database connections
    and table initialization.
    """

    def get_connection(self):
        """
        Provide a PostgreSQL connection
        from the connection pool.

        The connection is automatically
        returned to the pool after use.
        """

        with pool.connection() as connection:
            yield connection

    def check_connection(self):
        """
        Check whether PostgreSQL is reachable.
        """

        with pool.connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SELECT version();"
                )

                result = cursor.fetchone()

                return result[0]

    def create_tables(self):
        """
        Create application tables
        if they do not already exist.
        """

        with pool.connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    DOCUMENTS_TABLE
                )

                cursor.execute(
                    DOCUMENT_CHUNKS_TABLE
                )

                cursor.execute(
                    DOCUMENT_ANALYSIS_TABLE
                )

            connection.commit()


database = Database()