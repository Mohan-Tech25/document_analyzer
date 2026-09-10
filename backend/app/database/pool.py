from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector  # pyright: ignore[reportMissingImports]

from app.core.config import settings


def configure_connection(connection):
    """
    Configure every PostgreSQL connection
    created by the connection pool.
    """

    register_vector(connection)


pool = ConnectionPool(
    conninfo=settings.DATABASE_URL,
    min_size=settings.DB_POOL_MIN_SIZE,
    max_size=settings.DB_POOL_MAX_SIZE,
    timeout=settings.DB_POOL_TIMEOUT,
    configure=configure_connection,
)