from fastapi import (
    APIRouter,
    HTTPException,
)

from app.database.database import database


router = APIRouter(
    tags=["Health"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def health_check():
    """
    Check API and PostgreSQL health.
    """

    try:

        database_version = (
            database.check_connection()
        )

        return {
            "status": "healthy",
            "database": "connected",
            "postgresql": database_version,
        }

    except Exception:

        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "disconnected",
            },
        )