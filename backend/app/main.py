from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.exception_handlers import (
    document_analyzer_handler,
    document_not_found_handler,
    unsupported_file_handler,
)

from app.core.exceptions import (
    DocumentAnalyzerException,
    DocumentNotFoundException,
    UnsupportedFileException,
)

from app.database.database import database

from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.analysis import router as analysis_router
from app.api.health import router as health_router


app = FastAPI(
    title="Document Analyzer API",
    description="Document Intelligence API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# EXCEPTION HANDLERS
# ============================================================

app.add_exception_handler(
    DocumentNotFoundException,
    document_not_found_handler,
)

app.add_exception_handler(
    UnsupportedFileException,
    unsupported_file_handler,
)

app.add_exception_handler(
    DocumentAnalyzerException,
    document_analyzer_handler,
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    database.create_tables()


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    documents_router,
    prefix="/api",
)

app.include_router(
    chat_router,
    prefix="/api",
)

app.include_router(
    analysis_router,
    prefix="/api",
)

app.include_router(
    health_router,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Document Analyzer API is running."
    }
#uvicorn app.main:app --reload
#venv\Scripts\activate