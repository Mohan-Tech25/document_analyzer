import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    DATABASE_URL = os.getenv(
        "DATABASE_URL"
    )

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434"
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b"
    )

    OLLAMA_VISION_MODEL = os.getenv(
        "OLLAMA_VISION_MODEL",
        "qwen2.5vl:3b"
    )

    OLLAMA_EMBEDDING_MODEL = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text"
    )

    OCR_LANGUAGE = os.getenv(
        "OCR_LANGUAGE",
        "en"
    )

    SCANNED_PDF_OCR_LANGUAGE = os.getenv(
        "SCANNED_PDF_OCR_LANGUAGE",
        "en"
    )

    UPLOAD_DIR = os.getenv(
        "UPLOAD_DIR",
        "uploads"
    )

    MAX_FILE_SIZE_MB = int(
        os.getenv(
            "MAX_FILE_SIZE_MB",
            "20"
        )
    )

    DB_POOL_MIN_SIZE = int(
        os.getenv(
            "DB_POOL_MIN_SIZE",
            "2"
        )
    )

    DB_POOL_MAX_SIZE = int(
        os.getenv(
            "DB_POOL_MAX_SIZE",
            "10"
        )
    )

    DB_POOL_TIMEOUT = float(
        os.getenv(
            "DB_POOL_TIMEOUT",
            "30"
        )
    )

    MINERU_ENABLED = os.getenv(
    "MINERU_ENABLED",
    "true"
).lower() == "true"

    MINERU_COMMAND = os.getenv(
        "MINERU_COMMAND",
        "mineru"
    )

    MINERU_TIER = os.getenv(
        "MINERU_TIER",
        "standard"
    )

    MINERU_REMOTE = os.getenv(
        "MINERU_REMOTE",
        "true"
    ).lower() == "true"

    MINERU_TIMEOUT = int(
        os.getenv(
            "MINERU_TIMEOUT",
            "300"
        )
    )


settings = Settings()