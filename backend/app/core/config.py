import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    DATABASE_URL = os.getenv("DATABASE_URL")

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434"
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b"
    )

    OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
    )

    UPLOAD_DIR = os.getenv(
        "UPLOAD_DIR",
        "uploads"
    )

    DB_POOL_MIN_SIZE = int(
        os.getenv("DB_POOL_MIN_SIZE", "2")
    )

    DB_POOL_MAX_SIZE = int(
        os.getenv("DB_POOL_MAX_SIZE", "10")
    )

    DB_POOL_TIMEOUT = float(
        os.getenv("DB_POOL_TIMEOUT", "30")
    )


settings = Settings()