import httpx

from app.core.config import settings
from app.core.exceptions import EmbeddingException


class EmbeddingService:
    """
    Service responsible for creating text embeddings
    using the Ollama embedding model.
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_EMBEDDING_MODEL

    def create_embedding(
        self,
        text: str,
    ) -> list[float]:
        """
        Create a 768-dimensional embedding
        using the Ollama embedding model.

        Raises:
            EmbeddingException
        """

        if not text or not text.strip():
            raise EmbeddingException(
                "Text cannot be empty."
            )

        url = f"{self.base_url}/api/embed"

        payload = {
            "model": self.model,
            "input": text,
        }

        try:

            response = httpx.post(
                url,
                json=payload,
                timeout=60.0,
            )

            response.raise_for_status()

            data = response.json()

        except httpx.TimeoutException as e:

            raise EmbeddingException(
                "Embedding request timed out."
            ) from e

        except httpx.HTTPStatusError as e:

            raise EmbeddingException(
                "Ollama returned an HTTP error while creating the embedding."
            ) from e

        except httpx.RequestError as e:

            raise EmbeddingException(
                "Unable to connect to Ollama for embedding generation."
            ) from e

        except ValueError as e:

            raise EmbeddingException(
                "Ollama returned an invalid embedding response."
            ) from e

        embeddings = data.get("embeddings")

        if not embeddings:

            raise EmbeddingException(
                "Ollama returned no embedding."
            )

        embedding = embeddings[0]

        if len(embedding) != 768:

            raise EmbeddingException(
                f"Expected 768-dimensional embedding, "
                f"got {len(embedding)}."
            )

        return embedding


embedding_service = EmbeddingService()