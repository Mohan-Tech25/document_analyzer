import httpx

from app.core.config import settings
from app.core.exceptions import OllamaException


class OllamaService:
    """
    Handles communication with the Ollama LLM.
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the configured Ollama model.
        """

        if not prompt or not prompt.strip():
            raise OllamaException(
                "Prompt cannot be empty."
            )

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=120.0,
            )

            response.raise_for_status()

            data = response.json()

        except httpx.TimeoutException as e:
            raise OllamaException(
                "Ollama request timed out."
            ) from e

        except httpx.HTTPStatusError as e:
            raise OllamaException(
                "Ollama returned an HTTP error."
            ) from e

        except httpx.RequestError as e:
            raise OllamaException(
                "Unable to connect to Ollama."
            ) from e

        except ValueError as e:
            raise OllamaException(
                "Ollama returned an invalid response."
            ) from e

        response_text = data.get("response")

        if not response_text:
            raise OllamaException(
                "Ollama returned an empty response."
            )

        return response_text.strip()

    def understand_question(self, question: str) -> str:
        """
        Understand and normalize a user's question
        before semantic retrieval.
        """

        if not question or not question.strip():
            raise OllamaException(
                "Question cannot be empty."
            )

        prompt = f"""
You are a query understanding assistant for a
document question-answering system.

Rewrite the user's question into a clear question
while preserving the user's intended meaning.

Fix:
- spelling mistakes
- typing mistakes
- missing letters
- informal wording
- basic grammar mistakes

IMPORTANT:

If the user is asking about the document generally,
rewrite the question as a document-level question.

Examples:

"wat is docment contrnt ?"
-> What information does this document contain?

"wat is this docment abot?"
-> What is this document about?

"tell me abt dis docment"
-> What information does this document contain?

"wht does this file have?"
-> What information does this document contain?

"sumarize dis doc"
-> Summarize this document.

For specific questions, preserve the specific intent.

Examples:

"wht is jwt?"
-> What is JWT?

"wat is the persons name?"
-> What is the person's name?

Do NOT answer the question.

Return ONLY the corrected question.

USER QUESTION:

{question}
"""

        return self.generate_response(
            prompt=prompt
        )


ollama_service = OllamaService()