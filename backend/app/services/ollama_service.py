
import base64
import os
import tempfile
import time

import httpx
from PIL import Image

from app.core.config import settings
from app.core.exceptions import OllamaException


class OllamaService:
    """
    Handles communication with the Ollama LLM and
    vision-language model.
    """

    # Text generation can take longer for large document
    # analysis prompts, especially on CPU systems.
    TEXT_TIMEOUT = 600.0

    # Vision requests can be computationally expensive.
    VISION_TIMEOUT = 600.0

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.vision_model = settings.OLLAMA_VISION_MODEL

    # =========================================================
    # TEXT GENERATION
    # =========================================================

    def generate_response(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a response using the configured Ollama text model.

        Args:
            prompt:
                Prompt sent to the Ollama text model.

            json_mode:
                When True, Ollama is instructed to return
                a valid JSON response.

        Returns:
            Generated response text.
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

        # -----------------------------------------------------
        # JSON MODE
        # -----------------------------------------------------

        if json_mode:
            payload["format"] = "json"

        prompt_size = len(prompt)

        print(
            "\n"
            "==================================================\n"
            "OLLAMA TEXT REQUEST\n"
            "=================================================="
        )

        print(
            f"Base URL       : {self.base_url}"
        )

        print(
            f"Model          : {self.model}"
        )

        print(
            f"Prompt Size    : {prompt_size} characters"
        )

        print(
            f"JSON Mode      : {json_mode}"
        )

        print(
            f"Timeout        : {self.TEXT_TIMEOUT} seconds"
        )

        print(
            "Sending request to Ollama..."
        )

        print(
            "==================================================\n"
        )

        start_time = time.perf_counter()

        try:

            response = httpx.post(
                url,
                json=payload,
                timeout=self.TEXT_TIMEOUT,
            )

            elapsed_time = (
                time.perf_counter() - start_time
            )

            print(
                "\n========== OLLAMA RESPONSE =========="
            )

            print(
                f"HTTP Status    : {response.status_code}"
            )

            print(
                f"Response Time  : {elapsed_time:.2f} seconds"
            )

            response.raise_for_status()

            data = response.json()

        except httpx.TimeoutException as e:

            elapsed_time = (
                time.perf_counter() - start_time
            )

            print(
                "\n========== OLLAMA TIMEOUT =========="
            )

            print(
                f"Model         : {self.model}"
            )

            print(
                f"Prompt Size   : {prompt_size} characters"
            )

            print(
                f"Elapsed Time  : {elapsed_time:.2f} seconds"
            )

            print(
                f"Timeout Limit : {self.TEXT_TIMEOUT} seconds"
            )

            print(
                "====================================\n"
            )

            raise OllamaException(
                "Ollama request timed out."
            ) from e

        except httpx.HTTPStatusError as e:

            print(
                "\n========== OLLAMA HTTP ERROR =========="
            )

            print(
                f"Status Code: {e.response.status_code}"
            )

            print(
                f"Response: {e.response.text[:2000]}"
            )

            print(
                "=======================================\n"
            )

            raise OllamaException(
                "Ollama returned an HTTP error."
            ) from e

        except httpx.RequestError as e:

            print(
                "\n========== OLLAMA CONNECTION ERROR =========="
            )

            print(
                f"ERROR TYPE: {type(e).__name__}"
            )

            print(
                f"ERROR: {e}"
            )

            print(
                "=============================================\n"
            )

            raise OllamaException(
                "Unable to connect to Ollama."
            ) from e

        except ValueError as e:

            print(
                "\n========== OLLAMA JSON ERROR =========="
            )

            print(
                f"ERROR: {e}"
            )

            print(
                f"RAW RESPONSE: {response.text[:2000]}"
            )

            print(
                "=======================================\n"
            )

            raise OllamaException(
                "Ollama returned an invalid response."
            ) from e

        response_text = data.get("response")

        if not response_text:

            print(
                "\n========== OLLAMA EMPTY RESPONSE =========="
            )

            print(
                f"RAW DATA: {data}"
            )

            print(
                "============================================\n"
            )

            raise OllamaException(
                "Ollama returned an empty response."
            )

        print(
            f"Response Size  : {len(response_text)} characters"
        )

        print(
            "======================================\n"
        )

        return response_text.strip()

    # =========================================================
    # PREPARE IMAGE
    # =========================================================

    def _prepare_image(
        self,
        image_path: str,
    ) -> str:
        """
        Resize large images before sending them
        to the Ollama vision model.

        The original image is never modified.

        Returns:
            Path to the prepared image.
        """

        max_dimension = 2000

        try:

            with Image.open(image_path) as image:

                width, height = image.size

                # ---------------------------------------------
                # Image already small enough
                # ---------------------------------------------

                if (
                    width <= max_dimension
                    and height <= max_dimension
                ):
                    return image_path

                # ---------------------------------------------
                # Calculate new dimensions
                # while preserving aspect ratio
                # ---------------------------------------------

                scale = min(
                    max_dimension / width,
                    max_dimension / height,
                )

                new_width = int(
                    width * scale
                )

                new_height = int(
                    height * scale
                )

                resized_image = image.resize(
                    (
                        new_width,
                        new_height,
                    ),
                    Image.Resampling.LANCZOS,
                )

                # ---------------------------------------------
                # Create temporary image
                # ---------------------------------------------

                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                )

                temp_path = temp_file.name

                temp_file.close()

                # ---------------------------------------------
                # Convert to RGB
                # for JPEG compatibility
                # ---------------------------------------------

                if resized_image.mode != "RGB":

                    resized_image = (
                        resized_image.convert("RGB")
                    )

                resized_image.save(
                    temp_path,
                    format="JPEG",
                    quality=85,
                    optimize=True,
                )

                resized_image.close()

                return temp_path

        except OSError as e:

            raise OllamaException(
                "Unable to prepare the image for vision analysis."
            ) from e

    # =========================================================
    # VISION RESPONSE
    # =========================================================

    def generate_vision_response(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """
        Generate a response using the configured
        Ollama vision-language model.
        """

        if not image_path:

            raise OllamaException(
                "Image path cannot be empty."
            )

        if not prompt or not prompt.strip():

            raise OllamaException(
                "Vision prompt cannot be empty."
            )

        prepared_image_path = image_path

        try:

            # ---------------------------------------------
            # Resize large image if necessary
            # ---------------------------------------------

            prepared_image_path = (
                self._prepare_image(
                    image_path
                )
            )

            # ---------------------------------------------
            # Read prepared image
            # ---------------------------------------------

            with open(
                prepared_image_path,
                "rb",
            ) as image_file:

                image_bytes = image_file.read()

            image_base64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")

        except OSError as e:

            raise OllamaException(
                "Unable to read the image file."
            ) from e

        finally:

            # ---------------------------------------------
            # Delete temporary resized image
            # ---------------------------------------------

            if (
                prepared_image_path != image_path
                and os.path.exists(
                    prepared_image_path
                )
            ):

                try:

                    os.remove(
                        prepared_image_path
                    )

                except OSError:

                    pass

        # ---------------------------------------------
        # Ollama vision API
        # ---------------------------------------------

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [
                image_base64
            ],
            "stream": False,
            "options": {
                "num_ctx": 8192,
            },
        }

        print(
            "\n"
            "==================================================\n"
            "OLLAMA VISION REQUEST\n"
            "=================================================="
        )

        print(
            f"Base URL       : {self.base_url}"
        )

        print(
            f"Vision Model   : {self.vision_model}"
        )

        print(
            f"Prompt Size    : {len(prompt)} characters"
        )

        print(
            f"Image Size     : {len(image_bytes)} bytes"
        )

        print(
            f"Timeout        : {self.VISION_TIMEOUT} seconds"
        )

        print(
            "Sending vision request to Ollama..."
        )

        print(
            "==================================================\n"
        )

        start_time = time.perf_counter()

        try:

            response = httpx.post(
                url,
                json=payload,
                timeout=self.VISION_TIMEOUT,
            )

            elapsed_time = (
                time.perf_counter() - start_time
            )

            print(
                "\n========== OLLAMA VISION RESPONSE =========="
            )

            print(
                f"HTTP Status    : {response.status_code}"
            )

            print(
                f"Response Time  : {elapsed_time:.2f} seconds"
            )

            response.raise_for_status()

            data = response.json()

        except httpx.TimeoutException as e:

            elapsed_time = (
                time.perf_counter() - start_time
            )

            print(
                "\n========== OLLAMA VISION TIMEOUT =========="
            )

            print(
                f"Model         : {self.vision_model}"
            )

            print(
                f"Elapsed Time  : {elapsed_time:.2f} seconds"
            )

            print(
                f"Timeout Limit : {self.VISION_TIMEOUT} seconds"
            )

            print(
                "===========================================\n"
            )

            raise OllamaException(
                "Ollama vision request timed out."
            ) from e

        except httpx.HTTPStatusError as e:

            error_body = e.response.text.strip()

            print(
                "\n========== OLLAMA VISION HTTP ERROR =========="
            )

            print(
                f"Status Code: {e.response.status_code}"
            )

            print(
                f"Response: {error_body[:2000]}"
            )

            print(
                "===============================================\n"
            )

            raise OllamaException(
                f"Ollama vision HTTP error "
                f"{e.response.status_code}: "
                f"{error_body}"
            ) from e

        except httpx.RequestError as e:

            print(
                "\n========== OLLAMA VISION CONNECTION ERROR =========="
            )

            print(
                f"ERROR TYPE: {type(e).__name__}"
            )

            print(
                f"ERROR: {e}"
            )

            print(
                "======================================================\n"
            )

            raise OllamaException(
                "Unable to connect to Ollama vision model."
            ) from e

        except ValueError as e:

            print(
                "\n========== OLLAMA VISION JSON ERROR =========="
            )

            print(
                f"ERROR: {e}"
            )

            print(
                f"RAW RESPONSE: {response.text[:2000]}"
            )

            print(
                "===============================================\n"
            )

            raise OllamaException(
                "Ollama vision model returned an invalid response."
            ) from e

        # ---------------------------------------------
        # Read Ollama response
        # ---------------------------------------------

        response_text = data.get(
            "response"
        )

        if not response_text:

            print(
                "\n========== OLLAMA VISION EMPTY RESPONSE =========="
            )

            print(
                f"RAW DATA: {data}"
            )

            print(
                "===================================================\n"
            )

            raise OllamaException(
                "Ollama vision model returned an empty response."
            )

        print(
            f"Response Size  : {len(response_text)} characters"
        )

        print(
            "============================================\n"
        )

        return response_text.strip()

    # =========================================================
    # QUESTION UNDERSTANDING
    # =========================================================

    def understand_question(
        self,
        question: str,
    ) -> str:
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
