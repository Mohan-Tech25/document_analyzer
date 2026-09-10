from pathlib import Path

from app.core.exceptions import (
    DocumentExtractionException,
    OCRException,
)

from app.services.pdf_service import (
    pdf_service,
)

from app.services.ocr_service import (
    ocr_service,
)


class ExtractionService:
    """
    Service responsible for extracting text
    from supported document types.
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tiff",
    }

    PDF_EXTENSIONS = {
        ".pdf",
    }

    def extract_document_content(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Extract text from a PDF or image.

        Raises:
            DocumentExtractionException
            OCRException
        """

        path = Path(file_path)

        if not path.exists():
            raise DocumentExtractionException(
                "Uploaded file could not be found."
            )

        extension = path.suffix.lower()

        # -----------------------------------------
        # PDF
        # -----------------------------------------

        if extension in self.PDF_EXTENSIONS:

            try:
                return pdf_service.extract_pages(
                    str(path)
                )

            except Exception as e:

                raise DocumentExtractionException(
                    "Failed to extract text from PDF."
                ) from e

        # -----------------------------------------
        # IMAGE
        # -----------------------------------------

        if extension in self.IMAGE_EXTENSIONS:

            try:

                text = ocr_service.extract_text(
                    str(path)
                )

                return [
                    {
                        "page_number": None,
                        "text": text,
                    }
                ]

            except Exception as e:

                raise OCRException(
                    "Failed to extract text from image."
                ) from e

        # -----------------------------------------
        # UNSUPPORTED
        # -----------------------------------------

        raise DocumentExtractionException(
            f"Unsupported file type: {extension}"
        )


extraction_service = ExtractionService()