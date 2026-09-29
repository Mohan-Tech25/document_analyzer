from pathlib import Path

from app.core.config import settings
from app.core.exceptions import (
    DocumentExtractionException,
    OCRException,
)
from app.core.logging import app_logger
from app.services.pdf_service import pdf_service
from app.services.ocr_service import ocr_service
from app.services.mineru_service import mineru_service


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

    # ------------------------------------------------------------------
    # MinerU
    # ------------------------------------------------------------------

    def _extract_with_mineru(
        self,
        file_path: str,
    ) -> dict:
        """
        Extract optional document structure using MinerU.

        MinerU is an enhancement layer.

        Its Markdown is currently NOT merged directly
        into OCR text because scanned voter records
        are extracted separately using PaddleOCR.

        MinerU failure does not stop document extraction.
        """

        if not settings.MINERU_ENABLED:

            app_logger.debug(
                "MinerU extraction is disabled."
            )

            return {}

        try:

            app_logger.info(
                "Starting MinerU extraction."
            )

            result = (
                mineru_service
                .parse_document(
                    file_path
                )
            )

            markdown = (
                result.get(
                    "markdown",
                    "",
                )
                .strip()
            )

            mineru_result = {
                "sha256": result.get(
                    "sha256"
                ),
                "short_id": result.get(
                    "short_id"
                ),
                "tier": result.get(
                    "tier"
                ),
                "page_range": result.get(
                    "page_range"
                ),
                "markdown": markdown,
            }

            app_logger.info(
                "MinerU extraction completed. "
                f"SHA={mineru_result['sha256']} "
                f"Tier={mineru_result['tier']} "
                f"MarkdownLength="
                f"{len(markdown)}"
            )

            return mineru_result

        except Exception as e:

            app_logger.warning(
                "MinerU extraction failed. "
                "Continuing with existing "
                "PDF/OCR extraction. "
                f"Error: {e}"
            )

            return {}

    # ------------------------------------------------------------------
    # Main document extraction
    # ------------------------------------------------------------------

    def extract_document_content(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Extract text from a PDF or image.

        PDF workflow:

            1. Run MinerU as optional enhancement.
            2. Try embedded PDF text page-by-page.
            3. If a page has no embedded text:
                   render page
                   run PaddleOCR
            4. For scanned voter pages:
                   preserve page-level header/footer text
                   preserve voter records separately.

        The important voter-list structure becomes:

            page header
            voter 1
            voter 2
            voter 3
            ...

        This allows document-level questions such as:

            "What is the constituency name?"

        to retrieve the page header.

        Images:

            OCR directly.
        """

        path = Path(file_path)

        if not path.exists():

            raise DocumentExtractionException(
                "Uploaded file could not be found."
            )

        extension = path.suffix.lower()

        # ==============================================================
        # PDF
        # ==============================================================

        if extension in self.PDF_EXTENSIONS:

            # ----------------------------------------------------------
            # Optional MinerU
            # ----------------------------------------------------------

            self._extract_with_mineru(
                str(path)
            )

            # ----------------------------------------------------------
            # Try embedded PDF text
            # ----------------------------------------------------------

            try:

                pages = (
                    pdf_service
                    .extract_pages(
                        str(path)
                    )
                )

            except Exception as e:

                raise DocumentExtractionException(
                    "Failed to extract text from PDF."
                ) from e

            processed_pages = []

            # ----------------------------------------------------------
            # Page-by-page processing
            # ----------------------------------------------------------

            try:

                for page in pages:

                    page_number = page[
                        "page_number"
                    ]

                    text = page.get(
                        "text",
                        "",
                    )

                    # ==================================================
                    # Embedded PDF text exists
                    # ==================================================

                    if text and text.strip():

                        processed_pages.append(
                            {
                                "page_number": page_number,
                                "text": text.strip(),
                            }
                        )

                        continue

                    # ==================================================
                    # Scanned PDF page
                    # ==================================================

                    image_bytes = (
                        pdf_service
                        .render_page(
                            str(path),
                            page_number,
                        )
                    )

                    temp_image_path = (
                        path.parent
                        / (
                            f".ocr_page_"
                            f"{page_number}.png"
                        )
                    )

                    try:

                        # --------------------------------------------------
                        # Write temporary page image
                        # --------------------------------------------------

                        with open(
                            temp_image_path,
                            "wb",
                        ) as image_file:

                            image_file.write(
                                image_bytes
                            )

                        # --------------------------------------------------
                        # OCR voter page
                        # --------------------------------------------------
                        #
                        # IMPORTANT:
                        #
                        # include_page_text=True
                        #
                        # This preserves page headers such as:
                        #
                        #   சட்டமன்றத் தொகுதியின் எண் மற்றும் பெயர் :
                        #   86-எடப்பாடி
                        #
                        #   பாகம் எண் : 128
                        #
                        #   பிரிவு எண் மற்றும் பெயர் ...
                        #
                        #   பட்டியல் வெளியிடப்பட்ட நாள் :
                        #   22-01-2024
                        #
                        # and puts that text BEFORE voter records.
                        # --------------------------------------------------

                        voter_records = (
                            ocr_service
                            .extract_voter_records(
                                str(
                                    temp_image_path
                                ),
                                language=(
                                    settings
                                    .SCANNED_PDF_OCR_LANGUAGE
                                ),
                                include_page_text=True,
                            )
                        )

                        # --------------------------------------------------
                        # Convert OCR blocks into page text
                        # --------------------------------------------------

                        ocr_text = (
                            "\n\n".join(
                                voter_records
                            )
                        )

                    finally:

                        # --------------------------------------------------
                        # Delete temporary OCR image
                        # --------------------------------------------------

                        if temp_image_path.exists():

                            try:
                                temp_image_path.unlink()

                            except OSError:
                                pass

                    # ------------------------------------------------------
                    # Preserve processed page
                    # ------------------------------------------------------

                    processed_pages.append(
                        {
                            "page_number": page_number,
                            "text": (
                                ocr_text.strip()
                                if ocr_text
                                else ""
                            ),
                        }
                    )

            except Exception as e:

                raise OCRException(
                    "Failed to extract text "
                    "from scanned PDF page."
                ) from e

            return processed_pages

        # ==============================================================
        # IMAGE
        # ==============================================================

        if extension in self.IMAGE_EXTENSIONS:

            try:

                text = (
                    ocr_service
                    .extract_text(
                        str(path)
                    )
                )

                return [
                    {
                        "page_number": None,
                        "text": text,
                    }
                ]

            except Exception as e:

                raise OCRException(
                    "Failed to extract text "
                    "from image."
                ) from e

        # ==============================================================
        # Unsupported file
        # ==============================================================

        raise DocumentExtractionException(
            f"Unsupported file type: {extension}"
        )


# ----------------------------------------------------------------------
# Global instance
# ----------------------------------------------------------------------

extraction_service = ExtractionService()