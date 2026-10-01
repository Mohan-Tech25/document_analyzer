
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

    PDF strategy:

        Normal/text PDF
            ↓
        MinerU
            ↓
        Markdown
            ↓
        Chunking

        Scanned PDF
            ↓
        PaddleOCR - single pass
            ↓
        Is voter list?
            ├── YES → voter record text
            └── NO  → normal OCR text
            ↓
        Chunking

    Images:

        Image
            ↓
        OCR
            ↓
        Chunking
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
        Parse the PDF using MinerU.

        Returns MinerU parsing information including
        the Markdown representation.

        MinerU failure is handled by the caller so that
        the existing PDF/OCR extraction path can continue.
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

            if not markdown:

                app_logger.warning(
                    "MinerU returned empty Markdown."
                )

                return {}

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
                f"PageRange="
                f"{mineru_result['page_range']} "
                f"MarkdownLength="
                f"{len(markdown)}"
            )

            return mineru_result

        except Exception as exc:

            app_logger.warning(
                "MinerU extraction failed. "
                "Falling back to existing "
                "PDF/OCR extraction. "
                f"Error: {exc}"
            )

            return {}

    # ------------------------------------------------------------------
    # Determine whether PDF has embedded text
    # ------------------------------------------------------------------

    def _has_embedded_text(
        self,
        pages: list[dict],
    ) -> bool:
        """
        Determine whether the PDF contains usable
        embedded text.

        A PDF with embedded text is treated as a
        normal/text PDF.

        A PDF with no embedded text is treated as
        a scanned PDF.
        """

        for page in pages:

            text = page.get(
                "text",
                "",
            )

            if text and text.strip():

                return True

        return False

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

            1. Read PDF pages with PyMuPDF.
            2. Determine whether embedded text exists.
            3. If normal/text PDF:
                   use MinerU Markdown.
            4. If scanned PDF:
                   use PaddleOCR ONCE per page.
                   Detect voter structure from the
                   same OCR result.
                   If voter list:
                       use actual voter record text.
                   Otherwise:
                       use normal OCR text.
            5. Return processed pages for chunking.

        Images:

            OCR directly.

        IMPORTANT:

            Only voter-list extraction is treated
            specially here.

            Normal text PDFs, scanned non-voter
            PDFs, and images continue through their
            existing generic extraction path.
        """

        path = Path(
            file_path
        )

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
            # First inspect PDF pages
            # ----------------------------------------------------------

            try:

                pages = (
                    pdf_service
                    .extract_pages(
                        str(path)
                    )
                )

            except Exception as exc:

                raise DocumentExtractionException(
                    "Failed to extract text "
                    "from PDF."
                ) from exc

            if not pages:

                raise DocumentExtractionException(
                    "PDF contains no pages."
                )

            # ----------------------------------------------------------
            # Determine whether PDF has embedded text
            # ----------------------------------------------------------

            has_embedded_text = (
                self._has_embedded_text(
                    pages
                )
            )

            app_logger.info(
                "PDF text detection completed. "
                f"HasEmbeddedText="
                f"{has_embedded_text}"
            )

            # ==========================================================
            # NORMAL / TEXT PDF
            # ==========================================================

            if has_embedded_text:

                app_logger.info(
                    "PDF contains embedded text. "
                    "Attempting MinerU extraction."
                )

                mineru_result = (
                    self._extract_with_mineru(
                        str(path)
                    )
                )

                markdown = (
                    mineru_result.get(
                        "markdown",
                        "",
                    )
                    .strip()
                    if mineru_result
                    else ""
                )

                # ------------------------------------------------------
                # MinerU succeeded
                # ------------------------------------------------------

                if markdown:

                    app_logger.info(
                        "Using MinerU Markdown "
                        "as the primary extracted "
                        "document content."
                    )

                    return [
                        {
                            "page_number": None,
                            "text": markdown,
                        }
                    ]

                # ------------------------------------------------------
                # MinerU failed
                # ------------------------------------------------------

                app_logger.warning(
                    "MinerU Markdown unavailable "
                    "for normal PDF. "
                    "Falling back to embedded "
                    "PDF text."
                )

                processed_pages = []

                for page in pages:

                    page_number = page.get(
                        "page_number"
                    )

                    text = page.get(
                        "text",
                        "",
                    )

                    if (
                        not text
                        or not text.strip()
                    ):

                        continue

                    processed_pages.append(
                        {
                            "page_number": page_number,
                            "text": text.strip(),
                        }
                    )

                if not processed_pages:

                    raise DocumentExtractionException(
                        "PDF contains no usable text."
                    )

                return processed_pages

            # ==========================================================
            # SCANNED PDF
            # ==========================================================

            app_logger.info(
                "PDF contains no embedded text. "
                "Treating PDF as scanned document "
                "and using single-pass PaddleOCR."
            )

            processed_pages = []

            try:

                for page in pages:

                    page_number = page[
                        "page_number"
                    ]

                    # --------------------------------------------------
                    # Render scanned page
                    # --------------------------------------------------

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

                        # ----------------------------------------------
                        # Write temporary page image
                        # ----------------------------------------------

                        with open(
                            temp_image_path,
                            "wb",
                        ) as image_file:

                            image_file.write(
                                image_bytes
                            )

                        # ----------------------------------------------
                        # SINGLE-PASS OCR
                        # ----------------------------------------------

                        app_logger.info(
                            f"Processing scanned PDF "
                            f"page {page_number} "
                            "with single-pass OCR."
                        )

                        scanned_result = (
                            ocr_service
                            .extract_scanned_page(
                                image_path=str(
                                    temp_image_path
                                ),
                                language=(
                                    settings
                                    .SCANNED_PDF_OCR_LANGUAGE
                                ),
                                page_number=page_number,
                            )
                        )

                        # ----------------------------------------------
                        # Read single-pass result
                        # ----------------------------------------------

                        is_voter_list = (
                            scanned_result.get(
                                "is_voter_list",
                                False,
                            )
                        )

                        voter_records = (
                            scanned_result.get(
                                "records",
                                [],
                            )
                        )

                        ocr_text = (
                            scanned_result.get(
                                "text",
                                "",
                            )
                        )

                        # ----------------------------------------------
                        # VOTER LIST
                        # ----------------------------------------------

                        if (
                            is_voter_list
                            and voter_records
                        ):

                            app_logger.info(
                                f"Voter-list structure "
                                f"detected on page "
                                f"{page_number}. "
                                f"Records="
                                f"{len(voter_records)}"
                            )

                            # IMPORTANT:
                            #
                            # Do NOT serialize the complete
                            # voter dictionary using str(record).
                            #
                            # The previous implementation produced:
                            #
                            # {
                            #     'serial_number': 45,
                            #     'text': '...',
                            #     ...
                            # }
                            #
                            # That format caused the voter chunks
                            # to be treated as generic text and
                            # forced VoterRecordService to rebuild
                            # records from serialized dictionaries.
                            #
                            # Store only the actual OCR text here.
                            # ChunkingService will later perform
                            # voter-specific chunking because
                            # DocumentService passes document_type
                            # to it.

                            voter_texts = []

                            for record in voter_records:

                                if not isinstance(
                                    record,
                                    dict,
                                ):
                                    continue

                                record_text = (
                                    record.get(
                                        "text",
                                        "",
                                    )
                                )

                                if not isinstance(
                                    record_text,
                                    str,
                                ):
                                    continue

                                record_text = (
                                    record_text.strip()
                                )

                                if not record_text:
                                    continue

                                voter_texts.append(
                                    record_text
                                )

                            page_text = (
                                "\n\n".join(
                                    voter_texts
                                )
                            )

                        # ----------------------------------------------
                        # NORMAL SCANNED DOCUMENT
                        # ----------------------------------------------

                        else:

                            app_logger.info(
                                f"No voter-list "
                                f"structure detected "
                                f"on page "
                                f"{page_number}. "
                                "Using normal OCR text."
                            )

                            page_text = (
                                ocr_text
                            )

                        # ----------------------------------------------
                        # Store processed page
                        # ----------------------------------------------

                        processed_pages.append(
                            {
                                "page_number": page_number,
                                "text": (
                                    page_text.strip()
                                    if page_text
                                    else ""
                                ),
                            }
                        )

                    finally:

                        # ----------------------------------------------
                        # Delete temporary image
                        # ----------------------------------------------

                        if temp_image_path.exists():

                            try:

                                temp_image_path.unlink()

                            except OSError:

                                app_logger.warning(
                                    "Could not delete "
                                    "temporary OCR image: "
                                    f"{temp_image_path}"
                                )

            except Exception as exc:

                raise OCRException(
                    "Failed to extract text "
                    "from scanned PDF page."
                ) from exc

            # ----------------------------------------------------------
            # Validate scanned PDF result
            # ----------------------------------------------------------

            usable_pages = [
                page
                for page in processed_pages
                if page.get(
                    "text",
                    "",
                ).strip()
            ]

            if not usable_pages:

                raise DocumentExtractionException(
                    "Scanned PDF contains no "
                    "usable OCR text."
                )

            return processed_pages

        # ==============================================================
        # IMAGE
        # ==============================================================

        if extension in self.IMAGE_EXTENSIONS:

            try:

                text = (
                    ocr_service
                    .extract_document_text(
                        image_path=str(path),
                        language=settings.OCR_LANGUAGE,
                    )
                )

                return [
                    {
                        "page_number": None,
                        "text": text,
                    }
                ]

            except Exception as exc:

                raise OCRException(
                    "Failed to extract text "
                    "from image."
                ) from exc

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
