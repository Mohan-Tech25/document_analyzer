import pymupdf


class PDFService:
    """
    Service responsible for extracting text
    and rendering pages from PDF documents.
    """

    def extract_pages(
        self,
        pdf_path: str,
    ) -> list[dict]:
        """
        Extract text from each PDF page separately.
        """

        document = pymupdf.open(pdf_path)

        pages = []

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                text = page.get_text(
                    "text"
                )

                pages.append(
                    {
                        "page_number": page_number,
                        "text": (
                            text.strip()
                            if text
                            else ""
                        ),
                    }
                )

        finally:

            document.close()

        return pages

    def render_page(
        self,
        pdf_path: str,
        page_number: int,
    ) -> bytes:
        """
        Render one PDF page as PNG image bytes.

        page_number is 1-based.
        """

        document = pymupdf.open(
            pdf_path
        )

        try:

            page_index = page_number - 1

            if (
                page_index < 0
                or page_index >= len(document)
            ):
                raise ValueError(
                    f"Invalid PDF page number: "
                    f"{page_number}"
                )

            page = document[
                page_index
            ]

            # 2x resolution for OCR.
            matrix = pymupdf.Matrix(
                2,
                2,
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            return pixmap.tobytes(
                "png"
            )

        finally:

            document.close()


pdf_service = PDFService()