import pymupdf


class PDFService:
    """
    Service responsible for extracting text
    from PDF documents.
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

                text = page.get_text("text")

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


pdf_service = PDFService()