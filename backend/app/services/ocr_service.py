from paddleocr import PaddleOCR


class OCRService:
    """
    Service responsible for extracting text
    from images using PaddleOCR.
    """

    def __init__(self):
        """
        Initialize PaddleOCR once when the service is created.
        """

        self.ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,

            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_text(
        self,
        image_path: str,
    ) -> str:
        """
        Extract text from an image.
        """

        result = self.ocr.predict(
            image_path
        )

        extracted_lines = []

        for page in result:

            texts = page.get(
                "rec_texts",
                []
            )

            for text in texts:

                if text and text.strip():

                    extracted_lines.append(
                        text.strip()
                    )

        return "\n".join(
            extracted_lines
        )


ocr_service = OCRService()