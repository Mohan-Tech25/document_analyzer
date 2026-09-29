from paddleocr import PPStructureV3


class TableService:
    """
    Service responsible for extracting structured
    information from table-based documents.
    """

    def __init__(self):
        self.pipeline = PPStructureV3(
            device="cpu",
        )

    def extract_structure(
        self,
        file_path: str,
    ) -> list:
        """
        Extract structured document information.

        Returns:
            List of PaddleOCR parsing results.
        """

        if not file_path:
            raise ValueError(
                "File path cannot be empty."
            )

        try:
            results = self.pipeline.predict(
                file_path
            )

            return list(results)

        except Exception as e:
            raise RuntimeError(
                "Failed to extract document structure."
            ) from e


table_service = TableService()