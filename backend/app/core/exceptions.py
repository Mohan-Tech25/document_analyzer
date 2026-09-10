class DocumentAnalyzerException(Exception):
    """
    Base exception for application-specific errors.
    """

    def __init__(
        self,
        message: str,
    ):
        self.message = message
        super().__init__(message)


# ============================================================
# DOCUMENT EXCEPTIONS
# ============================================================

class DocumentNotFoundException(
    DocumentAnalyzerException
):
    """
    Raised when a requested document
    does not exist.
    """

    def __init__(
        self,
        document_id: int,
    ):
        super().__init__(
            f"Document {document_id} not found."
        )


class UnsupportedFileException(
    DocumentAnalyzerException
):
    """
    Raised when an uploaded file type
    is not supported.
    """

    def __init__(
        self,
        file_type: str,
    ):
        super().__init__(
            f"Unsupported file type: {file_type}"
        )


class DocumentExtractionException(
    DocumentAnalyzerException
):
    """
    Raised when document text extraction fails.
    """

    def __init__(
        self,
        message: str = "Document text extraction failed.",
    ):
        super().__init__(message)


class OCRException(
    DocumentAnalyzerException
):
    """
    Raised when OCR processing fails.
    """

    def __init__(
        self,
        message: str = "OCR processing failed.",
    ):
        super().__init__(message)


# ============================================================
# AI EXCEPTIONS
# ============================================================

class EmbeddingException(
    DocumentAnalyzerException
):
    """
    Raised when embedding generation fails.
    """

    def __init__(
        self,
        message: str = "Embedding generation failed.",
    ):
        super().__init__(message)


class OllamaException(
    DocumentAnalyzerException
):
    """
    Raised when communication with Ollama fails.
    """

    def __init__(
        self,
        message: str = "Ollama service failed.",
    ):
        super().__init__(message)


class ChatException(
    DocumentAnalyzerException
):
    """
    Raised when document chat cannot
    process the user's question.
    """

    def __init__(
        self,
        message: str = "Document chat failed.",
    ):
        super().__init__(message)


class AnalysisException(
    DocumentAnalyzerException
):
    """
    Raised when document analysis cannot
    be completed.
    """

    def __init__(
        self,
        message: str = "Document analysis failed.",
    ):
        super().__init__(message)


# ============================================================
# DATABASE EXCEPTIONS
# ============================================================

class DatabaseException(
    DocumentAnalyzerException
):
    """
    Raised when a database operation fails.
    """

    def __init__(
        self,
        message: str = "Database operation failed.",
    ):
        super().__init__(message)