class ChunkingService:
    """
    Service responsible for splitting document text
    into smaller overlapping chunks.
    """

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
        """
        Split text into overlapping chunks.
        """

        if not text or not text.strip():
            return []

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        text = text.strip()

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - chunk_overlap

        return chunks


chunking_service = ChunkingService()