import re


class ChunkingService:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        document_type: str | None = None,
    ) -> list[str]:

        if not text or not text.strip():
            return []

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        # --------------------------------------------------------
        # VOTER LIST
        #
        # Voter lists need record-aware boundaries because one
        # person should remain inside one chunk.
        # --------------------------------------------------------

        if document_type == "voter_list":

            return self._chunk_voter_list(
                text=text
            )

        # --------------------------------------------------------
        # ALL OTHER DOCUMENTS
        #
        # Works for:
        # - normal PDF
        # - scanned PDF
        # - OCR PDF
        # - images
        # - plain text
        # - OCR extracted text
        # - other document types
        # --------------------------------------------------------

        return self._chunk_generic(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ============================================================
    # GENERIC DOCUMENT CHUNKING
    # ============================================================

    def _chunk_generic(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:

        if not text or not text.strip():
            return []

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        text = text.strip()

        # --------------------------------------------------------
        # Normalize line endings.
        # --------------------------------------------------------

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # --------------------------------------------------------
        # Normalize excessive blank lines.
        #
        # Preserve paragraph boundaries instead of destroying
        # the complete document structure.
        # --------------------------------------------------------

        text = re.sub(
            r"\n[ \t]*\n[ \t]*\n+",
            "\n\n",
            text,
        )

        # --------------------------------------------------------
        # First try paragraph-based chunking.
        #
        # This prevents normal documents from being cut in the
        # middle of a paragraph whenever possible.
        # --------------------------------------------------------

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

        if not paragraphs:
            return []

        chunks = []

        current_parts = []

        current_length = 0

        for paragraph in paragraphs:

            paragraph_length = len(
                paragraph
            )

            # ----------------------------------------------------
            # Normal paragraph fits.
            # ----------------------------------------------------

            if (
                current_parts
                and current_length
                + paragraph_length
                + 2
                > chunk_size
            ):

                chunk = "\n\n".join(
                    current_parts
                ).strip()

                if chunk:
                    chunks.append(
                        chunk
                    )

                # ------------------------------------------------
                # Create overlap from the end of the previous
                # chunk.
                # ------------------------------------------------

                overlap_text = (
                    self._get_overlap_text(
                        chunk=chunk,
                        overlap=chunk_overlap,
                    )
                )

                if overlap_text:

                    current_parts = [
                        overlap_text
                    ]

                    current_length = len(
                        overlap_text
                    )

                else:

                    current_parts = []

                    current_length = 0

            # ----------------------------------------------------
            # Paragraph itself is larger than chunk size.
            #
            # Split it safely by lines first.
            # ----------------------------------------------------

            if paragraph_length > chunk_size:

                if current_parts:

                    chunk = "\n\n".join(
                        current_parts
                    ).strip()

                    if chunk:
                        chunks.append(
                            chunk
                        )

                    current_parts = []

                    current_length = 0

                long_chunks = (
                    self._split_long_text(
                        text=paragraph,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )

                chunks.extend(
                    long_chunks
                )

                continue

            current_parts.append(
                paragraph
            )

            current_length += (
                paragraph_length
                + (
                    2
                    if len(current_parts) > 1
                    else 0
                )
            )

        # --------------------------------------------------------
        # Final chunk.
        # --------------------------------------------------------

        if current_parts:

            chunk = "\n\n".join(
                current_parts
            ).strip()

            if chunk:
                chunks.append(
                    chunk
                )

        return chunks

    # ============================================================
    # GENERIC OVERLAP
    # ============================================================

    @staticmethod
    def _get_overlap_text(
        chunk: str,
        overlap: int,
    ) -> str:

        if not chunk or overlap <= 0:
            return ""

        if len(chunk) <= overlap:
            return chunk

        overlap_text = chunk[
            -overlap:
        ]

        # --------------------------------------------------------
        # Prefer starting the overlap at a line boundary.
        # --------------------------------------------------------

        newline_position = (
            overlap_text.find("\n")
        )

        if (
            newline_position >= 0
            and newline_position
            < len(overlap_text) - 1
        ):

            overlap_text = (
                overlap_text[
                    newline_position + 1:
                ]
            )

        return overlap_text.strip()

    # ============================================================
    # LONG TEXT SPLITTING
    # ============================================================

    def _split_long_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:

        if not text or not text.strip():
            return []

        text = text.strip()

        # --------------------------------------------------------
        # Try line-aware splitting first.
        # --------------------------------------------------------

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) > 1:

            chunks = []

            current_lines = []

            current_length = 0

            for line in lines:

                line_length = len(
                    line
                )

                if (
                    current_lines
                    and current_length
                    + line_length
                    + 1
                    > chunk_size
                ):

                    chunk = "\n".join(
                        current_lines
                    ).strip()

                    if chunk:
                        chunks.append(
                            chunk
                        )

                    overlap_text = (
                        self._get_overlap_text(
                            chunk=chunk,
                            overlap=chunk_overlap,
                        )
                    )

                    if overlap_text:

                        current_lines = [
                            overlap_text
                        ]

                        current_length = len(
                            overlap_text
                        )

                    else:

                        current_lines = []

                        current_length = 0

                current_lines.append(
                    line
                )

                current_length += (
                    line_length
                    + (
                        1
                        if len(current_lines) > 1
                        else 0
                    )
                )

            if current_lines:

                chunk = "\n".join(
                    current_lines
                ).strip()

                if chunk:
                    chunks.append(
                        chunk
                    )

            return chunks

        # --------------------------------------------------------
        # Final fallback:
        # character-based splitting.
        # --------------------------------------------------------

        chunks = []

        start = 0

        text_length = len(
            text
        )

        while start < text_length:

            end = (
                start
                + chunk_size
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(
                    chunk
                )

            if end >= text_length:
                break

            start = (
                end
                - chunk_overlap
            )

        return chunks

    # ============================================================
    # VOTER LIST CHUNKING
    # ============================================================

    def _chunk_voter_list(
        self,
        text: str,
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        # --------------------------------------------------------
        # Normalize line endings.
        # --------------------------------------------------------

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # --------------------------------------------------------
        # OCR/MinerU can put the header and first voter on the
        # same line.
        #
        # Example:
        #
        # ... 86-எடப்பாடி 1 பெயர்: அழுதா
        #
        # becomes:
        #
        # ... 86-எடப்பாடி
        #
        # 1 பெயர்: அழுதா
        # --------------------------------------------------------

        inline_voter_pattern = re.compile(
            r"(?<![\d-])"
            r"(\d{1,4})"
            r"(?:\s+|[.)]\s*)"
            r"(?=(?:பெயர்|பெயர|name)\s*[:：])",
            re.IGNORECASE,
        )

        normalized_lines = []

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            matches = list(
                inline_voter_pattern.finditer(
                    line
                )
            )

            if matches:

                first_match = matches[0]

                prefix = (
                    line[
                        :first_match.start()
                    ].strip()
                )

                suffix = (
                    line[
                        first_match.start():
                    ].strip()
                )

                if prefix:
                    normalized_lines.append(
                        prefix
                    )

                if suffix:
                    normalized_lines.append(
                        suffix
                    )

                continue

            normalized_lines.append(
                line
            )

        lines = [
            line.strip()
            for line in normalized_lines
            if line.strip()
        ]

        if not lines:
            return []

        # --------------------------------------------------------
        # Detect actual voter starts.
        # --------------------------------------------------------

        voter_start_pattern = re.compile(
            r"^(\d{1,4})"
            r"(?:"
            r"\s+(?=(?:பெயர்|பெயர|name)\s*[:：])"
            r"|"
            r"[.)]\s*(?=(?:பெயர்|பெயர|name)\s*[:：])"
            r"|"
            r"$"
            r")",
            re.IGNORECASE,
        )

        records = []

        current_record = []

        current_number = None

        prefix_lines = []

        for line in lines:

            match = voter_start_pattern.match(
                line
            )

            if match:

                voter_number = int(
                    match.group(1)
                )

                if 1 <= voter_number <= 9999:

                    if current_record:

                        records.append(
                            (
                                current_number,
                                current_record,
                            )
                        )

                    current_number = (
                        voter_number
                    )

                    current_record = [
                        line
                    ]

                    continue

            # ----------------------------------------------------
            # Header before first voter.
            # ----------------------------------------------------

            if not current_record:

                prefix_lines.append(
                    line
                )

            else:

                current_record.append(
                    line
                )

        # --------------------------------------------------------
        # Final voter.
        # --------------------------------------------------------

        if current_record:

            records.append(
                (
                    current_number,
                    current_record,
                )
            )

        chunks = []

        # ========================================================
        # DOCUMENT HEADER
        # ========================================================

        if prefix_lines:

            header = "\n".join(
                prefix_lines
            ).strip()

            if header:

                chunks.append(
                    header
                )

        # ========================================================
        # VOTER RECORDS
        # ========================================================

        for (
            voter_number,
            record_lines,
        ) in records:

            record_text = "\n".join(
                record_lines
            ).strip()

            if record_text:

                chunks.append(
                    record_text
                )

        # ========================================================
        # FALLBACK
        # ========================================================

        if not records:

            return self._chunk_generic(
                text=text,
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
            )

        return chunks