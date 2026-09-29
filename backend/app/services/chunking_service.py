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

        if document_type == "voter_list":
            return self._chunk_voter_list(text=text)

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

    # ============================================================
    # VOTER LIST CHUNKING
    # ============================================================

    def _chunk_voter_list(self, text: str) -> list[str]:

        text = text.strip()

        if not text:
            return []

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return []

        # --------------------------------------------------------
        # IMPORTANT
        # --------------------------------------------------------
        #
        # A voter record normally starts like:
        #
        # 1
        # 2
        # 10
        # 11 Name...
        # 12. Name...
        # 13) Name...
        #
        # But document headers can contain:
        #
        # 86-எடப்பாடி
        # 1-இருப்பாளி
        # 22-01-2024
        #
        # Therefore "-" MUST NOT be treated as a voter separator.
        #
        voter_start_pattern = re.compile(
            r"^(\d{1,4})(?:\s+|[.)]\s*|$)"
        )

        records = []

        current_record = []
        current_number = None

        prefix_lines = []

        for line in lines:

            match = voter_start_pattern.match(line)

            if match:

                voter_number = int(match.group(1))

                if 1 <= voter_number <= 9999:

                    # Save previous voter record
                    if current_record:
                        records.append(
                            (
                                current_number,
                                current_record,
                            )
                        )

                    current_number = voter_number

                    current_record = [line]

                    continue

            # ----------------------------------------------------
            # Before the first voter record
            # ----------------------------------------------------

            if not current_record:

                prefix_lines.append(line)

            else:

                current_record.append(line)

        # Save final voter record
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

            header = "\n".join(prefix_lines).strip()

            if header:

                chunks.append(header)

        # ========================================================
        # VOTER RECORDS
        # ========================================================

        for voter_number, record_lines in records:

            record_text = "\n".join(
                record_lines
            ).strip()

            if record_text:

                chunks.append(record_text)

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