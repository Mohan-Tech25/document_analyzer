import re
from typing import Any


class VoterRecordService:
    """
    Converts voter-list database chunks into structured
    voter records.

    Expected voter-list chunk format:

        Chunk 0:
            header / document information

        Chunk 1:
            1
            பெயர்: ...
            தாயின் பெயர்: ...
            வீட்டு எண் : ...
            வயது : ...
            பாலினம் : ...

        Chunk 2:
            2
            பெயர்: ...
            ...

    IMPORTANT:

    Each voter must be parsed only from its own chunk.

    We must NEVER search neighboring chunks or nearby
    document content for an EPIC number because that can
    incorrectly assign one voter's EPIC to another voter.
    """

    # ============================================================
    # PATTERNS
    # ============================================================

    SERIAL_PATTERN = re.compile(
        r"^\s*(\d{1,4})"
        r"(?:\s*$|\s+(?=(?:பெயர்|பெயர|name)\s*[:：])"
        r"|[.)]\s*(?=(?:பெயர்|பெயர|name)\s*[:：]))",
        flags=re.IGNORECASE,
    )

    EPIC_PATTERN = re.compile(
        r"\b[A-Z]{3}\d{7}\b",
        flags=re.IGNORECASE,
    )

    # ============================================================
    # NORMALIZE CONTENT
    # ============================================================

    def _normalize_content(
        self,
        content: str,
    ) -> str:
        """
        Normalize text stored in PostgreSQL.

        Converts escaped newline representations into
        real newlines and normalizes line endings.
        """

        if content is None:
            return ""

        text = str(content)

        text = text.replace(
            "\\\r\n",
            "\n",
        )

        text = text.replace(
            "\\\n",
            "\n",
        )

        text = text.replace(
            "\\\r",
            "\n",
        )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        return text.strip()

    # ============================================================
    # EXTRACT SERIAL NUMBER
    # ============================================================

    def _extract_serial_number(
        self,
        content: str,
    ) -> int | None:
        """
        Extract the serial number from the beginning
        of a voter chunk.

        Examples:

            45
            45 பெயர்: நந்தகுமார்
            45. பெயர்: நந்தகுமார்
            45) பெயர்: நந்தகுமார்

        Header chunks that do not begin with a valid
        voter serial return None.
        """

        if not content:
            return None

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        # --------------------------------------------------------
        # First line
        # --------------------------------------------------------

        first_line = lines[0]

        match = self.SERIAL_PATTERN.match(
            first_line
        )

        if match:

            try:
                return int(
                    match.group(1)
                )

            except ValueError:
                return None

        # --------------------------------------------------------
        # Fallback
        #
        # Some OCR output may place the serial and
        # name together in a slightly different form.
        # --------------------------------------------------------

        fallback_pattern = re.compile(
            r"^\s*(\d{1,4})"
            r"(?:\s+|[.)]\s*)"
            r"(?=(?:பெயர்|பெயர|name)\s*[:：])",
            flags=re.IGNORECASE,
        )

        fallback_match = (
            fallback_pattern.match(
                first_line
            )
        )

        if fallback_match:

            try:
                return int(
                    fallback_match.group(1)
                )

            except ValueError:
                return None

        return None

    # ============================================================
    # REMOVE SERIAL PREFIX
    # ============================================================

    def _remove_serial_prefix(
        self,
        text: str,
    ) -> str:
        """
        Remove the serial number from the beginning
        of a voter chunk.

        Example:

            45
            பெயர்: நந்தகுமார்

        becomes:

            பெயர்: நந்தகுமார்
        """

        if not text:
            return ""

        lines = text.splitlines()

        if not lines:
            return text.strip()

        first_line = lines[0].strip()

        # --------------------------------------------------------
        # A line containing only the serial
        # --------------------------------------------------------

        if re.fullmatch(
            r"\d{1,4}",
            first_line,
        ):

            return "\n".join(
                lines[1:]
            ).strip()

        # --------------------------------------------------------
        # Serial + name on same line
        # --------------------------------------------------------

        first_line_without_serial = re.sub(
            r"^\s*\d{1,4}"
            r"(?:[.)]\s*|\s+)",
            "",
            first_line,
            count=1,
        ).strip()

        if first_line_without_serial != first_line:

            lines[0] = (
                first_line_without_serial
            )

        return "\n".join(
            lines
        ).strip()

    # ============================================================
    # EXTRACT EPIC NUMBERS
    # ============================================================

    def _extract_epic_numbers(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract EPIC numbers ONLY from the current
        voter's own chunk.

        No neighboring chunk or reconstructed
        document content is inspected.
        """

        if not text:
            return []

        matches = self.EPIC_PATTERN.findall(
            text
        )

        return list(
            dict.fromkeys(
                match.upper()
                for match in matches
            )
        )

    # ============================================================
    # CLEAN FIELD VALUE
    # ============================================================

    def _clean_field_value(
        self,
        value: str | None,
    ) -> str | None:
        """
        Clean an extracted field value.
        """

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        # Remove common trailing OCR separators.
        value = value.strip(
            " -–—:："
        )

        if not value:
            return None

        return value

    # ============================================================
    # EXTRACT FIELD
    # ============================================================

    def _extract_field(
        self,
        text: str,
        patterns: list[str],
    ) -> str | None:
        """
        Extract a field using multiple possible OCR
        labels.

        The search is restricted to the current
        voter chunk.
        """

        if not text:
            return None

        for pattern_text in patterns:

            pattern = re.compile(
                pattern_text,
                flags=(
                    re.IGNORECASE
                    | re.MULTILINE
                ),
            )

            match = pattern.search(
                text
            )

            if not match:
                continue

            value = match.group(1)

            value = self._clean_field_value(
                value
            )

            if value:
                return value

        return None

    # ============================================================
    # EXTRACT NAME
    # ============================================================

    def _extract_name(
        self,
        text: str,
    ) -> str | None:
        """
        Extract voter name.

        Supported labels include:

            பெயர்:
            பெயர:
            Name:
        """

        return self._extract_field(
            text,
            [
                r"^\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                r"^\s*பெயர\s*[:：]\s*(.+?)\s*(?=\n|$)",
                r"^\s*name\s*[:：]\s*(.+?)\s*(?=\n|$)",
            ],
        )

    # ============================================================
    # EXTRACT RELATION TYPE + NAME
    # ============================================================

    def _extract_relation(
        self,
        text: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract relation type and relation name.

        Examples:

            தந்தையின் பெயர்: அய்யந்துரை
            தாயின் பெயர்: உண்ணாமலை
            கணவர் பெயர்: பரந்தாமன்

        Returns:

            (
                relation_type,
                relation_name,
            )
        """

        patterns = [
            (
                r"^\s*தந்தையின்\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "தந்தை",
            ),
            (
                r"^\s*தந்தை\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "தந்தை",
            ),
            (
                r"^\s*தாயின்\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "தாய்",
            ),
            (
                r"^\s*தாய்\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "தாய்",
            ),
            (
                r"^\s*கணவர்\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "கணவர்",
            ),
            (
                r"^\s*கணவரின்\s*பெயர்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "கணவர்",
            ),
            (
                r"^\s*father(?:'s)?\s*name\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "father",
            ),
            (
                r"^\s*mother(?:'s)?\s*name\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "mother",
            ),
            (
                r"^\s*husband(?:'s)?\s*name\s*[:：]\s*(.+?)\s*(?=\n|$)",
                "husband",
            ),
        ]

        for pattern_text, relation_type in patterns:

            match = re.search(
                pattern_text,
                text,
                flags=(
                    re.IGNORECASE
                    | re.MULTILINE
                ),
            )

            if not match:
                continue

            relation_name = (
                self._clean_field_value(
                    match.group(1)
                )
            )

            if relation_name:

                return (
                    relation_type,
                    relation_name,
                )

        return (
            None,
            None,
        )

    # ============================================================
    # EXTRACT HOUSE NUMBER
    # ============================================================

    def _extract_house_number(
        self,
        text: str,
    ) -> str | None:
        """
        Extract house number.

        Supports OCR variations such as:

            வீட்டு எண் :
            வீட்டு எண்:
            House No:
            House Number:
        """

        value = self._extract_field(
            text,
            [
                r"^\s*வீட்டு\s*எண்\s*[:：]\s*(.+?)\s*(?=\n|$)",
                r"^\s*வீட்டு\s*எண்\s*[:：]?\s*(.+?)\s*(?=\n|$)",
                r"^\s*house\s*(?:no|number)\s*[:：]\s*(.+?)\s*(?=\n|$)",
                r"^\s*house\s*(?:no|number)\s*[:：]?\s*(.+?)\s*(?=\n|$)",
            ],
        )

        if value is None:
            return None

        # --------------------------------------------------------
        # If OCR accidentally placed age on the same line,
        # keep only the house number portion.
        # --------------------------------------------------------

        value = re.split(
            r"\s+வயது\s*[:：]",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        value = re.split(
            r"\s+வய்து\s*[:：]",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        value = re.split(
            r"\s+age\s*[:：]",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        return self._clean_field_value(
            value
        )

    # ============================================================
    # EXTRACT AGE
    # ============================================================

    def _extract_age(
        self,
        text: str,
    ) -> int | None:
        """
        Extract age.

        Supports OCR spelling:

            வயது
            வய்து
            age
        """

        patterns = [
            r"^\s*வயது\s*[:：]\s*(\d{1,3})\b",
            r"^\s*வய்து\s*[:：]\s*(\d{1,3})\b",
            r"^\s*age\s*[:：]\s*(\d{1,3})\b",
        ]

        for pattern_text in patterns:

            match = re.search(
                pattern_text,
                text,
                flags=(
                    re.IGNORECASE
                    | re.MULTILINE
                ),
            )

            if not match:
                continue

            try:

                return int(
                    match.group(1)
                )

            except ValueError:

                return None

        return None

    # ============================================================
    # EXTRACT GENDER
    # ============================================================

    def _extract_gender(
        self,
        text: str,
    ) -> str | None:
        """
        Extract gender.
        """

        patterns = [
            r"^\s*பாலினம்\s*[:：]\s*(.+?)\s*(?=\n|$)",
            r"^\s*பாலினம்\s*[:：]?\s*(.+?)\s*(?=\n|$)",
            r"^\s*gender\s*[:：]\s*(.+?)\s*(?=\n|$)",
        ]

        value = self._extract_field(
            text,
            patterns,
        )

        if value is None:
            return None

        # --------------------------------------------------------
        # Keep only known gender values when they appear
        # at the beginning of an OCR-contaminated value.
        # --------------------------------------------------------

        gender_match = re.match(
            r"^(பெண்|ஆண்|female|male)\b",
            value,
            flags=re.IGNORECASE,
        )

        if gender_match:

            return gender_match.group(
                1
            )

        return value

    # ============================================================
    # BUILD STRUCTURED RECORD
    # ============================================================

    def _build_record(
        self,
        content: str,
        page_number: int | None = None,
    ) -> dict | None:
        """
        Convert one voter chunk into a structured
        voter record.

        Header chunks return None.
        """

        content = self._normalize_content(
            content
        )

        if not content:
            return None

        serial_number = (
            self._extract_serial_number(
                content
            )
        )

        # --------------------------------------------------------
        # Header / non-voter chunk
        # --------------------------------------------------------

        if serial_number is None:
            return None

        voter_text = (
            self._remove_serial_prefix(
                content
            )
        )

        # --------------------------------------------------------
        # Extract fields only from this voter chunk
        # --------------------------------------------------------

        name = self._extract_name(
            voter_text
        )

        relation_type, relation_name = (
            self._extract_relation(
                voter_text
            )
        )

        house_number = (
            self._extract_house_number(
                voter_text
            )
        )

        age = self._extract_age(
            voter_text
        )

        gender = self._extract_gender(
            voter_text
        )

        epic_numbers = (
            self._extract_epic_numbers(
                voter_text
            )
        )

        return {
            "serial_number": serial_number,
            "text": content,
            "page_number": page_number,
            "epic_numbers": epic_numbers,
            "name": name,
            "relation_type": relation_type,
            "relation_name": relation_name,
            "house_number": house_number,
            "age": age,
            "gender": gender,
        }

    # ============================================================
    # PARSE DATABASE CHUNK ROWS
    # ============================================================

    def parse_chunk_rows(
        self,
        rows: list[tuple],
    ) -> list[dict]:
        """
        Parse voter-list database chunks.

        New storage format:

            one header chunk
            one voter per chunk

        We intentionally do NOT concatenate all chunks.

        This is critical because each voter must remain
        isolated from neighboring voters.
        """

        records: list[dict] = []

        print(
            "\n========== VOTER CHUNK PARSING =========="
        )

        print(
            "DATABASE CHUNK ROWS:",
            len(rows),
        )

        # --------------------------------------------------------
        # Process each chunk independently
        # --------------------------------------------------------

        for row_index, row in enumerate(rows):

            if len(row) < 3:
                continue

            content = row[2]

            if content is None:
                continue

            content = self._normalize_content(
                str(content)
            )

            if not content:
                continue

            # ----------------------------------------------------
            # Determine page number if available.
            #
            # Existing repository rows use the content at
            # row[2]. We keep compatibility with that structure.
            #
            # If a fourth column exists and is an integer,
            # use it as page number.
            # ----------------------------------------------------

            page_number: int | None = None

            if len(row) > 3:

                possible_page = row[3]

                if isinstance(
                    possible_page,
                    int,
                ):
                    page_number = (
                        possible_page
                    )

            # ----------------------------------------------------
            # Build record from THIS chunk only
            # ----------------------------------------------------

            record = self._build_record(
                content=content,
                page_number=page_number,
            )

            # ----------------------------------------------------
            # Header / non-voter chunk
            # ----------------------------------------------------

            if record is None:

                print(
                    f"CHUNK {row_index}: "
                    "HEADER / NON-VOTER"
                )

                continue

            serial = record.get(
                "serial_number"
            )

            print(
                f"CHUNK {row_index}: "
                f"SERIAL={serial} "
                f"EPIC={record.get('epic_numbers', [])}"
            )

            records.append(
                record
            )

        # ========================================================
        # Remove duplicate serials
        # ========================================================

        final_records: list[dict] = []

        seen_serials: set[Any] = set()

        duplicate_serials: list[int] = []

        for record in records:

            serial = record.get(
                "serial_number"
            )

            if serial is None:
                continue

            if serial in seen_serials:

                duplicate_serials.append(
                    serial
                )

                continue

            seen_serials.add(
                serial
            )

            final_records.append(
                record
            )

        # ========================================================
        # Sort by serial number
        # ========================================================

        final_records.sort(
            key=lambda record: (
                record.get(
                    "serial_number"
                )
                if record.get(
                    "serial_number"
                ) is not None
                else 999999
            )
        )

        # ========================================================
        # Final parser debug
        # ========================================================

        parsed_serials = [
            record.get(
                "serial_number"
            )
            for record in final_records
            if record.get(
                "serial_number"
            ) is not None
        ]

        print(
            "\n========== PARSER FINAL DEBUG =========="
        )

        print(
            "TOTAL PARSED RECORDS:",
            len(final_records),
        )

        print(
            "PARSED SERIALS:",
            parsed_serials,
        )

        # --------------------------------------------------------
        # Missing serial detection
        # --------------------------------------------------------

        if parsed_serials:

            minimum_serial = min(
                parsed_serials
            )

            maximum_serial = max(
                parsed_serials
            )

            expected = set(
                range(
                    minimum_serial,
                    maximum_serial + 1,
                )
            )

            actual = set(
                parsed_serials
            )

            missing = sorted(
                expected - actual
            )

        else:

            missing = []

        # --------------------------------------------------------
        # Duplicate serial detection
        # --------------------------------------------------------

        duplicates = sorted(
            set(
                duplicate_serials
            )
        )

        # --------------------------------------------------------
        # Unexpected serial detection
        # --------------------------------------------------------

        unexpected = sorted(
            serial
            for serial in parsed_serials
            if serial < 1
            or serial > 9999
        )

        print(
            "MISSING SERIALS:",
            missing,
        )

        print(
            "DUPLICATE SERIALS:",
            duplicates,
        )

        print(
            "UNEXPECTED SERIALS:",
            unexpected,
        )

        # ========================================================
        # EPIC DEBUG
        # ========================================================

        print(
            "\n========== EPIC EXTRACTION RESULT =========="
        )

        for record in final_records:

            print(
                f"SERIAL={record.get('serial_number')} "
                f"EPIC={record.get('epic_numbers', [])}"
            )

        print(
            "============================================"
        )

        print(
            "========================================\n"
        )

        return final_records


# ================================================================
# MODULE-LEVEL SERVICE INSTANCE
#
# Required by:
#
# from app.services.voter_record_service import voter_record_service
# ================================================================

voter_record_service = VoterRecordService()