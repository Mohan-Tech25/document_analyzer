import re


class VoterRecordService:

    # ============================================================
    # PARSE A SINGLE VOTER CHUNK
    # ============================================================

    def parse_record(self, text: str) -> dict:

        if not text or not text.strip():
            return {}

        text = text.strip()

        record = {
            "serial_number": None,
            "name": None,
            "relation_type": None,
            "relation_name": None,
            "house_number": None,
            "age": None,
            "gender": None,
            "epic_numbers": [],
            "raw_text": text,
        }

        # ========================================================
        # SERIAL NUMBER
        # ========================================================

        # Normal format:
        #
        # 35
        # Name: ...
        #
        serial_match = re.search(
            r"^\s*(\d{1,4})\s*$",
            text,
            re.MULTILINE,
        )

        # OCR/chunk format:
        #
        # 35 Name: ...
        #
        if not serial_match:
            serial_match = re.search(
                r"^\s*(\d{1,4})(?=\s+)",
                text,
                re.MULTILINE,
            )

        # Another possible format:
        #
        # 35
        # பெயர்: ...
        #
        if not serial_match:
            serial_match = re.search(
                r"\b(\d{1,4})\b"
                r"\s*(?=(?:பெயர்|Name)\s*:?)",
                text,
                re.MULTILINE | re.IGNORECASE,
            )

        if serial_match:
            record["serial_number"] = int(
                serial_match.group(1)
            )

        # ========================================================
        # NAME
        # ========================================================

        name_match = re.search(
            r"(?:பெயர்|Name)"
            r"\s*:\s*(.+?)(?:\s*-\s*)?$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )

        if name_match:
            record["name"] = self._clean_value(
                name_match.group(1)
            )

        # ========================================================
        # RELATION
        # ========================================================

        relation_patterns = [

            (
                "father",
                r"(?:தந்தையின் பெயர்|"
                r"தந்தையின் பெ்யர்|"
                r"Father(?:'s)? Name)"
                r"\s*:\s*(.+?)(?:\s*-\s*)?$",
            ),

            (
                "husband",
                r"(?:கணவர் பெயர்|"
                r"Husband(?:'s)? Name)"
                r"\s*:\s*(.+?)(?:\s*-\s*)?$",
            ),

            (
                "mother",
                r"(?:தாயின் பெயர்|"
                r"Mother(?:'s)? Name)"
                r"\s*:\s*(.+?)(?:\s*-\s*)?$",
            ),

            (
                "other",
                r"(?:இதரர் பெயர்|"
                r"Other Name)"
                r"\s*:\s*(.+?)(?:\s*-\s*)?$",
            ),
        ]

        for relation_type, pattern in relation_patterns:

            relation_match = re.search(
                pattern,
                text,
                re.MULTILINE | re.IGNORECASE,
            )

            if relation_match:

                record["relation_type"] = relation_type

                record["relation_name"] = (
                    self._clean_value(
                        relation_match.group(1)
                    )
                )

                break

        # ========================================================
        # HOUSE NUMBER
        # ========================================================

        house_match = re.search(
            r"(?:வீட்டு\s*எண்|"
            r"வீட்டு\s*எனன்|"
            r"வீட்டு\s*எrஎன|"
            r"House\s*No)"
            r"\s*[:;]?\s*(.+?)(?:\s*$)",
            text,
            re.MULTILINE | re.IGNORECASE,
        )

        if house_match:

            house_value = (
                house_match.group(1)
                .strip()
            )

            house_value = re.sub(
                r"\b(?:Photo is|available)\b",
                "",
                house_value,
                flags=re.IGNORECASE,
            )

            record["house_number"] = (
                self._clean_value(
                    house_value
                )
            )

        # ========================================================
        # AGE
        # ========================================================

        age_matches = re.findall(
            r"(?:வயது|வய்து|Age)"
            r"\s*[:\-]?\s*(\d{1,3})",
            text,
            re.IGNORECASE,
        )

        if age_matches:

            valid_ages = []

            for value in age_matches:

                age = int(value)

                if 1 <= age <= 120:
                    valid_ages.append(age)

            if valid_ages:

                record["age"] = valid_ages[0]

                if len(valid_ages) > 1:

                    record["age_candidates"] = (
                        valid_ages
                    )

                    record["possible_ocr_conflict"] = True

        # ========================================================
        # GENDER
        # ========================================================

        gender_match = re.search(
            r"(?:பாலினம்|"
            r"பொலினம்|"
            r"பொாலினம்|"
            r"Gender)"
            r"\s*[:\-]?\s*"
            r"(ஆண்|பெண்|பெனன்|பென்|Male|Female)",
            text,
            re.IGNORECASE,
        )

        if gender_match:

            gender = (
                gender_match.group(1)
                .strip()
            )

            if gender.lower() == "female":

                record["gender"] = "Female"

            elif gender.lower() == "male":

                record["gender"] = "Male"

            elif gender in {
                "பெண்",
                "பெனன்",
                "பென்",
            }:

                record["gender"] = "Female"

            elif gender == "ஆண்":

                record["gender"] = "Male"

        # ========================================================
        # EPIC NUMBERS
        # ========================================================

        epic_matches = re.findall(
            r"\b[A-Z]{3}\d{7}\b",
            text.upper(),
        )

        record["epic_numbers"] = list(
            dict.fromkeys(
                epic_matches
            )
        )

        if len(record["epic_numbers"]) > 1:

            record["possible_ocr_conflict"] = True

        # ========================================================
        # RETURN RECORD
        # ========================================================

        return record

    # ============================================================
    # PARSE ALL VOTER CHUNKS
    # ============================================================

    def parse_records(
        self,
        chunks: list[str],
    ) -> list[dict]:

        records = []

        for chunk in chunks:

            record = self.parse_record(
                chunk
            )

            if not record:
                continue

            # ----------------------------------------------------
            # DO NOT REQUIRE SERIAL NUMBER
            #
            # OCR can sometimes fail to detect the serial number
            # even though the chunk contains a valid voter.
            #
            # DO NOT ACCEPT "NAME ALONE" AS PROOF OF A VOTER
            #
            # Voter-list documents also contain non-voter lines
            # such as constituency/page headers
            # ("... எண் மற்றும் பெயர் : 86-...") that reuse the
            # same word ("பெயர்" / "Name") the name regex looks
            # for, just with a different meaning ("constituency
            # name" rather than "person's name"). Those lines
            # then spuriously match the name pattern even though
            # they are not a voter entry.
            #
            # A genuine voter chunk always carries at least one
            # additional structured field beyond name — a serial
            # number, age, gender, house number, relation, or
            # EPIC number. A chunk where name is the ONLY thing
            # that matched is far more likely to be incidental
            # text than an actual voter record, so name alone is
            # no longer treated as sufficient on its own.
            # ----------------------------------------------------

            has_voter_data = any([
                record.get("serial_number") is not None,
                record.get("relation_name"),
                record.get("house_number"),
                record.get("age") is not None,
                record.get("gender"),
                bool(record.get("epic_numbers")),
            ])

            if not has_voter_data:
                continue

            records.append(record)

        # ========================================================
        # DO NOT RE-SORT BY SERIAL NUMBER.
        #
        # Records are kept in the order their chunks were
        # retrieved, which is already the true document/OCR
        # reading order (ChunkRepository.get_by_document orders
        # by chunk_index).
        #
        # Previously this method sorted records by
        # "serial_number", pushing any record whose serial
        # number OCR failed to read to the very end of the
        # list. Because "_record_position" (assigned by the
        # caller from this list's order) is what "the Nth
        # person" / "position N" resolves against, a single
        # misread or missing serial number anywhere in the
        # document would silently shift every later position
        # number away from the voter actually printed at that
        # position in the document. Preserving the original
        # chunk order makes "position" deterministic and tied
        # to the document itself, independent of how reliably
        # OCR read any individual serial number.
        #
        # "serial" lookups are unaffected by this change: they
        # already match on the "serial_number" field directly,
        # not on list order.
        # ========================================================

        return records

    # ============================================================
    # PARSE DATABASE CHUNK ROWS
    # ============================================================

    def parse_chunk_rows(
        self,
        rows: list[tuple],
    ) -> list[dict]:

        chunks = []

        for row in rows:

            # Expected database row:
            #
            # id
            # chunk_index
            # content
            # page_number
            # created_at

            if len(row) < 3:
                continue

            content = row[2]

            if not content:
                continue

            chunks.append(content)

        return self.parse_records(
            chunks
        )

    # ============================================================
    # CLEAN OCR VALUE
    # ============================================================

    def _clean_value(
        self,
        value: str,
    ) -> str:

        if not value:
            return ""

        value = value.strip()

        # Normalize multiple spaces
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        # Remove unwanted surrounding punctuation
        value = value.strip(
            " -:;"
        )

        return value


# ================================================================
# SINGLETON INSTANCE
# ================================================================

voter_record_service = VoterRecordService()