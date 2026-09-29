import json
import re
import unicodedata

from app.core.exceptions import ChatException
from app.services.ollama_service import OllamaService


class VoterQueryService:

    def __init__(
        self,
        ollama_service: OllamaService,
    ):
        self.ollama_service = ollama_service

    # =========================================================
    # UNDERSTAND QUERY
    # =========================================================

    def understand_query(
        self,
        question: str,
    ) -> dict:

        if not question or not question.strip():
            return {}

        prompt = f"""
You are a query-understanding system for a document analyzer.

The current document is a voter-list document.

Your job is to understand the user's question and return ONLY
valid JSON.

You are NOT answering the question.

You must classify the question into exactly ONE lookup_type.

Allowed lookup_type values:

"name"
"epic"
"serial"
"position"
"count"
"document"
"all"

------------------------------------------------------------
LOOKUP TYPE: name
------------------------------------------------------------

Use "name" when the user identifies a voter by their name.

Examples:

"Tell me about Nandhakumar"
"Who is Nandhakumar?"
"What is the age of Nandhakumar?"

------------------------------------------------------------
LOOKUP TYPE: epic
------------------------------------------------------------

Use "epic" when the user provides a Voter ID / EPIC number.

Examples:

"What is the name of RMK0799270?"
"Tell me about EPIC RMK0799270"
"Who has voter ID RMK0799270?"

IMPORTANT:

A Voter ID / EPIC number is NOT a serial number.

------------------------------------------------------------
LOOKUP TYPE: serial
------------------------------------------------------------

Use "serial" when the user explicitly refers to a voter
serial number.

Examples:

"What is the name of serial number 11?"
"What is the Voter ID of serial number 11?"
"Tell me about serial 45"
"Age of serial number 56"

The numeric value refers to the actual serial_number field.

------------------------------------------------------------
LOOKUP TYPE: position
------------------------------------------------------------

Use "position" when the user explicitly refers to the
record's position/order in the extracted voter records.

Examples:

"What is the voter at position 45?"
"Who is in position 10?"

Do NOT confuse position with Voter ID.

------------------------------------------------------------
LOOKUP TYPE: count
------------------------------------------------------------

Use "count" when the user asks for the number of voters
or total voters.

Examples:

"How many voters are there?"
"What is the total number of voters?"
"How many voters are on this list?"

------------------------------------------------------------
LOOKUP TYPE: document
------------------------------------------------------------

Use "document" when the question is about the document
itself rather than a specific voter record.

Examples:

"What is the constituency name of this document?"
"What is the constituency number?"
"What is the part number?"
"When was the voter list published?"
"What is the section name?"
"How many pages does this document have?"
"What is this document about?"
"What type of document is this?"
"What is the document date?"

For document questions, do NOT try to find a voter record.

The answer will be obtained from the actual extracted
document content.

------------------------------------------------------------
LOOKUP TYPE: all
------------------------------------------------------------

Use "all" when the user asks for information about all
voter records.

Examples:

"Show all voters"
"List all voters"
"Give me all voter details"

------------------------------------------------------------
REQUESTED FIELDS
------------------------------------------------------------

For voter questions, requested_fields must contain ONLY
the information the user actually wants.

Allowed voter fields:

"name"
"age"
"gender"
"house_number"
"relation_type"
"relation_name"
"epic_numbers"
"serial_number"
"all"

Examples:

Question:
"What is the Voter ID of serial number 11?"

Return:

{{
    "lookup_type": "serial",
    "value": 11,
    "requested_fields": ["epic_numbers"]
}}

Question:
"What is the age of serial number 11?"

Return:

{{
    "lookup_type": "serial",
    "value": 11,
    "requested_fields": ["age"]
}}

Question:
"Tell me about serial number 11"

Return:

{{
    "lookup_type": "serial",
    "value": 11,
    "requested_fields": ["all"]
}}

Question:
"What is the name of Nandhakumar?"

Return:

{{
    "lookup_type": "name",
    "value": "Nandhakumar",
    "requested_fields": ["name"]
}}

Question:
"How many voters are there?"

Return:

{{
    "lookup_type": "count",
    "value": null,
    "requested_fields": ["all"]
}}

------------------------------------------------------------
DOCUMENT QUESTIONS
------------------------------------------------------------

For document-level questions use:

{{
    "lookup_type": "document",
    "value": null,
    "requested_fields": ["all"]
}}

Examples:

Question:
"What is the constituency name?"

Return:

{{
    "lookup_type": "document",
    "value": null,
    "requested_fields": ["all"]
}}

Question:
"What is the part number?"

Return:

{{
    "lookup_type": "document",
    "value": null,
    "requested_fields": ["all"]
}}

Question:
"When was this voter list published?"

Return:

{{
    "lookup_type": "document",
    "value": null,
    "requested_fields": ["all"]
}}

Question:
"Summarize this document"

Return:

{{
    "lookup_type": "document",
    "value": null,
    "requested_fields": ["all"]
}}

------------------------------------------------------------
IMPORTANT RULES
------------------------------------------------------------

1. Return ONLY JSON.
2. Do not include markdown.
3. Do not answer the user's question.
4. Do not invent voter values.
5. Never treat a serial number as a Voter ID.
6. Never treat a position as a Voter ID.
7. Voter ID means EPIC/Voter ID only.
8. If the question is about the document itself, use
   lookup_type "document".
9. If the question identifies a specific voter, use the
   appropriate voter lookup type.
10. "Tell me about" a specific voter means requested_fields
    ["all"].
11. "How many voters" means lookup_type "count".

USER QUESTION:
{question}
"""

        try:

            response = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

            print(
                "\n========== QUERY UNDERSTANDING =========="
            )

            print(
                f"QUESTION: {question}"
            )

            print(
                f"RAW RESPONSE: {response}"
            )

            print(
                "=========================================\n"
            )

            if not response:
                return {}

            return self._parse_query_response(
                response=response,
                question=question,
            )

        except ChatException:
            raise

        except Exception as exc:

            raise ChatException(
                f"Failed to understand voter query: {exc}"
            ) from exc

    # =========================================================
    # PARSE QUERY RESPONSE
    # =========================================================

    def _parse_query_response(
        self,
        response: str,
        question: str,
    ) -> dict:

        if not response:
            return {}

        cleaned = response.strip()

        # -----------------------------------------------------
        # REMOVE MARKDOWN CODE FENCES
        # -----------------------------------------------------

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        # -----------------------------------------------------
        # EXTRACT JSON OBJECT
        # -----------------------------------------------------

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:
            return {}

        json_text = match.group(0)

        try:

            data = json.loads(
                json_text
            )

        except json.JSONDecodeError:

            print(
                "DEBUG: Failed to parse query JSON:"
            )

            print(
                json_text
            )

            return {}

        if not isinstance(
            data,
            dict,
        ):
            return {}

        lookup_type = self._normalize_lookup_type(
            data.get("lookup_type")
        )

        value = self._normalize_value(
            lookup_type=lookup_type,
            value=data.get("value"),
        )

        requested_fields = (
            self._normalize_requested_fields(
                data.get("requested_fields")
            )
        )

        # -----------------------------------------------------
        # DOCUMENT QUESTIONS
        # -----------------------------------------------------

        if lookup_type == "document":

            return {
                "lookup_type": "document",
                "value": None,
                "requested_fields": ["all"],
            }

        # -----------------------------------------------------
        # COUNT
        # -----------------------------------------------------

        if lookup_type == "count":

            return {
                "lookup_type": "count",
                "value": None,
                "requested_fields": ["all"],
            }

        # -----------------------------------------------------
        # VALIDATE FIELDS AGAINST QUESTION
        # -----------------------------------------------------

        requested_fields = (
            self._validate_requested_fields_against_question(
                requested_fields=requested_fields,
                question=question,
            )
        )

        return {
            "lookup_type": lookup_type,
            "value": value,
            "requested_fields": requested_fields,
        }

    # =========================================================
    # VALIDATE REQUESTED FIELDS
    # =========================================================

    def _validate_requested_fields_against_question(
        self,
        requested_fields: list[str],
        question: str,
    ) -> list[str]:

        fields = list(
            requested_fields
        )

        question_normalized = (
            self._normalize_text(
                question
            )
        )

        # -----------------------------------------------------
        # VOTER ID / EPIC
        # -----------------------------------------------------

        epic_terms = [
            "voter id",
            "voterid",
            "epic",
            "epic number",
            "epic no",
            "epic number",
        ]

        asks_epic = any(
            term in question_normalized
            for term in epic_terms
        )

        if asks_epic:

            fields = [
                field
                for field in fields
                if field
                not in {
                    "serial_number",
                    "position",
                }
            ]

            if "epic_numbers" not in fields:

                fields.append(
                    "epic_numbers"
                )

        # -----------------------------------------------------
        # SERIAL NUMBER
        # -----------------------------------------------------

        serial_terms = [
            "serial number",
            "serial no",
            "serial",
        ]

        asks_serial = any(
            term in question_normalized
            for term in serial_terms
        )

        if asks_serial:

            if "epic" not in question_normalized:

                if "voter id" not in question_normalized:

                    if "epic number" not in question_normalized:

                        if "serial_number" not in fields:

                            fields.append(
                                "serial_number"
                            )

        # -----------------------------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------------------------

        result = []

        for field in fields:

            if field not in result:

                result.append(
                    field
                )

        if not result:

            return ["name"]

        return result

    # =========================================================
    # NORMALIZE LOOKUP TYPE
    # =========================================================

    def _normalize_lookup_type(
        self,
        lookup_type,
    ) -> str:

        if lookup_type is None:
            return "name"

        value = (
            str(lookup_type)
            .lower()
            .strip()
        )

        aliases = {

            "name": "name",
            "voter_name": "name",
            "person_name": "name",

            "epic": "epic",
            "voter_id": "epic",
            "voterid": "epic",
            "epic_number": "epic",

            "serial": "serial",
            "serial_number": "serial",
            "serial_no": "serial",

            "position": "position",
            "index": "position",
            "record_position": "position",

            "count": "count",
            "total": "count",
            "number": "count",

            "document": "document",
            "document_level": "document",
            "document_info": "document",
            "metadata": "document",
            "document_metadata": "document",

            "all": "all",
            "list": "all",
            "records": "all",
        }

        return aliases.get(
            value,
            "name",
        )

    # =========================================================
    # NORMALIZE REQUESTED FIELDS
    # =========================================================

    def _normalize_requested_fields(
        self,
        requested_fields,
    ) -> list[str]:

        if not requested_fields:

            return ["name"]

        if isinstance(
            requested_fields,
            str,
        ):

            requested_fields = [
                requested_fields
            ]

        normalized = []

        aliases = {

            "name": "name",
            "voter_name": "name",
            "voter": "name",

            "age": "age",

            "gender": "gender",
            "sex": "gender",

            "house": "house_number",
            "house_no": "house_number",
            "house_number": "house_number",

            "relation": "relation_name",
            "relation_name": "relation_name",

            "relation_type": "relation_type",

            "father": "relation_name",
            "mother": "relation_name",
            "husband": "relation_name",
            "wife": "relation_name",

            "epic": "epic_numbers",
            "epic_number": "epic_numbers",
            "epic_numbers": "epic_numbers",
            "voter_id": "epic_numbers",
            "voterid": "epic_numbers",

            "serial": "serial_number",
            "serial_number": "serial_number",
            "serial_no": "serial_number",

            "position": "position",

            "all": "all",
        }

        for field in requested_fields:

            if field is None:
                continue

            normalized_field = (
                str(field)
                .lower()
                .strip()
            )

            normalized_field = aliases.get(
                normalized_field,
                normalized_field,
            )

            if (
                normalized_field
                and normalized_field not in normalized
            ):

                normalized.append(
                    normalized_field
                )

        if not normalized:

            return ["name"]

        return normalized

    # =========================================================
    # NORMALIZE VALUE
    # =========================================================

    def _normalize_value(
        self,
        lookup_type: str,
        value,
    ):

        if value is None:
            return None

        if lookup_type in {
            "serial",
            "position",
        }:

            if isinstance(
                value,
                int,
            ):

                return value

            text = (
                str(value)
                .strip()
            )

            match = re.search(
                r"\d+",
                text,
            )

            if match:

                try:

                    return int(
                        match.group(0)
                    )

                except ValueError:

                    return None

            return None

        if lookup_type == "epic":

            text = (
                str(value)
                .strip()
                .upper()
            )

            text = re.sub(
                r"^(EPIC|VOTER[\s_-]*ID)\s*[:#-]?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            return text.strip()

        if isinstance(
            value,
            str,
        ):

            return value.strip()

        return value

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    def _normalize_text(
        self,
        value: str,
    ) -> str:

        if value is None:
            return ""

        value = unicodedata.normalize(
            "NFC",
            str(value),
        )

        value = (
            value
            .casefold()
            .strip()
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    # =========================================================
    # RESOLVE NAME
    # =========================================================

    def resolve_name(
        self,
        records: list[dict],
        requested_name: str,
    ) -> list[dict]:

        if not records:
            return []

        if not requested_name:
            return []

        requested_normalized = (
            self._normalize_text(
                requested_name
            )
        )

        # -----------------------------------------------------
        # EXACT MATCH
        # -----------------------------------------------------

        exact_matches = []

        for record in records:

            record_name = record.get(
                "name"
            )

            if not record_name:
                continue

            if (
                self._normalize_text(
                    record_name
                )
                == requested_normalized
            ):

                exact_matches.append(
                    record
                )

        if exact_matches:

            return exact_matches

        # -----------------------------------------------------
        # PARTIAL MATCH
        # -----------------------------------------------------

        partial_matches = []

        for record in records:

            record_name = record.get(
                "name"
            )

            if not record_name:
                continue

            normalized_record_name = (
                self._normalize_text(
                    record_name
                )
            )

            if (
                requested_normalized
                in normalized_record_name
                or normalized_record_name
                in requested_normalized
            ):

                partial_matches.append(
                    record
                )

        if partial_matches:

            return partial_matches

        # -----------------------------------------------------
        # LLM NAME RESOLUTION
        # -----------------------------------------------------

        names = []

        for record in records:

            name = record.get(
                "name"
            )

            if name:

                names.append(
                    str(name)
                )

        if not names:
            return []

        unique_names = list(
            dict.fromkeys(
                names
            )
        )

        names_text = "\n".join(
            f"- {name}"
            for name in unique_names
        )

        prompt = f"""
You are resolving a user's name against names extracted
from a voter list.

USER NAME:
{requested_name}

AVAILABLE NAMES:
{names_text}

Rules:
- Select a name only if it is clearly the same person/name.
- Tamil and English representations of the same name may
  correspond.
- Do not invent a name.
- Do not return a name that is not in the available list.
- Return ONLY the exact name from AVAILABLE NAMES.
- If no reliable match exists, return:

NONE
"""

        try:

            response = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

        except Exception:

            return []

        if not response:
            return []

        resolved_name = (
            response
            .strip()
        )

        resolved_name = re.sub(
            r"^```.*?\n",
            "",
            resolved_name,
            flags=re.DOTALL,
        )

        resolved_name = re.sub(
            r"\n```$",
            "",
            resolved_name,
        ).strip()

        if (
            not resolved_name
            or resolved_name.upper() == "NONE"
        ):

            return []

        # -----------------------------------------------------
        # EXACT VALIDATION AGAINST ACTUAL RECORDS
        # -----------------------------------------------------

        for record in records:

            record_name = record.get(
                "name"
            )

            if not record_name:
                continue

            if (
                self._normalize_text(
                    record_name
                )
                == self._normalize_text(
                    resolved_name
                )
            ):

                return [record]

        return []

    # =========================================================
    # LOOKUP RECORDS
    # =========================================================

    def lookup_records(
        self,
        records: list[dict],
        query: dict,
    ) -> list[dict]:

        if not records:
            return []

        lookup_type = query.get(
            "lookup_type"
        )

        value = query.get(
            "value"
        )

        # -----------------------------------------------------
        # COUNT
        # -----------------------------------------------------

        if lookup_type == "count":

            return records

        # -----------------------------------------------------
        # ALL
        # -----------------------------------------------------

        if lookup_type == "all":

            return records

        # -----------------------------------------------------
        # DOCUMENT
        # -----------------------------------------------------

        if lookup_type == "document":

            return []

        # -----------------------------------------------------
        # POSITION
        # -----------------------------------------------------

        if lookup_type == "position":

            try:

                position = int(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                return []

            if position < 1:
                return []

            if position > len(records):
                return []

            return [
                records[position - 1]
            ]

        # -----------------------------------------------------
        # SERIAL
        # -----------------------------------------------------

        if lookup_type == "serial":

            try:

                serial = int(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                return []

            matches = []

            for record in records:

                record_serial = record.get(
                    "serial_number"
                )

                try:

                    if (
                        int(record_serial)
                        == serial
                    ):

                        matches.append(
                            record
                        )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

            return matches

        # -----------------------------------------------------
        # EPIC
        # -----------------------------------------------------

        if lookup_type == "epic":

            if value is None:
                return []

            requested_epic = (
                str(value)
                .strip()
                .upper()
            )

            matches = []

            for record in records:

                epic_numbers = record.get(
                    "epic_numbers"
                )

                if not epic_numbers:
                    continue

                if isinstance(
                    epic_numbers,
                    (list, tuple),
                ):

                    epic_values = [
                        str(epic)
                        .strip()
                        .upper()
                        for epic in epic_numbers
                        if epic is not None
                    ]

                else:

                    epic_values = [
                        str(
                            epic_numbers
                        )
                        .strip()
                        .upper()
                    ]

                if requested_epic in epic_values:

                    matches.append(
                        record
                    )

            return matches

        # -----------------------------------------------------
        # NAME
        # -----------------------------------------------------

        if lookup_type == "name":

            return self.resolve_name(
                records=records,
                requested_name=str(
                    value
                )
                if value is not None
                else "",
            )

        return []