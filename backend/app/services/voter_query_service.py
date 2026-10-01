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
"What is the serial number of Nandhakumar?"
"What is the voter ID of Nandhakumar?"

IMPORTANT:

When a question asks for information OF a person, the person's
name is the lookup value.

For example:

Question:
"What is the serial number of Nandhakumar?"

Return:

{{
    "lookup_type": "name",
    "value": "Nandhakumar",
    "requested_fields": ["serial_number"]
}}

Question:
"What is the age of Nandhakumar?"

Return:

{{
    "lookup_type": "name",
    "value": "Nandhakumar",
    "requested_fields": ["age"]
}}

Question:
"What is the Voter ID of Nandhakumar?"

Return:

{{
    "lookup_type": "name",
    "value": "Nandhakumar",
    "requested_fields": ["epic_numbers"]
}}

Never set value to null when a specific person's name is
clearly present in the question.

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

If the user asks for the name, age, gender, house number,
relation, or other specific field OF an EPIC, return that
specific requested field.

If the user asks broadly for information/details/about an EPIC,
return:

{{
    "lookup_type": "epic",
    "value": "RMK0799270",
    "requested_fields": ["all"]
}}

Examples:

Question:
"Tell me about voter ID RMK0035378."

Return:

{{
    "lookup_type": "epic",
    "value": "RMK0035378",
    "requested_fields": ["all"]
}}

Question:
"Give me the details of EPIC RMK0035378."

Return:

{{
    "lookup_type": "epic",
    "value": "RMK0035378",
    "requested_fields": ["all"]
}}

Question:
"What is the name of voter ID RMK0035378?"

Return:

{{
    "lookup_type": "epic",
    "value": "RMK0035378",
    "requested_fields": ["name"]
}}

Question:
"What is the age of voter ID RMK0035378?"

Return:

{{
    "lookup_type": "epic",
    "value": "RMK0035378",
    "requested_fields": ["age"]
}}

------------------------------------------------------------
LOOKUP TYPE: serial
------------------------------------------------------------

Use "serial" when the user explicitly refers to a voter
serial number.

A serial number identifies a specific voter record.

Examples:

"What is the name of serial number 11?"
"What is the Voter ID of serial number 11?"
"Tell me about serial 45"
"Tell me about serial number 45"
"Tell me the details of serial number 45"
"Give me details about serial number 45"
"What can you tell me about serial number 45?"
"Who is serial number 45?"
"Age of serial number 56"

IMPORTANT:

If the question contains "serial number N", "serial no N",
or "serial N", the lookup_type MUST be "serial".

The number N is the value of the serial lookup.

For example:

Question:
"Tell me about serial number 45"

Return exactly:

{{
    "lookup_type": "serial",
    "value": 45,
    "requested_fields": ["all"]
}}

Question:
"What can you tell me about serial number 45?"

Return exactly:

{{
    "lookup_type": "serial",
    "value": 45,
    "requested_fields": ["all"]
}}

"Tell me about" does NOT make a question a document query
when a specific voter serial number is present.

The presence of a specific serial number always identifies
a specific voter record.

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
"Summarize this document"

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

IMPORTANT:

Broad information requests such as:

"Tell me about voter ID RMK0035378"
"Tell me about serial number 45"
"Give me details about voter 45"
"Show information about this voter"

must use:

["all"]

The identifier used to find the voter is NOT automatically
a requested field.

For example, "voter ID RMK0035378" identifies the record,
but does NOT mean the user is asking to see only the voter ID.

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
"What is the serial number of Nandhakumar?"

Return:

{{
    "lookup_type": "name",
    "value": "Nandhakumar",
    "requested_fields": ["serial_number"]
}}

Question:
"What is the age of Nandhakumar?"

Return:

{{
    "lookup_type": "name",
    "value": "Nandhakumar",
    "requested_fields": ["age"]
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
9. If the question identifies a specific voter, use
   the appropriate voter lookup type.
10. "Tell me about" a specific voter means requested_fields
    ["all"].
11. "Give me details" about a specific voter means
    requested_fields ["all"].
12. "How many voters" means lookup_type "count".
13. A lookup value such as "serial number 21" or
    "position 21" identifies WHICH record to find.
14. Do NOT include "serial_number" or "position" in
    requested_fields unless the user explicitly asks
    for that field.
15. If the question asks for the serial number OF a named
    voter, use lookup_type "name", put the person's name
    in value, and use ["serial_number"].
16. If the question asks for the position OF a named voter,
    use lookup_type "name", put the person's name in value,
    and use ["position"].
17. If the question asks for the Voter ID OF a named voter,
    use lookup_type "name", put the person's name in value,
    and use ["epic_numbers"].
18. If the question asks for the age OF a named voter,
    use lookup_type "name", put the person's name in value,
    and use ["age"].
19. Never return value:null for a named-voter question when
    the person's name is explicitly present.
20. If the question contains "serial number N", "serial no N",
    or "serial N", it MUST use lookup_type "serial".
21. "Tell me about serial number N" means the specific voter
    record identified by serial number N, not the document.
22. If an EPIC/Voter ID is used only to identify a voter in a
    broad "tell me about/details/information" question, return
    requested_fields ["all"], not ["epic_numbers"].

USER QUESTION:
{question}
"""

        try:
            response = self.ollama_service.generate_response(
                prompt=prompt
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

        text = response.strip()

        # -----------------------------------------------------
        # REMOVE MARKDOWN CODE FENCES
        # -----------------------------------------------------

        text = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = text.replace(
            "```",
            "",
        ).strip()

        # -----------------------------------------------------
        # TRY TO EXTRACT A COMPLETE JSON OBJECT
        # -----------------------------------------------------

        json_match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if json_match:

            candidate = json_match.group(0).strip()

        else:

            # -------------------------------------------------
            # OLLAMA MAY RETURN TRUNCATED JSON.
            # -------------------------------------------------

            candidate = text.strip()

            # -------------------------------------------------
            # REPAIR UNFINISHED ARRAY
            # -------------------------------------------------

            if candidate.count("[") > candidate.count("]"):

                candidate += "]"

            # -------------------------------------------------
            # REPAIR UNFINISHED OBJECT
            # -------------------------------------------------

            if candidate.count("{") > candidate.count("}"):

                candidate += "}"

        # -----------------------------------------------------
        # PARSE JSON
        # -----------------------------------------------------

        try:

            data = json.loads(
                candidate
            )

        except json.JSONDecodeError:

            return {}

        if not isinstance(
            data,
            dict,
        ):

            return {}

        # -----------------------------------------------------
        # NORMALIZE LOOKUP TYPE
        # -----------------------------------------------------

        lookup_type = data.get(
            "lookup_type"
        )

        if isinstance(
            lookup_type,
            str,
        ):

            lookup_type = (
                lookup_type
                .strip()
                .lower()
            )

        else:

            lookup_type = ""

        lookup_type = self._normalize_lookup_type(
            lookup_type
        )

        # -----------------------------------------------------
        # NORMALIZE VALUE
        # -----------------------------------------------------

        value = data.get(
            "value"
        )

        value = self._normalize_value(
            lookup_type,
            value,
        )

        # -----------------------------------------------------
        # NORMALIZE REQUESTED FIELDS
        # -----------------------------------------------------

        requested_fields = data.get(
            "requested_fields",
            [],
        )

        if not isinstance(
            requested_fields,
            list,
        ):

            requested_fields = []

        requested_fields = self._normalize_requested_fields(
            requested_fields
        )

        # -----------------------------------------------------
        # VALIDATE REQUESTED FIELDS
        # -----------------------------------------------------

        requested_fields = (
            self._validate_requested_fields_against_question(
                requested_fields,
                question,
            )
        )

        # -----------------------------------------------------
        # DOCUMENT QUERY
        # -----------------------------------------------------

        if lookup_type == "document":

            result = {
                "lookup_type": "document",
                "value": None,
                "requested_fields": requested_fields,
            }

            return self._correct_explicit_serial_lookup(
                query=result,
                question=question,
            )

        # -----------------------------------------------------
        # COUNT QUERY
        # -----------------------------------------------------

        if lookup_type == "count":

            return {
                "lookup_type": "count",
                "value": None,
                "requested_fields": requested_fields,
            }

        # -----------------------------------------------------
        # NAME FALLBACK
        # -----------------------------------------------------

        if lookup_type == "name" and not value:

            extracted_name = (
                self._extract_name_from_question(
                    question
                )
            )

            if extracted_name:

                value = extracted_name

        # -----------------------------------------------------
        # BROAD SPECIFIC-VOTER QUERY
        #
        # If the user asks "tell me about", "give me details",
        # "show information", etc. about a specific voter,
        # the identifier is only used for lookup. The requested
        # fields must be "all".
        # -----------------------------------------------------

        if lookup_type in {
            "name",
            "epic",
            "serial",
            "position",
        }:

            if self._is_broad_voter_information_question(
                question
            ):

                requested_fields = ["all"]

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        result = {
            "lookup_type": lookup_type,
            "value": value,
            "requested_fields": requested_fields,
        }

        result = self._correct_explicit_serial_lookup(
            query=result,
            question=question,
        )

        return result

    # =========================================================
    # BROAD VOTER INFORMATION QUESTION
    # =========================================================

    def _is_broad_voter_information_question(
        self,
        question: str,
    ) -> bool:

        if not question:
            return False

        text = self._normalize_text(
            question
        )

        broad_patterns = [
            r"\btell me about\b",
            r"\bgive me details\b",
            r"\bgive me the details\b",
            r"\bshow me details\b",
            r"\bshow me the details\b",
            r"\bshow information\b",
            r"\bshow me information\b",
            r"\bgive me information\b",
            r"\bprovide information\b",
            r"\bwhat can you tell me about\b",
            r"\bwhat do you know about\b",
            r"\bwhat information do you have about\b",
            r"\bwhat information is available about\b",
            r"\bwhat information can you give me about\b",
            r"\bdetails about\b",
            r"\binformation about\b",
        ]

        return any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in broad_patterns
        )

    # =========================================================
    # CORRECT EXPLICIT SERIAL LOOKUP
    # =========================================================

    def _correct_explicit_serial_lookup(
        self,
        query: dict,
        question: str,
    ) -> dict:

        if not query or not question:
            return query

        question_text = str(question).strip()

        # -----------------------------------------------------
        # Detect an explicitly mentioned serial number.
        #
        # Examples:
        #   serial number 45
        #   serial no 45
        #   serial 45
        # -----------------------------------------------------

        serial_match = re.search(
            r"\bserial\s*(?:number|no\.?)?\s*[:#-]?\s*(\d{1,4})\b",
            question_text,
            flags=re.IGNORECASE,
        )

        if not serial_match:
            return query

        serial_value = int(
            serial_match.group(1)
        )

        lookup_type = query.get(
            "lookup_type"
        )

        # -----------------------------------------------------
        # If the user explicitly identified a serial number,
        # a document classification is incorrect.
        #
        # Keep this correction narrow so the system remains
        # LLM-based for general questions.
        # -----------------------------------------------------

        if lookup_type == "document":

            query = dict(query)

            query["lookup_type"] = "serial"
            query["value"] = serial_value

            requested_fields = query.get(
                "requested_fields"
            )

            if not isinstance(
                requested_fields,
                list,
            ) or not requested_fields:

                requested_fields = ["all"]

            query["requested_fields"] = (
                self._normalize_requested_fields(
                    requested_fields
                )
            )

        # -----------------------------------------------------
        # If the LLM already identified serial correctly but
        # failed to extract the value, recover the explicit
        # serial number from the question.
        # -----------------------------------------------------

        elif lookup_type == "serial":

            query = dict(query)

            if query.get("value") is None:
                query["value"] = serial_value

        return query

    # =========================================================
    # EXTRACT NAME FROM QUESTION
    # =========================================================

    def _extract_name_from_question(
        self,
        question: str
    ):
        """
        Recover a person's name when the query-understanding
        LLM returns lookup_type='name' but value=None.

        This is only a fallback for extracting the explicit
        name supplied by the user. Actual name matching is
        still handled by resolve_name().
        """

        if not question:
            return None

        text = str(question).strip()

        text = re.sub(
            r"[?؟!.\\s]+$",
            "",
            text,
        ).strip()

        patterns = [

            r"\b(?:what\s+is|what's)\s+"
            r"(?:the\s+)?serial\s+number\s+of\s+(.+)$",

            r"\b(?:what\s+is|what's)\s+"
            r"(?:the\s+)?position\s+of\s+(.+)$",

            r"\b(?:what\s+is|what's)\s+"
            r"(?:the\s+)?age\s+of\s+(.+)$",

            r"\b(?:what\s+is|what's)\s+"
            r"(?:the\s+)?(?:voter\s+id|voter\s+number|epic)"
            r"\s+of\s+(.+)$",

            r"\b(?:what\s+is|what's)\s+"
            r"(?:the\s+)?name\s+of\s+(.+)$",

            r"^\s*who\s+is\s+(.+)$",

            r"^\s*tell\s+me\s+about\s+(.+)$",

            r"^\s*tell\s+me\s+"
            r"(?:the\s+)?age\s+of\s+(.+)$",

            r"^\s*(?:give|show)\s+me\s+"
            r"(?:the\s+)?serial\s+number\s+of\s+(.+)$",

            r"^\s*(?:give|show)\s+me\s+"
            r"(?:the\s+)?(?:voter\s+id|voter\s+number|epic)"
            r"\s+of\s+(.+)$",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = match.group(1).strip()

            candidate = re.sub(
                r"[?؟!.,;:]+$",
                "",
                candidate,
            ).strip()

            if candidate:
                return candidate

        return None

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

        question_normalized = self._normalize_text(
            question
        )

        # -----------------------------------------------------
        # BROAD VOTER INFORMATION
        #
        # Do this BEFORE checking "voter id"/EPIC terms.
        #
        # Example:
        # "Tell me about voter ID RMK0035378"
        #
        # The voter ID identifies the record. It is not the
        # requested output field.
        # -----------------------------------------------------

        if self._is_broad_voter_information_question(
            question
        ):

            return ["all"]

        # -----------------------------------------------------
        # "all" means all useful fields.
        # -----------------------------------------------------

        if "all" in fields:
            fields = []

                # -----------------------------------------------------
        # IDENTITY QUESTION
        #
        # Example:
        # "Who is the voter with voter ID RMK0953109?"
        #
        # The Voter ID is used only to locate the voter.
        # "Who is" asks for the voter's name.
        # -----------------------------------------------------

        identity_question = any(
            re.search(
                pattern,
                question_normalized,
            )
            for pattern in [
                r"^\s*who\s+is\s+(?:the\s+)?voter\b",
                r"\bwho\s+is\s+the\s+voter\b",
                r"\bwho\s+has\s+(?:the\s+)?(?:voter\s+id|voterid|epic)\b",
                r"\bwhich\s+voter\s+has\s+(?:the\s+)?(?:voter\s+id|voterid|epic)\b",
            ]
        )

        if identity_question:
            return ["name"]
        # -----------------------------------------------------
        # VOTER ID / EPIC
        # -----------------------------------------------------

        epic_terms = [
            "voter id",
            "voterid",
            "voter id number",
            "voter number",
            "epic",
            "epic number",
            "epic no",
        ]

        # -----------------------------------------------------
        # VOTER ID / EPIC
        # -----------------------------------------------------

        epic_terms = [
            "voter id",
            "voterid",
            "voter id number",
            "voter number",
            "epic",
            "epic number",
            "epic no",
        ]

        asks_epic = any(
            term in question_normalized
            for term in epic_terms
        )

        if asks_epic:

            fields = [
                field
                for field in fields
                if field not in {
                    "serial_number",
                    "position",
                }
            ]

            if "epic_numbers" not in fields:
                fields.append(
                    "epic_numbers"
                )

        # -----------------------------------------------------
        # EXPLICIT SERIAL NUMBER REQUEST
        # -----------------------------------------------------

        explicit_serial_request = any(
            re.search(
                pattern,
                question_normalized,
            )
            for pattern in [
                r"\bwhat\s+(?:is|are)\s+(?:the\s+)?serial\s+number\b",
                r"\bwhat's\s+(?:the\s+)?serial\s+number\b",
                r"\bserial\s+number\s+of\b",
                r"\bserial\s+no\s+of\b",
                r"\bserial\s+number\s*[:?]",
            ]
        )

        if explicit_serial_request:

            fields = [
                field
                for field in fields
                if field != "position"
            ]

            if "serial_number" not in fields:
                fields.append(
                    "serial_number"
                )

        # -----------------------------------------------------
        # POSITION
        # -----------------------------------------------------

        explicit_position_request = any(
            re.search(
                pattern,
                question_normalized,
            )
            for pattern in [
                r"\bwhat\s+(?:is|are)\s+(?:the\s+)?position\b",
                r"\bwhat's\s+(?:the\s+)?position\b",
                r"\bposition\s+of\b",
            ]
        )

        if explicit_position_request:

            fields = [
                field
                for field in fields
                if field != "serial_number"
            ]

            if "position" not in fields:
                fields.append(
                    "position"
                )

        # -----------------------------------------------------
        # REMOVE "all"
        # -----------------------------------------------------

        fields = [
            field
            for field in fields
            if field != "all"
        ]

        # -----------------------------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------------------------

        result = []

        for field in fields:

            if field not in result:
                result.append(
                    field
                )

        # -----------------------------------------------------
        # Preserve whatever valid field Ollama identified.
        # -----------------------------------------------------

        if result:
            return result

        # -----------------------------------------------------
        # Last fallback
        # -----------------------------------------------------

        return ["name"]

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

        if "all" in normalized:
            return ["all"]

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
    # NORMALIZE OCR RECORD TEXT
    # =========================================================

    def _normalize_record_text(
        self,
        text,
    ) -> str:

        if text is None:
            return ""

        text = unicodedata.normalize(
            "NFC",
            str(text),
        )

        text = text.replace(
            "\\\r\\\n",
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

        return text

    # =========================================================
    # CLEAN EXTRACTED VALUE
    # =========================================================

    def _clean_field_value(
        self,
        value,
    ):

        if value is None:
            return None

        value = str(value).strip()

        value = value.strip(
            " \t\r\n:-–—"
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        if not value:
            return None

        return value

    # =========================================================
    # EXTRACT VALUE BETWEEN LABELS
    # =========================================================

    def _extract_between_labels(
        self,
        text: str,
        start_pattern: str,
        stop_patterns: list[str],
    ):
        """
        Extract the text between one field label and the
        earliest following field label.
        """

        start_match = re.search(
            start_pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not start_match:
            return None

        value_start = start_match.end()

        value_end = len(text)

        remaining_text = text[value_start:]

        for stop_pattern in stop_patterns:

            stop_match = re.search(
                stop_pattern,
                remaining_text,
                flags=re.IGNORECASE,
            )

            if not stop_match:
                continue

            candidate_end = (
                value_start
                + stop_match.start()
            )

            if candidate_end < value_end:
                value_end = candidate_end

        value = text[
            value_start:value_end
        ]

        return self._clean_field_value(
            value
        )

    # =========================================================
    # EXTRACT VOTER FIELDS FROM OCR TEXT
    # =========================================================

    def _extract_voter_fields(
        self,
        record: dict,
    ) -> dict:

        result = dict(record)

        text = self._normalize_record_text(
            record.get("text")
        )

        if not text:
            return result

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # -----------------------------------------------------
        # SERIAL NUMBER
        # -----------------------------------------------------

        serial_number = record.get(
            "serial_number"
        )

        if serial_number is not None:
            result["serial_number"] = serial_number

        # -----------------------------------------------------
        # EPIC / VOTER ID
        # -----------------------------------------------------

        epic_numbers = record.get(
            "epic_numbers"
        )

        if epic_numbers:
            result["epic_numbers"] = epic_numbers

        # -----------------------------------------------------
        # COMMON FIELD LABELS
        # -----------------------------------------------------

        common_stop_labels = [
            r"தந்தையின்\s*பெயர்\s*[:：]",
            r"தந்தை\s*பெயர்\s*[:：]",
            r"கணவரின்\s*பெயர்\s*[:：]",
            r"கணவர்\s*பெயர்\s*[:：]",
            r"தாயின்\s*பெயர்\s*[:：]",
            r"தாய்\s*பெயர்\s*[:：]",
            r"இதரர்\s*பெயர்\s*[:：]",
            r"வீட்டு\s*எண்\s*[:：]",
            r"வயது\s*[:：]",
            r"பாலினம்\s*[:：]",
        ]

        # -----------------------------------------------------
        # NAME
        # -----------------------------------------------------

        name = self._extract_between_labels(
            text=text,
            start_pattern=r"பெயர்\s*[:：]",
            stop_patterns=common_stop_labels,
        )

        if name:

            name = re.sub(
                r"\s*[-–—]\s*$",
                "",
                name,
            ).strip()

            name = self._clean_field_value(
                name
            )

            if name:
                result["name"] = name

        # -----------------------------------------------------
        # ENGLISH NAME FALLBACK
        # -----------------------------------------------------

        if not result.get("name"):

            english_name = self._extract_between_labels(
                text=text,
                start_pattern=r"\bname\b\s*[:：]",
                stop_patterns=[
                    r"\bfather\b\s*[:：]",
                    r"\bmother\b\s*[:：]",
                    r"\bhusband\b\s*[:：]",
                    r"\bhouse\s*(?:number|no)?\s*[:：]",
                    r"\bage\b\s*[:：]",
                    r"\bgender\b\s*[:：]",
                ],
            )

            if english_name:

                english_name = re.sub(
                    r"\s*[-–—]\s*$",
                    "",
                    english_name,
                ).strip()

                result["name"] = (
                    self._clean_field_value(
                        english_name
                    )
                )

        # -----------------------------------------------------
        # RELATION
        # -----------------------------------------------------

        relation_definitions = [

            (
                "தந்தை",
                r"தந்தையின்\s*பெயர்\s*[:：]",
            ),

            (
                "தந்தை",
                r"தந்தை\s*பெயர்\s*[:：]",
            ),

            (
                "கணவர்",
                r"கணவரின்\s*பெயர்\s*[:：]",
            ),

            (
                "கணவர்",
                r"கணவர்\s*பெயர்\s*[:：]",
            ),

            (
                "தாய்",
                r"தாயின்\s*பெயர்\s*[:：]",
            ),

            (
                "தாய்",
                r"தாய்\s*பெயர்\s*[:：]",
            ),

            (
                "இதரர்",
                r"இதரர்\s*பெயர்\s*[:：]",
            ),
        ]

        relation_stop_labels = [
            r"Photo\s+is\b",
            r"வீட்டு\s*எண்\s*[:：]",
            r"வயது\s*[:：]",
            r"பாலினம்\s*[:：]",
        ]

        for relation_type, start_pattern in relation_definitions:

            relation_name = self._extract_between_labels(
                text=text,
                start_pattern=start_pattern,
                stop_patterns=relation_stop_labels,
            )

            if not relation_name:
                continue

            relation_name = re.sub(
                r"\s*[-–—]\s*$",
                "",
                relation_name,
            ).strip()

            relation_name = re.sub(
                r"\s*Photo\s+is.*$",
                "",
                relation_name,
                flags=re.IGNORECASE,
            ).strip()

            relation_name = self._clean_field_value(
                relation_name
            )

            if relation_name:

                result["relation_type"] = (
                    relation_type
                )

                result["relation_name"] = (
                    relation_name
                )

                break

        # -----------------------------------------------------
        # HOUSE NUMBER
        # -----------------------------------------------------

        house_number = self._extract_between_labels(
            text=text,
            start_pattern=r"வீட்டு\s*(?:எண்|என்)\s*[:：]",
            stop_patterns=[
                r"வயது\s*[:：]",
                r"பாலினம்\s*[:：]",
            ],
        )

        if house_number:

            house_number = re.sub(
                r"\s*[-–—]\s*$",
                "",
                house_number,
            ).strip()

            house_number = self._clean_field_value(
                house_number
            )

        if house_number:
            result["house_number"] = (
                house_number
            )

        # -----------------------------------------------------
        # ENGLISH HOUSE NUMBER FALLBACK
        # -----------------------------------------------------

        if not result.get("house_number"):

            house_number = self._extract_between_labels(
                text=text,
                start_pattern=(
                    r"\bhouse\s*(?:number|no)?\s*[:：]"
                ),
                stop_patterns=[
                    r"\bage\b\s*[:：]",
                    r"\bgender\b\s*[:：]",
                ],
            )

            if house_number:

                house_number = re.sub(
                    r"\s*[-–—]\s*$",
                    "",
                    house_number,
                ).strip()

                result["house_number"] = (
                    self._clean_field_value(
                        house_number
                    )
                )

        # -----------------------------------------------------
        # AGE
        # -----------------------------------------------------

        age_patterns = [
            r"வயது\s*[:：]\s*(\d{1,3})",
            r"\bage\b\s*[:：]\s*(\d{1,3})",
        ]

        for pattern in age_patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            try:
                result["age"] = int(
                    match.group(1)
                )
            except ValueError:
                pass

            break

        # -----------------------------------------------------
        # GENDER
        # -----------------------------------------------------

        gender_patterns = [
            r"பாலினம்\s*[:：]\s*(.*)$",
            r"\bgender\b\s*[:：]\s*(.*)$",
            r"\bsex\b\s*[:：]\s*(.*)$",
        ]

        for pattern in gender_patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if not match:
                continue

            gender = self._clean_field_value(
                match.group(1)
            )

            if not gender:
                continue

            gender = re.sub(
                r"\s*[-–—]\s*$",
                "",
                gender,
            ).strip()

            gender = re.split(
                r"\s+(?:Photo\s+is)\b",
                gender,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]

            gender = self._clean_field_value(
                gender
            )

            if gender:
                result["gender"] = gender
                break

        # -----------------------------------------------------
        # FINAL CLEANUP
        # -----------------------------------------------------

        for field in [
            "name",
            "relation_name",
            "house_number",
            "gender",
        ]:

            value = result.get(field)

            if value is None:
                continue

            value = self._clean_field_value(
                value
            )

            if value:
                result[field] = value

        # -----------------------------------------------------
        # DEBUG
        # -----------------------------------------------------

        print(
            "\n========== VOTER FIELD EXTRACTION =========="
        )

        print(
            f"SERIAL: "
            f"{result.get('serial_number')}"
        )

        print(
            f"NAME: "
            f"{result.get('name', 'Not available')}"
        )

        print(
            f"RELATION TYPE: "
            f"{result.get('relation_type', 'Not available')}"
        )

        print(
            f"RELATION NAME: "
            f"{result.get('relation_name', 'Not available')}"
        )

        print(
            f"HOUSE NUMBER: "
            f"{result.get('house_number', 'Not available')}"
        )

        print(
            f"AGE: "
            f"{result.get('age', 'Not available')}"
        )

        print(
            f"GENDER: "
            f"{result.get('gender', 'Not available')}"
        )

        print(
            f"EPIC: "
            f"{result.get('epic_numbers', 'Not available')}"
        )

        print(
            "============================================\n"
        )

        return result

    # =========================================================
    # ENRICH ALL RECORDS
    # =========================================================

    def _enrich_records(
        self,
        records: list[dict],
    ) -> list[dict]:

        enriched = []

        for record in records:

            if not isinstance(record, dict):
                continue

            enriched_record = (
                self._extract_voter_fields(
                    record
                )
            )

            enriched.append(
                enriched_record
            )

        return enriched

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

        records = self._enrich_records(
            records
        )

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

- Select a name only if it clearly refers to the same person/name.
- Tamil and English representations of the same name may correspond.
- English-to-Tamil and Tamil-to-English transliteration is allowed.
- Compare pronunciation and spelling across English and Tamil.
- For example:
  Nandhakumar = நந்தகுமார்
- If an English name is a transliteration of a Tamil candidate,
  select that Tamil candidate.
- Do not invent a name.
- Do not return a name that is not in the AVAILABLE NAMES list.
- Return the exact name from AVAILABLE NAMES.
- The selected name must appear exactly as written in
  AVAILABLE NAMES.
- You may include a short explanation, but the exact selected
  candidate name must be present in your response.
- If no reliable match exists, return:

NONE
"""

        try:

            response = self.ollama_service.generate_response(
                prompt=prompt
            )

        except Exception:

            return []

        if not response:
            return []

        resolved_name = str(
            response
        ).strip()

        # -----------------------------------------------------
        # REMOVE MARKDOWN FENCE
        # -----------------------------------------------------

        resolved_name = re.sub(
            r"^```(?:text)?\s*",
            "",
            resolved_name,
            flags=re.IGNORECASE,
        )

        resolved_name = re.sub(
            r"\s*```$",
            "",
            resolved_name,
        ).strip()

        if (
            not resolved_name
            or resolved_name.upper() == "NONE"
        ):
            return []

        # -----------------------------------------------------
        # SEARCH RESPONSE FOR EXACT CANDIDATE
        # -----------------------------------------------------

        response_normalized = self._normalize_text(
            resolved_name
        )

        for candidate_name in unique_names:

            candidate_normalized = (
                self._normalize_text(
                    candidate_name
                )
            )

            if not candidate_normalized:
                continue

            if candidate_normalized in response_normalized:

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
                        == candidate_normalized
                    ):
                        return [record]

        # -----------------------------------------------------
        # EXACT VALIDATION AGAINST ACTUAL RECORDS
        # -----------------------------------------------------

        cleaned_resolved_name = (
            resolved_name.strip(
                " \t\r\n.,:;\"'`*-"
            )
        )

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
                    cleaned_resolved_name
                )
            ):

                return [record]

        return []

    # =========================================================
    # FILTER FIELDS
    # =========================================================

    def _filter_requested_fields(
        self,
        records: list[dict],
        requested_fields: list[str],
    ) -> list[dict]:

        if not records:
            return []

        requested_fields = (
            self._normalize_requested_fields(
                requested_fields
            )
        )

        # -----------------------------------------------------
        # "all"
        # -----------------------------------------------------

        if "all" in requested_fields:

            filtered_records = []

            for record in records:

                filtered_records.append(
                    {
                        "serial_number": record.get(
                            "serial_number"
                        ),
                        "name": record.get(
                            "name"
                        ),
                        "relation_type": record.get(
                            "relation_type"
                        ),
                        "relation_name": record.get(
                            "relation_name"
                        ),
                        "house_number": record.get(
                            "house_number"
                        ),
                        "age": record.get(
                            "age"
                        ),
                        "gender": record.get(
                            "gender"
                        ),
                        "epic_numbers": record.get(
                            "epic_numbers"
                        ),
                        "_record_position": record.get(
                            "_record_position"
                        ),
                    }
                )

            return filtered_records

        # -----------------------------------------------------
        # Only requested fields
        # -----------------------------------------------------

        filtered_records = []

        for record in records:

            filtered = {}

            for field in requested_fields:

                if field == "position":

                    filtered["position"] = record.get(
                        "_record_position"
                    )

                elif field == "epic_numbers":

                    epic_numbers = record.get(
                        "epic_numbers"
                    )

                    filtered["epic_numbers"] = (
                        epic_numbers
                        if epic_numbers
                        else "Not available"
                    )

                else:

                    value = record.get(
                        field
                    )

                    filtered[field] = (
                        value
                        if value is not None
                        else "Not available"
                    )

            filtered_records.append(
                filtered
            )

        return filtered_records

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

        records = self._enrich_records(
            records
        )

        lookup_type = query.get(
            "lookup_type"
        )

        value = query.get(
            "value"
        )

        requested_fields = (
            query.get(
                "requested_fields"
            )
            or ["all"]
        )

        requested_fields = (
            self._normalize_requested_fields(
                requested_fields
            )
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

            return self._filter_requested_fields(
                records=records,
                requested_fields=requested_fields,
            )

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
                position = int(value)

            except (
                TypeError,
                ValueError,
            ):
                return []

            if position < 1:
                return []

            if position > len(records):
                return []

            matched = [
                records[position - 1]
            ]

            print(
                "\n========== LOOKUP DEBUG =========="
            )

            print(
                f"LOOKUP TYPE: {lookup_type}"
            )

            print(
                f"LOOKUP VALUE: {value}"
            )

            print(
                f"REQUESTED FIELDS: "
                f"{requested_fields}"
            )

            print(
                f"MATCHED COUNT: {len(matched)}"
            )

            print(
                f"MATCHED RECORD: {matched[0]}"
            )

            print(
                "==================================\n"
            )

            return self._filter_requested_fields(
                records=matched,
                requested_fields=requested_fields,
            )

        # -----------------------------------------------------
        # SERIAL
        # -----------------------------------------------------

        if lookup_type == "serial":

            try:
                serial = int(value)

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

            print(
                "\n========== LOOKUP DEBUG =========="
            )

            print(
                f"LOOKUP TYPE: {lookup_type}"
            )

            print(
                f"LOOKUP VALUE: {value}"
            )

            print(
                f"REQUESTED FIELDS: "
                f"{requested_fields}"
            )

            print(
                f"MATCHED COUNT: {len(matches)}"
            )

            if matches:

                print(
                    f"MATCHED RECORD: "
                    f"{matches[0]}"
                )

            print(
                "==================================\n"
            )

            return self._filter_requested_fields(
                records=matches,
                requested_fields=requested_fields,
            )

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
                        str(epic_numbers)
                        .strip()
                        .upper()
                    ]

                if requested_epic in epic_values:

                    matches.append(
                        record
                    )

            print(
                "\n========== LOOKUP DEBUG =========="
            )

            print(
                f"LOOKUP TYPE: {lookup_type}"
            )

            print(
                f"LOOKUP VALUE: {value}"
            )

            print(
                f"REQUESTED FIELDS: "
                f"{requested_fields}"
            )

            print(
                f"MATCHED COUNT: {len(matches)}"
            )

            if matches:

                print(
                    f"MATCHED RECORD: "
                    f"{matches[0]}"
                )

            print(
                "==================================\n"
            )

            return self._filter_requested_fields(
                records=matches,
                requested_fields=requested_fields,
            )

        # -----------------------------------------------------
        # NAME
        # -----------------------------------------------------

        if lookup_type == "name":

            matches = self.resolve_name(
                records=records,
                requested_name=(
                    str(value)
                    if value is not None
                    else ""
                ),
            )

            print(
                "\n========== LOOKUP DEBUG =========="
            )

            print(
                f"LOOKUP TYPE: {lookup_type}"
            )

            print(
                f"LOOKUP VALUE: {value}"
            )

            print(
                f"REQUESTED FIELDS: "
                f"{requested_fields}"
            )

            print(
                f"MATCHED COUNT: {len(matches)}"
            )

            if matches:

                print(
                    f"MATCHED RECORD: "
                    f"{matches[0]}"
                )

            print(
                "==================================\n"
            )

            return self._filter_requested_fields(
                records=matches,
                requested_fields=requested_fields,
            )

        return []