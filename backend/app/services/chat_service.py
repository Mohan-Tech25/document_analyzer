
import os

from psycopg import Connection

from app.core.exceptions import ChatException
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService
from app.services.ollama_service import OllamaService
from app.services.retrieval_service import RetrievalService
from app.services.voter_query_service import VoterQueryService
from app.services.voter_record_service import VoterRecordService


class ChatService:

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        analysis_repository: AnalysisRepository,
        document_repository: DocumentRepository,
        retrieval_service: RetrievalService,
        ollama_service: OllamaService,
        voter_record_service: VoterRecordService,
        voter_query_service: VoterQueryService,
    ):
        self.chunk_repository = chunk_repository
        self.analysis_repository = analysis_repository
        self.document_repository = document_repository
        self.retrieval_service = retrieval_service
        self.ollama_service = ollama_service
        self.voter_record_service = voter_record_service
        self.voter_query_service = voter_query_service

    # =========================================================
    # IMAGE CHECK
    # =========================================================

    def _is_image(
        self,
        filename: str,
    ) -> bool:

        if not filename:
            return False

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
        }

        extension = os.path.splitext(
            filename
        )[1].lower()

        return extension in image_extensions

    # =========================================================
    # GET VOTER RECORDS
    # =========================================================

    def _get_voter_records(
        self,
        connection: Connection,
        document_id: int,
    ) -> list[dict]:

        rows = self.chunk_repository.get_by_document(
            connection,
            document_id,
        )

        if not rows:
            return []

        records = self.voter_record_service.parse_chunk_rows(
            rows
        )

        # Preserve document/chunk order.
        # Important for position lookups.
        for position, record in enumerate(
            records,
            start=1,
        ):
            record["_record_position"] = position

        return records

    # =========================================================
    # EPIC / VOTER ID FORMATTER
    # =========================================================

    def _format_epic_numbers(
        self,
        record,
    ) -> str:

        if not record:
            return "Not available"

        epic_numbers = None

        if isinstance(
            record,
            dict,
        ):

            epic_numbers = record.get(
                "epic_numbers"
            )

            if epic_numbers is None:
                epic_numbers = record.get(
                    "epic"
                )

            if epic_numbers is None:
                epic_numbers = record.get(
                    "voter_id"
                )

        else:

            epic_numbers = getattr(
                record,
                "epic_numbers",
                None,
            )

            if epic_numbers is None:
                epic_numbers = getattr(
                    record,
                    "epic",
                    None,
                )

            if epic_numbers is None:
                epic_numbers = getattr(
                    record,
                    "voter_id",
                    None,
                )

        if not epic_numbers:
            return "Not available"

        if isinstance(
            epic_numbers,
            (list, tuple),
        ):

            values = [
                str(value).strip()
                for value in epic_numbers
                if (
                    value is not None
                    and str(value).strip()
                )
            ]

            if not values:
                return "Not available"

            return ", ".join(
                values
            )

        value = str(
            epic_numbers
        ).strip()

        if not value:
            return "Not available"

        return value

    # =========================================================
    # GET VOTER FIELD VALUE
    # =========================================================

    def _get_voter_field_value(
        self,
        record,
        field_name: str,
    ) -> str:

        if not record:
            return "Not available"

        field_name = (
            str(field_name)
            .lower()
            .strip()
        )

        aliases = {

            "name": [
                "name",
                "voter_name",
                "english_name",
                "tamil_name",
            ],

            "age": [
                "age",
            ],

            "gender": [
                "gender",
                "sex",
            ],

            "epic": [
                "epic_numbers",
                "epic",
                "voter_id",
            ],

            "epic_numbers": [
                "epic_numbers",
                "epic",
                "voter_id",
            ],

            "voter_id": [
                "epic_numbers",
                "epic",
                "voter_id",
            ],

            "serial": [
                "serial_number",
                "serial",
            ],

            "serial_number": [
                "serial_number",
                "serial",
            ],

            "position": [
                "_record_position",
                "position",
            ],

            "house": [
                "house_number",
                "house",
            ],

            "house_number": [
                "house_number",
                "house",
            ],

            "relation_type": [
                "relation_type",
            ],

            "relation_name": [
                "relation_name",
            ],
        }

        # -----------------------------------------------------
        # EPIC / VOTER ID
        # -----------------------------------------------------

        if field_name in {
            "epic",
            "epic_numbers",
            "voter_id",
        }:

            return self._format_epic_numbers(
                record
            )

        # -----------------------------------------------------
        # OTHER FIELDS
        # -----------------------------------------------------

        possible_keys = aliases.get(
            field_name,
            [field_name],
        )

        for key in possible_keys:

            if isinstance(
                record,
                dict,
            ):

                value = record.get(
                    key
                )

            else:

                value = getattr(
                    record,
                    key,
                    None,
                )

            if (
                value is not None
                and str(value).strip()
            ):

                return str(
                    value
                ).strip()

        return "Not available"

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

        normalized_fields = []

        for field in requested_fields:

            if field is None:
                continue

            field = (
                str(field)
                .lower()
                .strip()
            )

            if (
                field
                and field not in normalized_fields
            ):

                normalized_fields.append(
                    field
                )

        if not normalized_fields:
            return ["name"]

        # -----------------------------------------------------
        # ALL MEANS ALL ACTUAL VOTER FIELDS
        # -----------------------------------------------------

        if "all" in normalized_fields:

            return [
                "serial_number",
                "name",
                "relation_type",
                "relation_name",
                "house_number",
                "age",
                "gender",
                "voter_id",
            ]

        return normalized_fields

    # =========================================================
    # BUILD REQUESTED VOTER CONTEXT
    # =========================================================

    def _build_requested_voter_context(
        self,
        matched_records,
        requested_fields,
    ) -> str:

        if not matched_records:
            return ""

        requested_fields = (
            self._normalize_requested_fields(
                requested_fields
            )
        )

        display_names = {

            "serial_number": "Serial Number",

            "name": "Name",

            "relation_type": "Relation Type",

            "relation_name": "Relation Name",

            "house_number": "House Number",

            "age": "Age",

            "gender": "Gender",

            "voter_id": "Voter ID",

            "epic": "Voter ID",

            "epic_numbers": "Voter ID",

            "position": "Position",
        }

        lines = []

        for index, record in enumerate(
            matched_records,
            start=1,
        ):

            record_lines = []

            for field in requested_fields:

                value = (
                    self._get_voter_field_value(
                        record,
                        field,
                    )
                )

                display_field = (
                    display_names.get(
                        field,
                        field.replace(
                            "_",
                            " ",
                        ).title(),
                    )
                )

                record_lines.append(
                    f"{display_field}: {value}"
                )

            if record_lines:

                lines.append(
                    f"Record {index}:\n"
                    + "\n".join(
                        record_lines
                    )
                )

        return "\n\n".join(
            lines
        )

    # =========================================================
    # GET DOCUMENT LEVEL CONTEXT
    # =========================================================

    def _get_document_level_context(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> str:
        """
        Retrieve document content for document-level questions.

        The document header is always prioritized because voter-list
        metadata such as constituency, part number, section name,
        publication date, and total page count are normally stored
        there.

        Semantic retrieval is still used so this remains a generic
        LLM-based document QA flow rather than a completely
        rule-based system.
        """

        context_chunks = []

        # -----------------------------------------------------
        # GET ALL DOCUMENT CHUNKS
        # -----------------------------------------------------

        try:

            document_chunks = (
                self.retrieval_service
                .get_document_chunks(
                    connection=connection,
                    document_id=document_id,
                )
            )

        except Exception as exc:

            print(
                "DEBUG: document chunk retrieval failed:"
            )

            print(exc)

            document_chunks = []

        # -----------------------------------------------------
        # IDENTIFY DOCUMENT HEADER
        #
        # Chunk 0 is the dedicated header chunk created by the
        # voter-list chunking pipeline.
        # -----------------------------------------------------

        header_chunk = None

        for chunk in document_chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_index = chunk.get(
                "chunk_index"
            )

            if chunk_index == 0:

                header_chunk = chunk
                break

        # -----------------------------------------------------
        # FIRST: DOCUMENT HEADER
        #
        # Always prioritize the header for document-level
        # metadata questions.
        # -----------------------------------------------------

        if header_chunk:

            header_copy = dict(
                header_chunk
            )

            header_copy[
                "_document_header"
            ] = True

            context_chunks.append(
                header_copy
            )

        # -----------------------------------------------------
        # SECOND: SEMANTIC RETRIEVAL
        #
        # Keep semantic retrieval so document QA remains
        # generic and can answer questions beyond metadata.
        # -----------------------------------------------------

        try:

            semantic_chunks = (
                self.retrieval_service
                .retrieve_relevant_chunks(
                    connection=connection,
                    document_id=document_id,
                    question=question,
                    top_k=8,
                )
            )

            if semantic_chunks:

                context_chunks.extend(
                    semantic_chunks
                )

        except Exception as exc:

            print(
                "DEBUG: semantic document retrieval failed:"
            )

            print(exc)

        # -----------------------------------------------------
        # THIRD: EARLY DOCUMENT CHUNKS
        #
        # Include a few early chunks because additional
        # document-level information can appear immediately
        # after the header.
        # -----------------------------------------------------

        if document_chunks:

            context_chunks.extend(
                document_chunks[:5]
            )

        # -----------------------------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------------------------

        unique_chunks = {}

        for chunk in context_chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_id = chunk.get(
                "id"
            )

            if chunk_id is not None:

                key = (
                    "id",
                    chunk_id,
                )

            else:

                key = (
                    "content",
                    chunk.get(
                        "content"
                    ),
                    chunk.get(
                        "chunk_index"
                    ),
                    chunk.get(
                        "page_number"
                    ),
                )

            if key not in unique_chunks:

                unique_chunks[key] = chunk

        # -----------------------------------------------------
        # SORT DOCUMENT ORDER
        #
        # Header is explicitly kept first.
        # -----------------------------------------------------

        result = list(
            unique_chunks.values()
        )

        result.sort(
            key=lambda chunk: (
                0
                if chunk.get(
                    "_document_header",
                    False,
                )
                else 1,

                chunk.get(
                    "page_number"
                )
                if chunk.get(
                    "page_number"
                ) is not None
                else 999999,

                chunk.get(
                    "chunk_index"
                )
                if chunk.get(
                    "chunk_index"
                ) is not None
                else 999999,
            )
        )

        # -----------------------------------------------------
        # BUILD TEXT
        # -----------------------------------------------------

        context_parts = []

        for chunk in result:

            content = chunk.get(
                "content",
                "",
            )

            if not content:

                continue

            chunk_index = chunk.get(
                "chunk_index"
            )

            page_number = chunk.get(
                "page_number"
            )

            is_header = chunk.get(
                "_document_header",
                False,
            )

            # -------------------------------------------------
            # DOCUMENT HEADER
            # -------------------------------------------------

            if is_header:

                context_parts.append(
                    "===== DOCUMENT HEADER =====\n"
                    f"{content}\n"
                    "===== END DOCUMENT HEADER ====="
                )

                continue

            # -------------------------------------------------
            # NORMAL CHUNK
            # -------------------------------------------------

            location = []

            if page_number is not None:

                location.append(
                    f"Source Page {page_number}"
                )

            if chunk_index is not None:

                location.append(
                    f"Source Chunk {chunk_index}"
                )

            if location:

                label = " | ".join(
                    location
                )

                context_parts.append(
                    f"[{label}]\n{content}"
                )

            else:

                context_parts.append(
                    content
                )

        return "\n\n".join(
            context_parts
        )

    # =========================================================
    # ANSWER DOCUMENT QUESTION
    # =========================================================

    def _answer_document_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> str:
        """
        Answer questions about the document itself.

        The answer is grounded only in extracted document text.

        The document header is explicitly prioritized because it
        contains important metadata such as:

        - constituency
        - part number
        - section name
        - publication date
        - total page count
        """

        context = (
            self._get_document_level_context(
                connection=connection,
                document_id=document_id,
                question=question,
            )
        )

        if not context.strip():

            return (
                "The requested information is not "
                "available in the provided content."
            )

        # -----------------------------------------------------
        # DETECT PAGE COUNT QUESTIONS
        #
        # This does NOT extract or hardcode the answer.
        #
        # It only gives the LLM additional instructions about
        # how to interpret page-count metadata.
        # -----------------------------------------------------

        question_lower = (
            question
            .lower()
            .strip()
        )

        page_count_question = any(
            phrase in question_lower
            for phrase in [
                "how many pages",
                "how many page",
                "total pages",
                "total number of pages",
                "number of pages",
                "page count",
                "page-count",
                "pages are there",
            ]
        )

        if page_count_question:

            page_count_instruction = """
IMPORTANT FOR PAGE COUNT QUESTIONS:

The document header may contain several different numbers.

When answering the total number of pages:

- Look specifically for the document's total-page
  information.
- Tamil phrases such as "மொத்தப் பக்கங்கள்" mean
  "total pages".
- A value such as "மொத்தப் பக்கங்கள் 36" means the
  document contains 36 total pages.
- A phrase such as "பக்கம் 4" means the current page
  number printed on that page. It does NOT mean that
  the document has 4 pages.
- "Page 4", "Source Page 4", "Chunk 0", and similar
  retrieval labels are NOT the document's total page count.
- Do not calculate the page count from the number of
  retrieved chunks.
- Do not use an arbitrary number from a voter record as
  the page count.
- If the document explicitly states a total page count,
  use that value exactly.
"""

        else:

            page_count_instruction = ""

        prompt = f"""
You are answering a question about a document.

Use ONLY the document content provided below.

DOCUMENT CONTENT:
{context}

USER QUESTION:
{question}

{page_count_instruction}

GENERAL RULES:

- Answer only from the provided document content.
- Do not use outside knowledge.
- Do not invent information.
- Do not guess.
- Do not assume information that is not explicitly present.
- The section marked "DOCUMENT HEADER" contains source
  metadata extracted from the document and should be
  given priority for document-level metadata questions.
- Retrieval labels such as "Source Page" and "Source Chunk"
  are system-generated labels. They are NOT document facts.
- Do not confuse a source/retrieval label with information
  written inside the document.
- If the requested information exists, give the direct
  answer first.
- If the requested information is not present, say:

"The requested information is not available in the provided content."

- Preserve names, numbers, dates, and identifiers exactly
  as they appear in the document when possible.
- If the document contains Tamil text, preserve the Tamil
  value when appropriate.
- Do not turn unrelated OCR text into an answer.
- Keep the answer clear and concise.
"""

        # -----------------------------------------------------
        # DEBUG CONTEXT
        # -----------------------------------------------------

        print(
            "\n========== DOCUMENT QUESTION DEBUG =========="
        )

        print(
            f"QUESTION: {question}"
        )

        print(
            "DOCUMENT CONTEXT:"
        )

        print(
            context
        )

        print(
            "=============================================\n"
        )

        # -----------------------------------------------------
        # FINAL LLM
        # -----------------------------------------------------

        answer = (
            self.ollama_service
            .generate_response(
                prompt=prompt
            )
        )

        if not answer:

            return (
                "The requested information is not "
                "available in the provided content."
            )

        return answer.strip()

    # =========================================================
    # ANSWER VOTER QUESTION
    # =========================================================

    def _answer_voter_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> str:
        """
        Voter-list question answering.

        Flow:

        User question
                ↓
        VoterQueryService
                ↓
        Structured intent
                ↓
        ┌──────────────────────┐
        │ document question?  │
        └──────────┬───────────┘
                   │
             document context
                   │
                   ↓
               Final LLM

        OR:

        Structured voter query
                ↓
        Python voter lookup
                ↓
        Requested fields
                ↓
        Verified context
                ↓
        Final LLM
        """

        try:

            # -------------------------------------------------
            # QUERY UNDERSTANDING
            # -------------------------------------------------

            understood = (
                self.voter_query_service
                .understand_query(
                    question
                )
            )

            if not understood:

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            lookup_type = understood.get(
                "lookup_type"
            )

            # -------------------------------------------------
            # DOCUMENT-LEVEL QUESTION
            # -------------------------------------------------

            if lookup_type == "document":

                return (
                    self._answer_document_question(
                        connection=connection,
                        document_id=document_id,
                        question=question,
                    )
                )

            requested_fields = (
                understood.get(
                    "requested_fields",
                    [],
                )
            )

            # -------------------------------------------------
            # NORMALIZE REQUESTED FIELDS
            # -------------------------------------------------

            requested_fields = (
                self._normalize_requested_fields(
                    requested_fields
                )
            )

            # -------------------------------------------------
            # GET RECORDS
            # -------------------------------------------------

            records = (
                self._get_voter_records(
                    connection=connection,
                    document_id=document_id,
                )
            )

            # -------------------------------------------------
            # DEBUG
            # -------------------------------------------------

            print(
                "\n========== VOTER RECORD DEBUG =========="
            )

            print(
                f"TOTAL RECORDS: {len(records)}"
            )

            for record in records:

                if record.get(
                    "serial_number"
                ) in {
                    10,
                    11,
                    12,
                    13,
                }:

                    print(record)

            print(
                "========================================\n"
            )

            if not records:

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            # -------------------------------------------------
            # COUNT
            # -------------------------------------------------

            if lookup_type == "count":

                return (
                    f"There are {len(records)} voters."
                )

            # -------------------------------------------------
            # ALL
            # -------------------------------------------------

            if lookup_type == "all":

                matched_records = records

            else:

                # ---------------------------------------------
                # LOOKUP DEBUG
                # ---------------------------------------------

                print(
                    "\n========== LOOKUP DEBUG =========="
                )

                print(
                    f"LOOKUP TYPE: {lookup_type}"
                )

                print(
                    "LOOKUP VALUE: "
                    f"{understood.get('value')}"
                )

                print(
                    "REQUESTED FIELDS: "
                    f"{requested_fields}"
                )

                # ---------------------------------------------
                # PYTHON RECORD LOOKUP
                # ---------------------------------------------

                matched_records = (
                    self.voter_query_service
                    .lookup_records(
                        records=records,
                        query=understood,
                    )
                )

                print(
                    "MATCHED COUNT: "
                    f"{len(matched_records)}"
                )

                for record in matched_records:

                    print(
                        "MATCHED RECORD:"
                    )

                    print(record)

                print(
                    "=================================\n"
                )

            # -------------------------------------------------
            # NO MATCH
            # -------------------------------------------------

            if not matched_records:

                print(
                    "DEBUG: lookup_records() "
                    "returned NO MATCH"
                )

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            # -------------------------------------------------
            # BUILD SAFE CONTEXT
            # -------------------------------------------------

            matched_context = (
                self._build_requested_voter_context(
                    matched_records=matched_records,
                    requested_fields=requested_fields,
                )
            )

            print(
                "\n========== CONTEXT DEBUG =========="
            )

            print(
                matched_context
            )

            print(
                "===================================\n"
            )

            if not matched_context:

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            # -------------------------------------------------
            # SINGLE RECORD + SINGLE FIELD
            # -------------------------------------------------

            if (
                len(matched_records) == 1
                and len(requested_fields) == 1
            ):

                field = requested_fields[0]

                value = (
                    self._get_voter_field_value(
                        matched_records[0],
                        field,
                    )
                )

                if value == "Not available":

                    return (
                        "The requested information is not "
                        "available in the provided content."
                    )

                field_lower = (
                    field
                    .lower()
                    .strip()
                )

                field_display = (
                    field
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                )

                # -------------------------------------------------
                # NAME
                # -------------------------------------------------

                if field_lower == "name":

                    return (
                        f"The voter's name is {value}."
                    )

                # -------------------------------------------------
                # VOTER ID
                # -------------------------------------------------

                if field_lower in {
                    "epic",
                    "epic_numbers",
                    "voter_id",
                }:

                    return (
                        f"The Voter ID is {value}."
                    )

                # -------------------------------------------------
                # AGE
                # -------------------------------------------------

                if field_lower == "age":

                    return (
                        f"The voter's age is {value}."
                    )

                # -------------------------------------------------
                # SERIAL NUMBER
                # -------------------------------------------------

                if field_lower == "serial_number":

                    return (
                        f"The serial number is {value}."
                    )

                # -------------------------------------------------
                # GENDER
                # -------------------------------------------------

                if field_lower == "gender":

                    return (
                        f"The voter's gender is {value}."
                    )

                # -------------------------------------------------
                # HOUSE NUMBER
                # -------------------------------------------------

                if field_lower == "house_number":

                    return (
                        f"The house number is {value}."
                    )

                return (
                    f"The {field_display} is {value}."
                )

            # -------------------------------------------------
            # MULTIPLE FIELDS / MULTIPLE RECORDS
            # -------------------------------------------------

            prompt = f"""
You are answering a question about a voter-list document.

Use ONLY the verified voter information provided below.

USER QUESTION:
{question}

VERIFIED VOTER INFORMATION:
{matched_context}

REQUESTED FIELDS:
{", ".join(requested_fields)}

Rules:
- Answer only from the verified voter information.
- Do not invent information.
- Do not infer missing values.
- Do not use outside knowledge.
- Do not substitute serial number for Voter ID.
- Do not substitute position for Voter ID.
- Do not substitute age for Voter ID.
- Do not substitute house number for Voter ID.
- Voter ID means the actual EPIC number.
- If Voter ID is requested, use only the value shown under
  "Voter ID".
- Do not add information that is not present.
- If a requested value is "Not available", say that the
  requested information is not available.
- Answer naturally and concisely.
"""

            answer = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

            if not answer:

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            return answer.strip()

        except ChatException:

            raise

        except Exception as exc:

            raise ChatException(
                f"Failed to answer voter question: {exc}"
            ) from exc

    # =========================================================
    # GENERIC DOCUMENT CONTEXT
    # =========================================================

    def _get_generic_document_context(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ):

        context_chunks = []

        # -----------------------------------------------------
        # SEMANTIC RETRIEVAL
        # -----------------------------------------------------

        try:

            semantic_chunks = (
                self.retrieval_service
                .retrieve_relevant_chunks(
                    connection=connection,
                    document_id=document_id,
                    question=question,
                    top_k=3,
                )
            )

            if semantic_chunks:

                context_chunks.extend(
                    semantic_chunks
                )

        except Exception:

            semantic_chunks = []

        # -----------------------------------------------------
        # DOCUMENT CHUNKS
        # -----------------------------------------------------

        try:

            document_chunks = (
                self.retrieval_service
                .get_document_chunks(
                    connection=connection,
                    document_id=document_id,
                )
            )

            if document_chunks:

                context_chunks.extend(
                    document_chunks[:2]
                )

        except Exception:

            document_chunks = []

        # -----------------------------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------------------------

        unique_chunks = {}

        for chunk in context_chunks:

            chunk_id = chunk.get(
                "id"
            )

            if chunk_id is not None:

                key = chunk_id

            else:

                key = (
                    chunk.get(
                        "chunk_index"
                    ),
                    chunk.get(
                        "content"
                    ),
                )

            if key not in unique_chunks:

                unique_chunks[key] = chunk

        # -----------------------------------------------------
        # SORT BY CHUNK INDEX
        # -----------------------------------------------------

        result = list(
            unique_chunks.values()
        )

        result.sort(
            key=lambda chunk: (
                chunk.get(
                    "chunk_index"
                )
                if chunk.get(
                    "chunk_index"
                ) is not None
                else 999999
            )
        )

        return result

    # =========================================================
    # MAIN CHAT METHOD
    # =========================================================

    def answer_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> str:

        try:

            # -------------------------------------------------
            # VALIDATE QUESTION
            # -------------------------------------------------

            question = question.strip()

            if not question:

                raise ChatException(
                    "Question cannot be empty."
                )

            # -------------------------------------------------
            # GET DOCUMENT
            # -------------------------------------------------

            document = (
                self.document_repository
                .get_by_id(
                    connection,
                    document_id,
                )
            )

            if not document:

                raise ChatException(
                    "Document not found."
                )

            # -------------------------------------------------
            # DOCUMENT REPOSITORY RETURNS:
            #
            # 0 -> id
            # 1 -> filename
            # 2 -> file_type
            # 3 -> document_type
            # 4 -> file_path
            # 5 -> status
            # 6 -> created_at
            # -------------------------------------------------

            filename = document[1]

            document_type = document[3]

            filename = (
                filename
                if filename
                else ""
            )

            document_type = (
                document_type
                if document_type
                else ""
            )

            # =================================================
            # IMAGE FLOW
            # =================================================

            if self._is_image(
                filename
            ):

                analysis = (
                    self.analysis_repository
                    .get_by_document_id(
                        connection,
                        document_id,
                    )
                )

                if analysis:

                    if isinstance(
                        analysis,
                        dict,
                    ):

                        analysis_text = (
                            analysis.get(
                                "analysis",
                                "",
                            )
                        )

                    else:

                        analysis_text = (
                            getattr(
                                analysis,
                                "analysis",
                                "",
                            )
                        )

                    if analysis_text:

                        prompt = f"""
You are answering a question about an image.

IMAGE ANALYSIS:
{analysis_text}

USER QUESTION:
{question}

Rules:
- Answer only from the image analysis.
- Do not invent information.
- Do not use outside knowledge.
- If the answer is not present, say:

"The requested information is not available in the provided content."

- Keep the answer concise.
"""

                        return (
                            self.ollama_service
                            .generate_response(
                                prompt=prompt
                            )
                        )

            # =================================================
            # VOTER LIST FLOW
            # =================================================

            if document_type == "voter_list":

                return (
                    self._answer_voter_question(
                        connection=connection,
                        document_id=document_id,
                        question=question,
                    )
                )

            # =================================================
            # GENERIC DOCUMENT RAG
            # =================================================

            understood_question = (
                self.ollama_service
                .understand_question(
                    question
                )
            )

            if not understood_question:

                understood_question = question

            # -------------------------------------------------
            # HYBRID RETRIEVAL
            # -------------------------------------------------

            retrieved_chunks = (
                self._get_generic_document_context(
                    connection=connection,
                    document_id=document_id,
                    question=understood_question,
                )
            )

            # -------------------------------------------------
            # NO CONTEXT
            # -------------------------------------------------

            if not retrieved_chunks:

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            # -------------------------------------------------
            # BUILD CONTEXT
            # -------------------------------------------------

            context_parts = []

            for chunk in retrieved_chunks:

                content = chunk.get(
                    "content",
                    "",
                )

                if not content:
                    continue

                chunk_index = chunk.get(
                    "chunk_index"
                )

                page_number = chunk.get(
                    "page_number"
                )

                location = []

                if chunk_index is not None:

                    location.append(
                        f"Chunk {chunk_index}"
                    )

                if page_number is not None:

                    location.append(
                        f"Page {page_number}"
                    )

                if location:

                    label = " | ".join(
                        location
                    )

                    context_parts.append(
                        f"[{label}]\n{content}"
                    )

                else:

                    context_parts.append(
                        content
                    )

            context = "\n\n".join(
                context_parts
            )

            # -------------------------------------------------
            # FINAL LLM PROMPT
            # -------------------------------------------------

            prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document
content.

DOCUMENT CONTENT:
{context}

USER QUESTION:
{question}

Rules:
- Use only information present in the document content.
- Do not use outside knowledge.
- Do not invent facts.
- Do not assume information that is not explicitly present.
- If the answer is not available in the provided content, say:

"The requested information is not available in the provided content."

- For summaries, summarize the information actually present
  in the provided document content.
- For factual questions, give the direct answer first.
- Keep the answer clear and concise.
"""

            # -------------------------------------------------
            # FINAL GENERATION
            # -------------------------------------------------

            answer = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

            if not answer:

                return (
                    "The requested information is not "
                    "available in the provided content."
                )

            return answer.strip()

        except ChatException:

            raise

        except Exception as exc:

            raise ChatException(
                f"Failed to answer question: {exc}"
            ) from exc


# =============================================================
# SERVICE INITIALIZATION
# =============================================================

chunk_repository = ChunkRepository()

analysis_repository = AnalysisRepository()

document_repository = DocumentRepository()

embedding_service = EmbeddingService()

retrieval_service = RetrievalService(
    chunk_repository=chunk_repository,
    embedding_service=embedding_service,
)

ollama_service = OllamaService()

voter_record_service = VoterRecordService()

voter_query_service = VoterQueryService(
    ollama_service=ollama_service,
)

chat_service = ChatService(
    chunk_repository=chunk_repository,
    analysis_repository=analysis_repository,
    document_repository=document_repository,
    retrieval_service=retrieval_service,
    ollama_service=ollama_service,
    voter_record_service=voter_record_service,
    voter_query_service=voter_query_service,
)
