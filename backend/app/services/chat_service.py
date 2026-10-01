import math

from psycopg import Connection

from app.core.exceptions import ChatException

from app.repositories.chunk_repository import ChunkRepository
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.document_repository import DocumentRepository

from app.services.retrieval_service import RetrievalService
from app.services.embedding_service import EmbeddingService
from app.services.ollama_service import OllamaService
from app.services.voter_record_service import VoterRecordService
from app.services.voter_query_service import VoterQueryService


# ================================================================
# CONSTANTS
# ================================================================

UNAVAILABLE_MESSAGE = (
    "The requested information is not available in the provided content."
)

MAX_DOCUMENT_CONTEXT_CHARS = 10000

MAX_DOCUMENT_ANSWER_CONTEXT_CHARS = 7000

MAX_LABEL_VALUE_EVIDENCE_CHARS = 3500

MAX_SEMANTIC_FALLBACK_CHARS = 6000

MAX_VOTER_RECORD_CONTEXT_CHARS = 20000

SEMANTIC_TOP_K = 5

HEADER_CHUNKS = 3

MAX_DOCUMENT_FIELD_CANDIDATES = 30

MAX_DOCUMENT_FIELD_EMBEDDING_CHARS = 300

DOCUMENT_FIELD_SIMILARITY_THRESHOLD = 0.25


class ChatService:

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        analysis_repository: AnalysisRepository,
        retrieval_service: RetrievalService,
        embedding_service: EmbeddingService,
        ollama_service: OllamaService,
        voter_record_service: VoterRecordService,
        voter_query_service: VoterQueryService,
    ):
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.analysis_repository = analysis_repository
        self.retrieval_service = retrieval_service
        self.embedding_service = embedding_service
        self.ollama_service = ollama_service
        self.voter_record_service = voter_record_service
        self.voter_query_service = voter_query_service

    # ============================================================
    # DOCUMENT ROW -> DICT
    # ============================================================

    @staticmethod
    def _document_row_to_dict(row) -> dict:

        if row is None:
            return {}

        if isinstance(row, dict):
            return row

        document = {}

        document["id"] = (
            row[0]
            if len(row) > 0
            else None
        )

        document["file_name"] = (
            row[1]
            if len(row) > 1
            else None
        )

        document["file_type"] = (
            row[2]
            if len(row) > 2
            else None
        )

        document["document_type"] = (
            row[3]
            if len(row) > 3
            else None
        )

        document["file_path"] = (
            row[4]
            if len(row) > 4
            else None
        )

        return document

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def _is_image(
        document: dict,
    ) -> bool:

        document_type = str(
            document.get(
                "document_type",
                "",
            )
        ).lower()

        mime_type = str(
            document.get(
                "mime_type",
                "",
            )
        ).lower()

        file_name = str(
            document.get(
                "file_name",
                "",
            )
        ).lower()

        image_types = {
            "image",
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "tiff",
            "tif",
        }

        image_mimes = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/tiff",
        }

        image_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
        )

        return (
            document_type in image_types
            or mime_type in image_mimes
            or file_name.endswith(
                image_extensions
            )
        )

    @staticmethod
    def _sort_document_chunks(
        chunks: list[dict],
    ) -> list[dict]:

        return sorted(
            chunks,
            key=lambda chunk: (
                int(
                    chunk.get(
                        "chunk_index"
                    )
                )
                if chunk.get(
                    "chunk_index"
                ) is not None
                else 999999999
            ),
        )

    @staticmethod
    def _convert_chunk_rows(
        rows: list,
    ) -> list[dict]:

        if not rows:
            return []

        chunks = []

        for row in rows:

            if not row or len(row) < 4:
                continue

            content = row[2]

            if not content:
                continue

            chunks.append(
                {
                    "id": row[0],
                    "chunk_index": row[1],
                    "content": content,
                    "page_number": row[3],
                    "created_at": (
                        row[4]
                        if len(row) > 4
                        else None
                    ),
                }
            )

        return ChatService._sort_document_chunks(
            chunks
        )

    # ============================================================
    # VOTER HELPERS
    # ============================================================

    def _get_voter_records(
        self,
        connection: Connection,
        document_id: int,
    ) -> list[dict]:

        rows = (
            self.chunk_repository.get_by_document(
                connection=connection,
                document_id=document_id,
            )
        )

        if not rows:
            return []

        return self.voter_record_service.parse_chunk_rows(
            rows
        )

    @staticmethod
    def _format_epic_numbers(
        record: dict,
    ) -> str:

        epic_numbers = record.get(
            "epic_numbers"
        )

        if not epic_numbers:
            return UNAVAILABLE_MESSAGE

        if isinstance(
            epic_numbers,
            list,
        ):

            valid_values = [
                str(value).strip()
                for value in epic_numbers
                if value is not None
                and str(value).strip()
            ]

            if not valid_values:
                return UNAVAILABLE_MESSAGE

            return ", ".join(
                valid_values
            )

        value = str(
            epic_numbers
        ).strip()

        if not value:
            return UNAVAILABLE_MESSAGE

        return value

    @staticmethod
    def _get_voter_field_value(
        record: dict,
        field_name: str,
    ) -> str:

        field_name = (
            field_name
            .lower()
            .strip()
        )

        # --------------------------------------------------------
        # EPIC / VOTER ID
        # --------------------------------------------------------

        if field_name in {
            "epic",
            "epic_number",
            "epic_numbers",
            "voter_id",
            "voterid",
            "voter_id_number",
        }:

            return ChatService._format_epic_numbers(
                record
            )

        # --------------------------------------------------------
        # NORMAL FIELD
        # --------------------------------------------------------

        value = record.get(
            field_name
        )

        if value is None:
            return UNAVAILABLE_MESSAGE

        if isinstance(
            value,
            list,
        ):

            values = [
                str(item).strip()
                for item in value
                if item is not None
                and str(item).strip()
            ]

            if not values:
                return UNAVAILABLE_MESSAGE

            return ", ".join(
                values
            )

        value = str(
            value
        ).strip()

        if not value:
            return UNAVAILABLE_MESSAGE

        return value

    @staticmethod
    def _normalize_requested_fields(
        fields,
    ) -> list[str]:

        if not fields:
            return ["all"]

        if isinstance(
            fields,
            str,
        ):
            fields = [fields]

        normalized = []

        for field in fields:

            if field is None:
                continue

            value = (
                str(field)
                .strip()
                .lower()
            )

            if not value:
                continue

            if value not in normalized:
                normalized.append(
                    value
                )

        return (
            normalized
            if normalized
            else ["all"]
        )

    def _build_complete_voter_context(
        self,
        records: list[dict],
    ) -> str:

        if not records:
            return ""

        sections = []

        current_length = 0

        for record in records:

            lines = []

            for key, value in record.items():

                # ------------------------------------------------
                # Do not expose internal OCR aliases as separate
                # voter IDs.
                # ------------------------------------------------

                if key in {
                    "epic_numbers",
                    "epic",
                    "voter_id",
                }:

                    value = (
                        self._format_epic_numbers(
                            record
                        )
                    )

                if value is None:
                    continue

                if isinstance(
                    value,
                    list,
                ):

                    values = [
                        str(item).strip()
                        for item in value
                        if item is not None
                        and str(item).strip()
                    ]

                    if not values:
                        continue

                    value = ", ".join(
                        values
                    )

                value = str(
                    value
                ).strip()

                if not value:
                    continue

                lines.append(
                    f"{key}: {value}"
                )

            if not lines:
                continue

            section = "\n".join(
                lines
            )

            section_length = (
                len(section) + 2
            )

            if (
                current_length
                + section_length
                > MAX_VOTER_RECORD_CONTEXT_CHARS
            ):
                break

            sections.append(
                section
            )

            current_length += (
                section_length
            )

        return "\n\n".join(
            sections
        )

    # ============================================================
    # STORED ANALYSIS
    # ============================================================

    def _get_stored_analysis(
        self,
        connection: Connection,
        document_id: int,
    ) -> str:

        analysis = (
            self.analysis_repository.get_by_document(
                connection=connection,
                document_id=document_id,
            )
        )

        if not analysis:
            return ""

        if isinstance(
            analysis,
            dict,
        ):

            for key in (
                "analysis",
                "content",
                "result",
                "summary",
                "text",
            ):

                value = analysis.get(
                    key
                )

                if value:
                    return str(
                        value
                    )

            return str(
                analysis
            )

        if isinstance(
            analysis,
            (list, tuple),
        ):

            values = [
                str(value)
                for value in analysis
                if value
            ]

            return "\n".join(
                values
            )

        return str(
            analysis
        )

    # ============================================================
    # DOCUMENT FIELD PARSING
    # ============================================================

    @staticmethod
    def _split_document_label_value(
        line: str,
    ) -> tuple[str, str] | None:

        if not line:
            return None

        normalized_line = (
            " ".join(
                str(line)
                .strip()
                .split()
            )
        )

        if not normalized_line:
            return None

        separators = (
            "：",
            ":",
            "=",
            " - ",
            " – ",
            " — ",
        )

        separator_used = None

        for separator in separators:

            if separator in normalized_line:

                separator_used = (
                    separator
                )

                break

        if separator_used is None:
            return None

        label, value = (
            normalized_line.split(
                separator_used,
                1,
            )
        )

        label = label.strip()
        value = value.strip()

        if not label or not value:
            return None

        if len(label) > 150:
            return None

        if len(value) > 500:
            return None

        if len(label.split()) > 25:
            return None

        return (
            label,
            value,
        )

    def _extract_document_fields(
        self,
        chunks: list[dict],
    ) -> list[dict]:

        if not chunks:
            return []

        fields = []

        seen = set()

        for chunk in chunks:

            content = str(
                chunk.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            for line in content.splitlines():

                normalized_line = (
                    " ".join(
                        line
                        .strip()
                        .split()
                    )
                )

                if not normalized_line:
                    continue

                result = (
                    self._split_document_label_value(
                        normalized_line
                    )
                )

                if not result:
                    continue

                label, value = result

                dedupe_key = (
                    label.lower(),
                    value.lower(),
                )

                if dedupe_key in seen:
                    continue

                seen.add(
                    dedupe_key
                )

                fields.append(
                    {
                        "label": label,
                        "value": value,
                        "source_line": (
                            normalized_line
                        ),
                        "page_number": (
                            chunk.get(
                                "page_number"
                            )
                        ),
                        "chunk_index": (
                            chunk.get(
                                "chunk_index"
                            )
                        ),
                    }
                )

                if (
                    len(fields)
                    >= MAX_DOCUMENT_FIELD_CANDIDATES
                ):
                    return fields

        return fields

    # ============================================================
    # EMBEDDING SIMILARITY
    # ============================================================

    @staticmethod
    def _cosine_similarity(
        vector_a,
        vector_b,
    ) -> float:

        if not vector_a or not vector_b:
            return 0.0

        if len(vector_a) != len(vector_b):
            return 0.0

        dot_product = 0.0

        magnitude_a = 0.0

        magnitude_b = 0.0

        for a, b in zip(
            vector_a,
            vector_b,
        ):

            a = float(a)
            b = float(b)

            dot_product += (
                a * b
            )

            magnitude_a += (
                a * a
            )

            magnitude_b += (
                b * b
            )

        if (
            magnitude_a == 0.0
            or magnitude_b == 0.0
        ):
            return 0.0

        return (
            dot_product
            / (
                math.sqrt(
                    magnitude_a
                )
                *
                math.sqrt(
                    magnitude_b
                )
            )
        )

    def _select_relevant_document_field(
        self,
        question: str,
        fields: list[dict],
    ) -> dict | None:

        if not question:
            return None

        if not fields:
            return None

        try:

            question_embedding = (
                self.embedding_service.create_embedding(
                    question.strip()
                )
            )

        except Exception:
            return None

        if not question_embedding:
            return None

        best_field = None

        best_score = -1.0

        for field in fields:

            label = str(
                field.get(
                    "label",
                    "",
                )
            ).strip()

            value = str(
                field.get(
                    "value",
                    "",
                )
            ).strip()

            if not label:
                continue

            field_text = (
                f"{label}: {value}"
            )

            field_text = (
                field_text[
                    :MAX_DOCUMENT_FIELD_EMBEDDING_CHARS
                ]
            )

            try:

                field_embedding = (
                    self.embedding_service.create_embedding(
                        field_text
                    )
                )

            except Exception:
                continue

            if not field_embedding:
                continue

            score = (
                self._cosine_similarity(
                    question_embedding,
                    field_embedding,
                )
            )

            if score > best_score:

                best_score = score

                best_field = field

        if (
            best_field is None
            or best_score
            < DOCUMENT_FIELD_SIMILARITY_THRESHOLD
        ):
            return None

        selected_field = dict(
            best_field
        )

        selected_field[
            "similarity"
        ] = best_score

        return selected_field

    # ============================================================
    # DOCUMENT CHUNK FORMATTING
    # ============================================================

    def _format_document_chunks(
        self,
        chunks: list[dict],
        max_chars: int,
    ) -> str:

        if not chunks:
            return ""

        sections = []

        current_length = 0

        for chunk in chunks:

            content = str(
                chunk.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            page_number = chunk.get(
                "page_number"
            )

            chunk_index = chunk.get(
                "chunk_index"
            )

            section = (
                f"[Page {page_number} | "
                f"Chunk {chunk_index}]\n"
                f"{content}"
            )

            section_length = (
                len(section) + 2
            )

            if (
                current_length
                + section_length
                > max_chars
            ):

                remaining = (
                    max_chars
                    - current_length
                )

                if remaining > 100:

                    sections.append(
                        section[
                            :remaining
                        ]
                    )

                break

            sections.append(
                section
            )

            current_length += (
                section_length
            )

        return "\n\n".join(
            sections
        )

    def _build_label_value_evidence(
        self,
        chunks: list[dict],
        max_chars: int,
    ) -> str:

        if not chunks:
            return ""

        fields = (
            self._extract_document_fields(
                chunks
            )
        )

        if not fields:
            return ""

        sections = []

        current_length = 0

        for field in fields:

            page_number = field.get(
                "page_number"
            )

            chunk_index = field.get(
                "chunk_index"
            )

            page_text = (
                str(page_number)
                if page_number is not None
                else "N/A"
            )

            chunk_text = (
                str(chunk_index)
                if chunk_index is not None
                else "N/A"
            )

            section = (
                f"[Page {page_text} | "
                f"Chunk {chunk_text}]\n"
                f"Source line: "
                f"{field.get('source_line')}\n"
                f"Document field: "
                f"{field.get('label')}\n"
                f"Document value: "
                f"{field.get('value')}"
            )

            section_length = (
                len(section) + 2
            )

            if (
                current_length
                + section_length
                > max_chars
            ):
                break

            sections.append(
                section
            )

            current_length += (
                section_length
            )

        return "\n\n".join(
            sections
        )

    # ============================================================
    # GENERAL DOCUMENT CONTEXT
    # ============================================================

    def _get_document_level_context(
        self,
        connection: Connection,
        document_id: int,
        question: str,
        document: dict,
    ) -> str:

        all_chunks = (
            self.retrieval_service.get_document_chunks(
                connection=connection,
                document_id=document_id,
            )
        )

        all_chunks = (
            self._sort_document_chunks(
                all_chunks
            )
        )

        if not all_chunks:
            return ""

        header_chunks = all_chunks[
            :HEADER_CHUNKS
        ]

        semantic_chunks = (
            self.retrieval_service.retrieve_relevant_chunks(
                connection=connection,
                document_id=document_id,
                question=question,
                top_k=SEMANTIC_TOP_K,
                document_type=document.get(
                    "document_type"
                ),
            )
        )

        sections = []

        if header_chunks:

            header_text = (
                self._format_document_chunks(
                    header_chunks,
                    MAX_DOCUMENT_CONTEXT_CHARS,
                )
            )

            if header_text:

                sections.append(
                    "DOCUMENT HEADER:\n"
                    + header_text
                )

        if semantic_chunks:

            semantic_text = (
                self._format_document_chunks(
                    semantic_chunks,
                    MAX_DOCUMENT_CONTEXT_CHARS,
                )
            )

            if semantic_text:

                sections.append(
                    "RELEVANT DOCUMENT CONTENT:\n"
                    + semantic_text
                )

        analysis_text = (
            self._get_stored_analysis(
                connection=connection,
                document_id=document_id,
            )
        )

        if analysis_text:

            sections.append(
                "DOCUMENT ANALYSIS:\n"
                + analysis_text
            )

        context = "\n\n".join(
            sections
        )

        return context[
            :MAX_DOCUMENT_CONTEXT_CHARS
        ]

    # ============================================================
    # DOCUMENT ANSWER CONTEXT
    # ============================================================

    def _get_document_answer_context(
        self,
        connection: Connection,
        document_id: int,
        question: str,
        document: dict,
    ) -> str:

        all_chunks = (
            self.retrieval_service.get_document_chunks(
                connection=connection,
                document_id=document_id,
            )
        )

        all_chunks = (
            self._sort_document_chunks(
                all_chunks
            )
        )

        if not all_chunks:
            return ""

        # --------------------------------------------------------
        # HEADER
        # --------------------------------------------------------

        header_chunks = all_chunks[
            :HEADER_CHUNKS
        ]

        header_fields = (
            self._extract_document_fields(
                header_chunks
            )
        )

        selected_header_field = (
            self._select_relevant_document_field(
                question=question,
                fields=header_fields,
            )
        )

        if selected_header_field:

            label = str(
                selected_header_field.get(
                    "label",
                    "",
                )
            ).strip()

            value = str(
                selected_header_field.get(
                    "value",
                    "",
                )
            ).strip()

            source_line = str(
                selected_header_field.get(
                    "source_line",
                    "",
                )
            ).strip()

            page_number = (
                selected_header_field.get(
                    "page_number"
                )
            )

            chunk_index = (
                selected_header_field.get(
                    "chunk_index"
                )
            )

            return (
                "DOCUMENT EVIDENCE:\n"
                f"Field: {label}\n"
                f"Value: {value}\n"
                f"Source: {source_line}\n"
                f"Page: {page_number}\n"
                f"Chunk: {chunk_index}"
            )[
                :MAX_DOCUMENT_ANSWER_CONTEXT_CHARS
            ]

        # --------------------------------------------------------
        # SEMANTIC SEARCH
        # --------------------------------------------------------

        semantic_chunks = (
            self.retrieval_service.retrieve_relevant_chunks(
                connection=connection,
                document_id=document_id,
                question=question,
                top_k=SEMANTIC_TOP_K,
                document_type=document.get(
                    "document_type"
                ),
            )
        )

        if semantic_chunks:

            semantic_fields = (
                self._extract_document_fields(
                    semantic_chunks
                )
            )

            selected_semantic_field = (
                self._select_relevant_document_field(
                    question=question,
                    fields=semantic_fields,
                )
            )

            if selected_semantic_field:

                label = str(
                    selected_semantic_field.get(
                        "label",
                        "",
                    )
                ).strip()

                value = str(
                    selected_semantic_field.get(
                        "value",
                        "",
                    )
                ).strip()

                source_line = str(
                    selected_semantic_field.get(
                        "source_line",
                        "",
                    )
                ).strip()

                page_number = (
                    selected_semantic_field.get(
                        "page_number"
                    )
                )

                chunk_index = (
                    selected_semantic_field.get(
                        "chunk_index"
                    )
                )

                return (
                    "DOCUMENT EVIDENCE:\n"
                    f"Field: {label}\n"
                    f"Value: {value}\n"
                    f"Source: {source_line}\n"
                    f"Page: {page_number}\n"
                    f"Chunk: {chunk_index}"
                )[
                    :MAX_DOCUMENT_ANSWER_CONTEXT_CHARS
                ]

            label_value_evidence = (
                self._build_label_value_evidence(
                    chunks=semantic_chunks,
                    max_chars=MAX_LABEL_VALUE_EVIDENCE_CHARS,
                )
            )

            if label_value_evidence:

                return label_value_evidence[
                    :MAX_DOCUMENT_ANSWER_CONTEXT_CHARS
                ]

            raw_semantic_context = (
                self._format_document_chunks(
                    semantic_chunks,
                    MAX_SEMANTIC_FALLBACK_CHARS,
                )
            )

            if raw_semantic_context:

                return raw_semantic_context[
                    :MAX_DOCUMENT_ANSWER_CONTEXT_CHARS
                ]

        # --------------------------------------------------------
        # HEADER FALLBACK
        # --------------------------------------------------------

        header_evidence = (
            self._build_label_value_evidence(
                chunks=header_chunks,
                max_chars=MAX_LABEL_VALUE_EVIDENCE_CHARS,
            )
        )

        if header_evidence:

            return header_evidence[
                :MAX_DOCUMENT_ANSWER_CONTEXT_CHARS
            ]

        return self._format_document_chunks(
            chunks=header_chunks,
            max_chars=MAX_DOCUMENT_ANSWER_CONTEXT_CHARS,
        )

    # ============================================================
    # GENERIC DOCUMENT QUESTION
    # ============================================================

    def _answer_document_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
        document: dict,
    ) -> str:

        all_chunks = (
            self.retrieval_service.get_document_chunks(
                connection=connection,
                document_id=document_id,
            )
        )

        all_chunks = (
            self._sort_document_chunks(
                all_chunks
            )
        )

        if not all_chunks:
            return UNAVAILABLE_MESSAGE

        # --------------------------------------------------------
        # HEADER FIELD SEARCH
        # --------------------------------------------------------

        header_chunks = all_chunks[
            :HEADER_CHUNKS
        ]

        header_fields = (
            self._extract_document_fields(
                header_chunks
            )
        )

        selected_header_field = (
            self._select_relevant_document_field(
                question=question,
                fields=header_fields,
            )
        )

        if selected_header_field:

            value = str(
                selected_header_field.get(
                    "value",
                    "",
                )
            ).strip()

            if value:
                return value

        # --------------------------------------------------------
        # SEMANTIC FIELD SEARCH
        # --------------------------------------------------------

        semantic_chunks = (
            self.retrieval_service.retrieve_relevant_chunks(
                connection=connection,
                document_id=document_id,
                question=question,
                top_k=SEMANTIC_TOP_K,
                document_type=document.get(
                    "document_type"
                ),
            )
        )

        if semantic_chunks:

            semantic_fields = (
                self._extract_document_fields(
                    semantic_chunks
                )
            )

            selected_semantic_field = (
                self._select_relevant_document_field(
                    question=question,
                    fields=semantic_fields,
                )
            )

            if selected_semantic_field:

                value = str(
                    selected_semantic_field.get(
                        "value",
                        "",
                    )
                ).strip()

                if value:
                    return value

        # --------------------------------------------------------
        # LLM FALLBACK
        # --------------------------------------------------------

        context = (
            self._get_document_answer_context(
                connection=connection,
                document_id=document_id,
                question=question,
                document=document,
            )
        )

        if not context:
            return UNAVAILABLE_MESSAGE

        prompt = f"""
You are a document question-answering system.

Your ONLY source of truth is the DOCUMENT CONTENT provided below.

You MUST answer the USER QUESTION using ONLY information that
appears in the DOCUMENT CONTENT.

Do NOT use:
- general knowledge
- internet knowledge
- training knowledge
- assumptions
- guesses
- outside information

DOCUMENT CONTENT:
----------------
{context}
----------------

USER QUESTION:
{question}

RULES:

1. Answer the user's question using only the document content.
2. Do not add information from your own knowledge.
3. Do not guess missing information.
4. Preserve the original language/script when appropriate.
5. If the question asks for a specific value, give the specific
   value directly.
6. If the question asks for a summary or explanation, summarize
   only the information contained in the document.
7. Do not mention unsupported information.
8. If the requested information cannot be found, reply exactly:

{UNAVAILABLE_MESSAGE}

ANSWER:
""".strip()

        try:

            answer = (
                self.ollama_service.generate_response(
                    prompt=prompt
                )
            )

        except Exception as exc:

            raise ChatException(
                f"Failed to generate document answer: {exc}"
            ) from exc

        if not answer:
            return UNAVAILABLE_MESSAGE

        answer = str(
            answer
        ).strip()

        if not answer:
            return UNAVAILABLE_MESSAGE

        return answer

    # ============================================================
    # VOTER QUESTION
    # ============================================================

    def _answer_voter_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
        document: dict,
    ) -> str:

        # --------------------------------------------------------
        # STEP 1
        # Understand the natural-language question.
        # --------------------------------------------------------

        query_info = (
            self.voter_query_service.understand_query(
                question
            )
        )

        if not query_info:
            return UNAVAILABLE_MESSAGE

        lookup_type = str(
            query_info.get(
                "lookup_type",
                "",
            )
        ).strip().lower()

        # --------------------------------------------------------
        # DOCUMENT-LEVEL QUESTION
        # --------------------------------------------------------

        if lookup_type == "document":

            return self._answer_document_question(
                connection=connection,
                document_id=document_id,
                question=question,
                document=document,
            )

        # --------------------------------------------------------
        # STEP 2
        # Load actual voter records.
        # --------------------------------------------------------

        records = self._get_voter_records(
            connection=connection,
            document_id=document_id,
        )

        if not records:
            return UNAVAILABLE_MESSAGE

        # --------------------------------------------------------
        # STEP 3
        # Perform Python lookup.
        # --------------------------------------------------------

        matched_records = (
            self.voter_query_service.lookup_records(
                records=records,
                query=query_info,
            )
        )

        # --------------------------------------------------------
        # TARGETED NAME LOOKUP FALLBACK
        # --------------------------------------------------------

        if (
            not matched_records
            and lookup_type == "name"
        ):

            lookup_value = query_info.get(
                "value"
            )

            if lookup_value:

                requested_name = (
                    str(
                        lookup_value
                    )
                    .strip()
                    .casefold()
                )

                fallback_matches = []

                for record in records:

                    record_name = record.get(
                        "name"
                    )

                    if not record_name:
                        continue

                    normalized_record_name = (
                        str(
                            record_name
                        )
                        .strip()
                        .casefold()
                    )

                    if not normalized_record_name:
                        continue

                    if (
                        normalized_record_name
                        == requested_name
                    ):

                        fallback_matches.append(
                            record
                        )

                        continue

                    if (
                        requested_name
                        in normalized_record_name
                        or normalized_record_name
                        in requested_name
                    ):

                        fallback_matches.append(
                            record
                        )

                if fallback_matches:

                    matched_records = (
                        fallback_matches
                    )

                    print(
                        "\n"
                        "========== NAME LOOKUP FALLBACK =========="
                    )

                    print(
                        "LOOKUP VALUE:",
                        lookup_value,
                    )

                    print(
                        "FALLBACK MATCHED COUNT:",
                        len(matched_records),
                    )

                    print(
                        "FALLBACK MATCHED RECORD:",
                        matched_records[0],
                    )

                    print(
                        "=========================================="
                    )

        if not matched_records:
            return UNAVAILABLE_MESSAGE

        # --------------------------------------------------------
        # COUNT
        # --------------------------------------------------------

        if lookup_type == "count":

            voter_records = []

            seen_serial_numbers = set()

            for record in matched_records:

                serial_number = record.get(
                    "serial_number"
                )

                try:

                    serial_number = int(
                        serial_number
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if serial_number < 1:
                    continue

                if serial_number in seen_serial_numbers:
                    continue

                seen_serial_numbers.add(
                    serial_number
                )

                voter_records.append(
                    record
                )

            print(
                "VOTER COUNT:",
                len(voter_records),
            )

            if not voter_records:
                return UNAVAILABLE_MESSAGE

            return (
                f"There are "
                f"{len(voter_records)} "
                f"voters."
            )

        # --------------------------------------------------------
        # REQUESTED FIELDS
        # --------------------------------------------------------

        requested_fields = (
            self._normalize_requested_fields(
                query_info.get(
                    "requested_fields"
                )
            )
        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # If an EPIC lookup has already found exactly one voter
        # and the user asks for all information, keep the complete
        # record and let the LLM explain it naturally.
        #
        # Do not convert this into an EPIC-only answer.
        # --------------------------------------------------------

        if (
            lookup_type == "epic"
            and len(matched_records) == 1
            and not requested_fields
        ):

            requested_fields = ["all"]

        # --------------------------------------------------------
        # EXACT SINGLE-FIELD LOOKUP
        #
        # Python remains the source of truth for exact fields.
        # --------------------------------------------------------

        if (
            len(matched_records) == 1
            and len(requested_fields) == 1
            and requested_fields[0] != "all"
        ):

            record = matched_records[0]

            requested_field = (
                requested_fields[0]
            )

            # Normalize EPIC / voter ID aliases.
            if requested_field in {
                "epic",
                "epic_number",
                "epic_numbers",
                "voter_id",
                "voterid",
                "voter_id_number",
            }:

                requested_field = (
                    "epic_numbers"
                )

            exact_value = (
                self._get_voter_field_value(
                    record=record,
                    field_name=requested_field,
                )
            )

            if exact_value == UNAVAILABLE_MESSAGE:
                return UNAVAILABLE_MESSAGE

            return exact_value

        # --------------------------------------------------------
        # MULTI-FIELD / GENERAL VOTER QUESTION
        #
        # This remains LLM-based.
        # --------------------------------------------------------

        complete_context = (
            self._build_complete_voter_context(
                matched_records
            )
        )

        if not complete_context:
            return UNAVAILABLE_MESSAGE

        fields_text = ", ".join(
            requested_fields
        )

        # --------------------------------------------------------
        # ALL-FIELDS INSTRUCTION
        #
        # When the user asks for complete voter information,
        # the LLM must explicitly include every available field.
        # --------------------------------------------------------

        if "all" in requested_fields:

            fields_instruction = """
The user requested COMPLETE information about the voter.

You MUST explicitly include EVERY AVAILABLE FIELD from the
verified voter record.

When present, the answer MUST include:

- Serial Number
- Voter ID / EPIC Number
- Name
- Relation Type
- Relation Name
- House Number
- Age
- Gender
- Any other voter-related field present in the verified record

IMPORTANT:

1. Do NOT omit the Serial Number.

2. Do NOT omit the Voter ID / EPIC Number.

3. Do NOT omit the Name.

4. Do NOT omit the Relation Type when present.

5. Do NOT omit the Relation Name when present.

6. Do NOT omit the House Number when present.

7. Do NOT omit the Age when present.

8. Do NOT omit the Gender when present.

9. Do NOT omit any other available voter-related field.

10. Do NOT consider a field optional merely because another
    field already identifies the voter.

The answer MUST explicitly represent every available field
from the verified record.

For example, if the verified record contains:

serial_number = 60
epic_numbers = ["RMK0188789"]
name = "சம்பு"
relation_type = "தந்தை"
relation_name = "பழனிசாமி"
house_number = "14-9"
age = 47
gender = "ஆண்"

the answer MUST include:

Serial Number: 60
Voter ID / EPIC Number: RMK0188789
Name: சம்பு
Relation Type: தந்தை
Relation Name: பழனிசாமி
House Number: 14-9
Age: 47
Gender: ஆண்

You may write the answer naturally, but every available field
must be clearly represented.

Do NOT return only the person's name.

Do NOT return only the voter ID.

Do NOT return only the person's age.

Do NOT return only the person's house number.

Do NOT return only a short identification sentence.

Do NOT give a partial summary when complete voter information
is available.

The goal is to describe the COMPLETE matched voter record.
""".strip()

        else:

            fields_instruction = f"""
The user requested these specific fields:

{fields_text}

Include EVERY requested field that is present in the
verified voter record.

Do not add unrelated fields unless needed to make the answer
understandable.
""".strip()

        # --------------------------------------------------------
        # FINAL LLM PROMPT
        # --------------------------------------------------------

        prompt = f"""
You are answering a natural-language question using ONLY the
verified voter record data extracted from the uploaded document.

A Python lookup has ALREADY found the matching voter record.

Therefore, the existence of the matching voter is VERIFIED.

VOTER RECORD DATA:
------------------
{complete_context}
------------------

USER QUESTION:
{question}

REQUESTED FIELDS:
{fields_text}

{fields_instruction}

IMPORTANT:

The voter record above is the authoritative source of truth.

The matching record has already been found by the application.

You must answer the user's question from that record.

Do NOT decide that the record is unavailable when the requested
information is present above.

RULES:

1. Use ONLY the voter record data above.
2. Do not use general knowledge.
3. Do not use internet knowledge.
4. Do not invent information.
5. Do not guess information that is not present.
6. Do not confuse serial number with voter ID / EPIC.
7. Serial number is NOT an EPIC number.
8. Position is NOT an EPIC number.
9. Age is NOT an EPIC number.
10. House number is NOT an EPIC number.
11. Preserve Tamil names and original scripts.
12. If the user asks for a person's details, use the matching
    voter record.
13. If the user asks "tell me about" a voter, provide the
    COMPLETE available details from the matched voter record.
14. If REQUESTED FIELDS contains "all", include EVERY available
    field from the matched voter record.
15. If the user identifies a voter using a voter ID / EPIC,
    use that voter record to answer the question.
16. Do not simply repeat the voter ID when the user asks for
    information about the voter.
17. If the user asks who has a particular voter ID, provide the
    person's name if the name is present.
18. If the user asks for the voter ID itself, provide the EPIC
    number from epic_numbers.
19. Only say that information is unavailable when the specific
    requested information is genuinely absent from the verified
    voter record.

ANSWER:
""".strip()

        try:

            answer = (
                self.ollama_service.generate_response(
                    prompt=prompt
                )
            )

        except Exception as exc:

            raise ChatException(
                f"Failed to generate voter answer: {exc}"
            ) from exc

        if not answer:
            return UNAVAILABLE_MESSAGE

        answer = str(
            answer
        ).strip()

        if not answer:
            return UNAVAILABLE_MESSAGE

        # --------------------------------------------------------
        # LLM SAFETY RETRY
        #
        # Sometimes a small model may incorrectly return the
        # unavailable message even though Python already verified
        # the record.
        #
        # Retry only for a matched multi-field/general question.
        # This does NOT replace the normal LLM flow.
        # --------------------------------------------------------

        if (
            answer.casefold()
            == UNAVAILABLE_MESSAGE.casefold()
            and matched_records
            and (
                "all" in requested_fields
                or len(requested_fields) > 1
            )
        ):

            retry_prompt = f"""
You have been given a VERIFIED voter record.

The application already found an exact matching voter.

Do NOT say that the information is unavailable.

Use the verified record below to answer the user's question.

VERIFIED VOTER RECORD:
----------------------
{complete_context}
----------------------

USER QUESTION:
{question}

REQUESTED FIELDS:
{fields_text}

{fields_instruction}

IMPORTANT:

The verified voter record above is the authoritative source
of truth.

The matching voter has already been found by Python.

You MUST answer using the verified record.

Do NOT invent missing values.

Do NOT substitute one field for another.

For example:

- Serial Number is NOT the Voter ID.
- Position is NOT the Voter ID.
- Age is NOT the Voter ID.
- House Number is NOT the Voter ID.

If the user asks to tell them about the voter, provide the
COMPLETE available details from the verified voter record.

When REQUESTED FIELDS contains "all", EVERY available field
must be explicitly represented in the answer.

Preserve Tamil names and original scripts.

The requested information is available in the verified record,
so provide the complete answer directly.

ANSWER:
""".strip()

            try:

                retry_answer = (
                    self.ollama_service.generate_response(
                        prompt=retry_prompt
                    )
                )

            except Exception as exc:

                raise ChatException(
                    f"Failed to generate voter retry answer: {exc}"
                ) from exc

            if retry_answer:

                retry_answer = str(
                    retry_answer
                ).strip()

                if (
                    retry_answer
                    and retry_answer.casefold()
                    != UNAVAILABLE_MESSAGE.casefold()
                ):

                    return retry_answer

        return answer

    # ============================================================
    # MAIN ANSWER METHOD
    # ============================================================

    def answer_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> str:

        if not question or not question.strip():

            raise ChatException(
                "Question cannot be empty."
            )

        question = question.strip()

        # --------------------------------------------------------
        # LOAD DOCUMENT
        # --------------------------------------------------------

        document_row = (
            self.document_repository.get_by_id(
                connection=connection,
                document_id=document_id,
            )
        )

        if not document_row:

            raise ChatException(
                "Document not found."
            )

        document = (
            self._document_row_to_dict(
                document_row
            )
        )

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        if self._is_image(
            document
        ):

            return self._answer_image_question(
                connection=connection,
                document_id=document_id,
                question=question,
                document=document,
            )

        # --------------------------------------------------------
        # DOCUMENT TYPE
        # --------------------------------------------------------

        document_type = str(
            document.get(
                "document_type",
                "",
            )
        ).strip().lower()

        # --------------------------------------------------------
        # VOTER LIST
        # --------------------------------------------------------

        if document_type == "voter_list":

            return self._answer_voter_question(
                connection=connection,
                document_id=document_id,
                question=question,
                document=document,
            )

        # --------------------------------------------------------
        # GENERIC DOCUMENT
        # --------------------------------------------------------

        return self._answer_document_question(
            connection=connection,
            document_id=document_id,
            question=question,
            document=document,
        )


# ================================================================
# SERVICE INITIALIZATION
# ================================================================

document_repository = DocumentRepository()

chunk_repository = ChunkRepository()

analysis_repository = AnalysisRepository()

embedding_service = EmbeddingService()

ollama_service = OllamaService()

retrieval_service = RetrievalService(
    chunk_repository=chunk_repository,
    embedding_service=embedding_service,
)

voter_record_service = VoterRecordService()

voter_query_service = VoterQueryService(
    ollama_service=ollama_service,
)

chat_service = ChatService(
    document_repository=document_repository,
    chunk_repository=chunk_repository,
    analysis_repository=analysis_repository,
    retrieval_service=retrieval_service,
    embedding_service=embedding_service,
    ollama_service=ollama_service,
    voter_record_service=voter_record_service,
    voter_query_service=voter_query_service,
)