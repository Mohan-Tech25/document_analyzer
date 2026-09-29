import os

from psycopg import Connection

from app.core.exceptions import AnalysisException

from app.repositories.document_repository import DocumentRepository
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.chunk_repository import ChunkRepository

from app.services.ollama_service import OllamaService


class AnalysisService:
    """
    Service responsible for document analysis.

    Responsibilities:

    - Retrieve document information
    - Retrieve previous analyses
    - Generate document-level summaries
    - Analyze images using the vision model
    - Manage analysis transactions

    Important:
    Document analysis is different from document question answering.

    This service explains what the document is and what information
    it contains. It should not behave like a lookup system.
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        analysis_repository: AnalysisRepository,
        chunk_repository: ChunkRepository,
        ollama_service: OllamaService,
    ):
        self.document_repository = document_repository
        self.analysis_repository = analysis_repository
        self.chunk_repository = chunk_repository
        self.ollama_service = ollama_service

    # ========================================================
    # GET DOCUMENT
    # ========================================================

    def get_document(
        self,
        connection: Connection,
        document_id: int,
    ):
        return self.document_repository.get_by_id(
            connection=connection,
            document_id=document_id,
        )

    # ========================================================
    # GET DOCUMENT ANALYSES
    # ========================================================

    def get_document_analyses(
        self,
        connection: Connection,
        document_id: int,
    ):
        rows = self.analysis_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        return [
            {
                "id": row[0],
                "document_id": row[1],
                "analysis_type": row[2],
                "result": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    def _is_image(
        self,
        file_type: str,
        file_path: str,
    ) -> bool:
        """
        Determine whether the uploaded document is an image.
        """

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".gif",
            ".tiff",
            ".tif",
        }

        if file_type:

            file_type_lower = file_type.lower()

            if file_type_lower.startswith("image/"):
                return True

            if file_type_lower in {
                "jpg",
                "jpeg",
                "png",
                "webp",
                "bmp",
                "gif",
                "tiff",
                "tif",
            }:
                return True

        if file_path:

            extension = os.path.splitext(
                file_path
            )[1].lower()

            if extension in image_extensions:
                return True

        return False

    # ========================================================
    # BUILD DOCUMENT CONTEXT
    # ========================================================

    def _build_document_context(
        self,
        chunks,
    ) -> str:
        """
        Build a compact representative context for the LLM.

        We intentionally do NOT send every repeated record when
        the document contains a large number of similar chunks.

        The goal is to provide:

        - beginning/header information
        - representative middle content
        - ending information

        This allows the LLM to understand the document as a whole
        without getting distracted by one repeated record.
        """

        valid_chunks = [
            chunk
            for chunk in chunks
            if len(chunk) > 2 and chunk[2]
        ]

        if not valid_chunks:
            return ""

        # ----------------------------------------------------
        # Small documents
        # ----------------------------------------------------

        if len(valid_chunks) <= 6:

            selected_chunks = valid_chunks

        else:

            # ------------------------------------------------
            # Select representative chunks.
            #
            # First 3:
            # Usually contains document/header information.
            #
            # Middle 2:
            # Shows the repeated record/document structure.
            #
            # Last 2:
            # Helps identify ending/context information.
            # ------------------------------------------------

            first_chunks = valid_chunks[:3]

            middle_index = len(valid_chunks) // 2

            middle_start = max(
                0,
                middle_index - 1,
            )

            middle_chunks = valid_chunks[
                middle_start:middle_start + 2
            ]

            last_chunks = valid_chunks[-2:]

            selected_chunks = (
                first_chunks
                + middle_chunks
                + last_chunks
            )

        # ----------------------------------------------------
        # Remove duplicate chunk objects while preserving order
        # ----------------------------------------------------

        unique_chunks = []

        seen = set()

        for chunk in selected_chunks:

            chunk_id = chunk[0]

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            unique_chunks.append(chunk)

        # ----------------------------------------------------
        # Build compact text
        # ----------------------------------------------------

        context_parts = []

        for chunk in unique_chunks:

            text = chunk[2]

            page_number = (
                chunk[3]
                if len(chunk) > 3
                else None
            )

            if page_number is not None:

                context_parts.append(
                    f"[PAGE {page_number}]\n\n{text}"
                )

            else:

                context_parts.append(text)

        return "\n\n".join(
            context_parts
        ).strip()

    # ========================================================
    # SUMMARIZE DOCUMENT
    # ========================================================

    def summarize_document(
        self,
        connection: Connection,
        document_id: int,
    ) -> dict:
        """
        Generate a document-level AI analysis.

        Text/PDF documents use the normal Ollama model.

        Images use the vision-language model.

        The analysis is document-level and generic.

        It should explain:

        - what the document is
        - what it is about
        - its overall context
        - what kinds of information it contains

        It must NOT behave like a question-answering system.
        """

        # -----------------------------------------
        # Retrieve document
        # -----------------------------------------

        document = self.document_repository.get_by_id(
            connection=connection,
            document_id=document_id,
        )

        if document is None:
            raise AnalysisException(
                "Document not found."
            )

        # -----------------------------------------
        # Database structure:
        #
        # row[0] = id
        # row[1] = filename
        # row[2] = file_type
        # row[3] = document_type
        # row[4] = file_path
        # row[5] = status
        # row[6] = created_at
        # -----------------------------------------

        file_type = document[2]
        document_type = document[3]
        file_path = document[4]

        # -----------------------------------------
        # IMAGE ANALYSIS
        # -----------------------------------------

        if self._is_image(
            file_type=file_type,
            file_path=file_path,
        ):
            return self._analyze_image(
                connection=connection,
                document_id=document_id,
                file_path=file_path,
            )

        # -----------------------------------------
        # TEXT/PDF ANALYSIS
        # -----------------------------------------

        chunks = self.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        if not chunks:
            raise AnalysisException(
                "No content found for this document."
            )

        # -----------------------------------------
        # Build full document text
        #
        # Used only for debugging / validation.
        # We do not send all of it to the 3B model.
        # -----------------------------------------

        document_text_parts = []

        for chunk in chunks:

            if not chunk[2]:
                continue

            page_number = chunk[3]

            if page_number is not None:

                document_text_parts.append(
                    f"[PAGE {page_number}]\n\n{chunk[2]}"
                )

            else:

                document_text_parts.append(
                    chunk[2]
                )

        document_text = "\n\n".join(
            document_text_parts
        ).strip()

        if not document_text:
            raise AnalysisException(
                "Document contains no text."
            )

        # -----------------------------------------
        # Document type
        # -----------------------------------------

        detected_document_type = (
            document_type
            if document_type
            else "unknown"
        )

        # -----------------------------------------
        # Build compact representative context
        # -----------------------------------------

        document_context = (
            self._build_document_context(chunks)
        )

        if not document_context:
            raise AnalysisException(
                "Unable to build document context."
            )

        # =================================================
        # DOCUMENT-LEVEL ANALYSIS PROMPT
        # =================================================

        prompt = f"""
You are a document analysis AI.

Analyze the document as a WHOLE DOCUMENT.

Your job is to explain the document's context, type, subject, purpose, structure,
and the kinds of information it contains.

IMPORTANT:
This is DOCUMENT ANALYSIS, not question answering.

Do NOT select or describe one individual record, voter, person, student,
patient, transaction, or row unless the document itself is specifically
about that single individual.

The supplied context may contain repeated records. Treat them as a collection.

DOCUMENT CONTEXT:
{document_context}

TASK:

Provide a concise document-level analysis using this structure:

Summary:
Explain what this document is about in 2-4 sentences.

Document Context:
- Document type:
- Subject:
- Purpose/context:
- Location/administrative information:
- Important dates:

Information Structure:
- Main categories of information:
- Common fields/data types:
- Overall nature of the records/content:

STRICT GROUNDING RULES:

1. Base the analysis only on the supplied document context.
2. Never invent facts that are not supported by the document.
3. Never infer an issuing authority, organization, country, state, department,
   government body, or institution unless it is explicitly stated.
4. Never use phrases such as "implied", "likely", "probably", or "suggests"
   to fill missing document information.
5. Do not assume a field exists just because it is common for this type of
   document.
6. If a requested category is not explicitly available, write:
   "Not explicitly stated in the document."
7. Do not call a document a "representative sample" unless the document
   explicitly says that it is a sample.
8. Do not select an arbitrary person or record as the document's subject.
9. Do not provide detailed information about individual records.
10. You may classify the document based on the content itself, but do not
    invent external metadata.
11. If the document explicitly contains a specific administrative value,
    date, section, part number, constituency, title, or other metadata,
    report it accurately.
12. If the document contains repeated records, describe the record structure
    and information categories rather than listing individual records.

IMPORTANT OUTPUT RULE:

Return ONLY the document analysis.
Do not mention these instructions.
Do not ask questions.
Respond in ENGLISH.
"""

        # -----------------------------------------
        # DEBUG INFORMATION
        # -----------------------------------------

        print(
            "\n"
            "==================================================\n"
            "ANALYSIS SERVICE DEBUG\n"
            "=================================================="
        )

        print(
            f"Document ID            : {document_id}"
        )

        print(
            f"Document Type          : {detected_document_type}"
        )

        print(
            f"File Type              : {file_type}"
        )

        print(
            f"File Path              : {file_path}"
        )

        print(
            f"Total Chunk Count      : {len(chunks)}"
        )

        print(
            f"Full Document Size     : "
            f"{len(document_text)} characters"
        )

        print(
            f"LLM Context Size       : "
            f"{len(document_context)} characters"
        )

        print(
            f"Prompt Size            : "
            f"{len(prompt)} characters"
        )

        print(
            "--------------------------------------------------"
        )

        print(
            "LLM DOCUMENT CONTEXT:"
        )

        print(
            document_context
        )

        print(
            "--------------------------------------------------"
        )

        print(
            "OLLAMA REQUEST STARTING..."
        )

        print(
            "==================================================\n"
        )

        # -----------------------------------------
        # Generate summary and save analysis
        # -----------------------------------------

        try:

            summary = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

            print(
                "\n========== OLLAMA RESPONSE =========="
            )

            print(
                summary
            )

            print(
                "======================================\n"
            )

            if not summary or not summary.strip():
                raise AnalysisException(
                    "The AI model returned an empty analysis."
                )

            # -----------------------------------------
            # Save analysis
            # -----------------------------------------

            print(
                "DEBUG: Creating analysis database record..."
            )

            analysis_id = (
                self.analysis_repository.create(
                    connection=connection,
                    document_id=document_id,
                    analysis_type="summary",
                    result=summary,
                )
            )

            print(
                f"DEBUG: Analysis record created. "
                f"analysis_id={analysis_id}"
            )

            print(
                "DEBUG: Committing transaction..."
            )

            connection.commit()

            print(
                "DEBUG: Transaction committed successfully."
            )

            return {
                "analysis_id": analysis_id,
                "document_id": document_id,
                "analysis_type": "summary",
                "result": summary,
            }

        except AnalysisException:

            print(
                "\n========== ANALYSIS EXCEPTION =========="
            )

            print(
                "AnalysisException occurred."
            )

            connection.rollback()

            print(
                "DEBUG: Transaction rolled back."
            )

            print(
                "========================================\n"
            )

            raise

        except Exception as e:

            print(
                "\n"
                "==================================================\n"
                "ANALYSIS SERVICE ERROR\n"
                "=================================================="
            )

            print(
                f"ERROR TYPE: {type(e).__name__}"
            )

            print(
                f"ERROR: {e}"
            )

            print(
                f"DOCUMENT ID: {document_id}"
            )

            print(
                f"CHUNK COUNT: {len(chunks)}"
            )

            print(
                f"DOCUMENT TEXT SIZE: "
                f"{len(document_text)} characters"
            )

            print(
                f"LLM CONTEXT SIZE: "
                f"{len(document_context)} characters"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                "FULL TRACEBACK:"
            )

            import traceback

            traceback.print_exc()

            print(
                "--------------------------------------------------"
            )

            try:

                connection.rollback()

                print(
                    "DEBUG: Transaction rolled back successfully."
                )

            except Exception as rollback_error:

                print(
                    "DEBUG: Rollback itself failed:"
                )

                print(
                    f"{type(rollback_error).__name__}: "
                    f"{rollback_error}"
                )

            print(
                "==================================================\n"
            )

            raise AnalysisException(
                f"Document analysis failed: "
                f"{type(e).__name__}: {e}"
            ) from e

    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    def _analyze_image(
        self,
        connection: Connection,
        document_id: int,
        file_path: str,
    ) -> dict:
        """
        Analyze an image using the Ollama vision model.

        OCR text is included when available, but the actual
        image is always sent to the vision model.
        """

        if not file_path:
            raise AnalysisException(
                "Image file path is missing."
            )

        if not os.path.exists(file_path):
            raise AnalysisException(
                "Image file could not be found."
            )

        # -----------------------------------------
        # Get OCR text if available
        # -----------------------------------------

        chunks = self.chunk_repository.get_by_document(
            connection=connection,
            document_id=document_id,
        )

        ocr_text = ""

        if chunks:

            ocr_text = "\n\n".join(
                chunk[2]
                for chunk in chunks
                if chunk[2]
            ).strip()

        # -----------------------------------------
        # Build vision prompt
        # -----------------------------------------

        if ocr_text:

            prompt = f"""
You are analyzing an image containing a document.

Respond in ENGLISH.

Analyze the DOCUMENT AS A WHOLE.

Use both:

1. The actual image.
2. The OCR text.

This is document analysis, not question answering.

Identify:

- what kind of document it is
- what it is about
- its overall purpose/context
- important document-level information
- common information categories
- common record/table structure when present
- important visible document elements

If the image contains many repeated records,
treat them as a collection.

Do NOT select one arbitrary person,
record, row, or item.

Do NOT make one individual's information
the main summary.

Do not invent information.

If something is unclear, say so.

Return a concise document-level analysis.

OCR TEXT:

{ocr_text}
"""

        else:

            prompt = """
You are analyzing an image containing a document.

Respond in ENGLISH.

Analyze the DOCUMENT AS A WHOLE.

Identify:

- document type
- subject
- purpose/context
- important document-level information
- information categories
- table or record structure when visible
- important visible text or document elements

If the document contains many records,
describe the collection rather than one record.

Do NOT select one arbitrary person,
record, row, or item.

Do not invent information.

If something is unclear, say so.

Keep the response concise and document-level.
"""

        # -----------------------------------------
        # Generate vision response
        # -----------------------------------------

        try:

            print(
                "\n========== IMAGE ANALYSIS DEBUG =========="
            )

            print(
                f"Document ID: {document_id}"
            )

            print(
                f"Image Path: {file_path}"
            )

            print(
                f"OCR Text Size: {len(ocr_text)} characters"
            )

            print(
                f"Vision Prompt Size: {len(prompt)} characters"
            )

            print(
                "Calling Ollama vision model..."
            )

            result = (
                self.ollama_service
                .generate_vision_response(
                    image_path=file_path,
                    prompt=prompt,
                )
            )

            print(
                "Ollama vision response received."
            )

            print(
                "\n========== VISION RESPONSE =========="
            )

            print(
                result
            )

            print(
                "======================================\n"
            )

            if not result or not result.strip():
                raise AnalysisException(
                    "The vision model returned an empty analysis."
                )

            analysis_id = (
                self.analysis_repository.create(
                    connection=connection,
                    document_id=document_id,
                    analysis_type="vision",
                    result=result,
                )
            )

            print(
                f"Vision analysis record created: "
                f"{analysis_id}"
            )

            connection.commit()

            print(
                "Image analysis transaction committed."
            )

            print(
                "==========================================\n"
            )

            return {
                "analysis_id": analysis_id,
                "document_id": document_id,
                "analysis_type": "vision",
                "result": result,
            }

        except AnalysisException:

            connection.rollback()

            raise

        except Exception as e:

            print(
                "\n"
                "==================================================\n"
                "IMAGE ANALYSIS ERROR\n"
                "=================================================="
            )

            print(
                f"ERROR TYPE: {type(e).__name__}"
            )

            print(
                f"ERROR: {e}"
            )

            print(
                f"DOCUMENT ID: {document_id}"
            )

            print(
                f"IMAGE PATH: {file_path}"
            )

            print(
                "--------------------------------------------------"
            )

            print(
                "FULL TRACEBACK:"
            )

            import traceback

            traceback.print_exc()

            print(
                "--------------------------------------------------"
            )

            try:

                connection.rollback()

                print(
                    "DEBUG: Image analysis transaction "
                    "rolled back successfully."
                )

            except Exception as rollback_error:

                print(
                    "DEBUG: Image rollback failed:"
                )

                print(
                    f"{type(rollback_error).__name__}: "
                    f"{rollback_error}"
                )

            print(
                "==================================================\n"
            )

            raise AnalysisException(
                f"Image analysis failed: "
                f"{type(e).__name__}: {e}"
            ) from e


analysis_service = AnalysisService(
    document_repository=DocumentRepository(),
    analysis_repository=AnalysisRepository(),
    chunk_repository=ChunkRepository(),
    ollama_service=OllamaService(),
)