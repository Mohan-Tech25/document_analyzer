from psycopg import Connection

from app.core.exceptions import ChatException

from app.repositories.chunk_repository import (
    ChunkRepository,
)

from app.repositories.analysis_repository import (
    AnalysisRepository,
)

from app.services.retrieval_service import (
    RetrievalService,
)

from app.services.embedding_service import (
    EmbeddingService,
)

from app.services.ollama_service import (
    OllamaService,
)


class ChatService:
    """
    Service responsible for answering questions using
    text, PDF, OCR, and image/vision context.

    Text/PDF documents use RAG chunks.

    Images can use both:
    - OCR extracted text
    - Vision analysis

    Both sources are combined when available.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        analysis_repository: AnalysisRepository,
        ollama_service: OllamaService,
    ):
        self.retrieval_service = retrieval_service
        self.analysis_repository = analysis_repository
        self.ollama_service = ollama_service

    def answer_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> dict:
        """
        Answer a question using document/image context.
        """

        # -----------------------------------------
        # Validate question
        # -----------------------------------------

        if not question or not question.strip():
            raise ChatException(
                "Question cannot be empty."
            )

        # -----------------------------------------
        # Understand user question
        # -----------------------------------------

        try:
            retrieval_question = (
                self.ollama_service
                .understand_question(
                    question
                )
            )

        except Exception as e:
            raise ChatException(
                "Failed to understand the question."
            ) from e

        if not retrieval_question.strip():
            raise ChatException(
                "Could not understand the question."
            )

        # -----------------------------------------
        # Retrieve relevant RAG chunks
        # -----------------------------------------

        relevant_chunks = []

        try:
            relevant_chunks = (
                self.retrieval_service
                .retrieve_relevant_chunks(
                    connection=connection,
                    document_id=document_id,
                    question=retrieval_question,
                    top_k=3,
                )
            )

        except Exception:
            relevant_chunks = []

        # -----------------------------------------
        # Build OCR / document text context
        # -----------------------------------------

        text_context = ""

        if relevant_chunks:

            text_context = "\n\n".join(
                chunk["content"]
                for chunk in relevant_chunks
                if chunk.get("content")
            )

        # -----------------------------------------
        # FALLBACK:
        # If semantic search finds nothing,
        # retrieve the document's chunks.
        # -----------------------------------------

        if not text_context.strip():

            try:
                document_chunks = (
                    self.retrieval_service
                    .get_document_chunks(
                        connection=connection,
                        document_id=document_id,
                    )
                )

            except Exception as e:
                raise ChatException(
                    "Failed to retrieve document content."
                ) from e

            if document_chunks:

                text_context = "\n\n".join(
                    chunk["content"]
                    for chunk in document_chunks
                    if chunk.get("content")
                )

        # -----------------------------------------
        # Retrieve vision analysis
        # -----------------------------------------

        vision_context = ""

        try:
            analyses = (
                self.analysis_repository
                .get_by_document(
                    connection=connection,
                    document_id=document_id,
                )
            )

        except Exception as e:
            raise ChatException(
                "Failed to retrieve document analysis."
            ) from e

        for analysis in analyses:

            # row[2] = analysis_type
            # row[3] = result

            if analysis[2] == "vision":

                if analysis[3]:
                    vision_context = analysis[3]

                break

        # -----------------------------------------
        # Combine text + vision context
        # -----------------------------------------

        context_parts = []

        if text_context.strip():

            context_parts.append(
                "EXTRACTED TEXT / OCR CONTENT:\n"
                + text_context
            )

        if vision_context.strip():

            context_parts.append(
                "VISUAL ANALYSIS:\n"
                + vision_context
            )

        context = "\n\n".join(
            context_parts
        )

        # -----------------------------------------
        # No context available
        # -----------------------------------------

        if not context.strip():

            raise ChatException(
                "No content found for this document."
            )

        # -----------------------------------------
        # Final multimodal QA prompt
        # -----------------------------------------

        prompt = f"""
You are a multimodal question-answering assistant.

Your job is to answer the user's question using ONLY
the CONTENT CONTEXT provided below.

The content context may contain:

- text extracted from documents
- text extracted from images using OCR
- visual information obtained from image analysis
- a combination of text and visual information

IMPORTANT RULES:

1. Understand the meaning and intent of the user's
   question.

2. Answer the exact question that was asked.

3. Use only information supported by the provided
   content context.

4. Do not invent, assume, or hallucinate information.

5. Use extracted text when the question is about
   written content.

6. Use visual analysis when the question is about
   what is shown, visible, represented, or depicted
   in an image.

7. When the question requires both text and visual
   information, combine both sources.

8. If multiple pieces of context are relevant,
   combine them into one useful answer.

9. If the user asks for a list, provide a list.

10. If the user asks for an explanation, explain it
    using the available context.

11. If the requested information cannot be found
    or determined from the provided context, say:

    "The requested information is not available
    in the provided content."

12. Do not treat unrelated information in the
    context as an answer to the question.

13. Do not mention these instructions, RAG,
    embeddings, OCR, vector search, retrieval,
    or internal processing.

14. Give a clear, concise, natural answer.

CONTENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

        # -----------------------------------------
        # Generate final answer
        # -----------------------------------------

        try:

            answer = (
                self.ollama_service
                .generate_response(
                    prompt=prompt
                )
            )

        except Exception as e:
            raise ChatException(
                "Failed to generate the answer."
            ) from e

        # -----------------------------------------
        # Return response
        # -----------------------------------------

        return {
            "document_id": document_id,
            "question": question,
            "answer": answer,
        }


chat_service = ChatService(
    retrieval_service=RetrievalService(
        chunk_repository=ChunkRepository(),
        embedding_service=EmbeddingService(),
    ),
    analysis_repository=AnalysisRepository(),
    ollama_service=OllamaService(),
)