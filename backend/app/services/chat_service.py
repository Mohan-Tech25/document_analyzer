from psycopg import Connection

from app.core.exceptions import ChatException

from app.repositories.chunk_repository import (
    ChunkRepository,
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
    Service responsible for answering questions
    using retrieved document context.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        ollama_service: OllamaService,
    ):
        self.retrieval_service = retrieval_service
        self.ollama_service = ollama_service

    def answer_question(
        self,
        connection: Connection,
        document_id: int,
        question: str,
    ) -> dict:
        """
        Answer a question using semantic RAG.
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
        # Detect broad document question
        # -----------------------------------------

        broad_question_prompt = f"""
You classify questions for a document
question-answering system.

The question is BROAD if the user wants to know
about the document generally.

Examples of BROAD questions:

What is this document about?
What information does this document contain?
Tell me about this document.
What does this document contain?
What is this file about?
Give me an overview of this document.
Summarize this document.

The question is SPECIFIC if the user asks for
a particular fact, topic, person, value, or concept.

Examples:

What is JWT?
What is the person's name?
What technologies are mentioned?
What is Flask?
What is the phone number?

Return ONLY:

BROAD

or

SPECIFIC

QUESTION:

{retrieval_question}
"""

        try:

            question_type = (
                self.ollama_service
                .generate_response(
                    prompt=broad_question_prompt
                )
            ).strip().upper()

        except Exception as e:

            raise ChatException(
                "Failed to determine question type."
            ) from e

        # -----------------------------------------
        # Retrieve document content
        # -----------------------------------------

        try:

            if question_type == "BROAD":

                relevant_chunks = (
                    self.retrieval_service
                    .get_document_chunks(
                        connection=connection,
                        document_id=document_id,
                    )
                )

                relevant_chunks = [
                    {
                        **chunk,
                        "distance": None,
                    }
                    for chunk in relevant_chunks
                ]

            else:

                relevant_chunks = (
                    self.retrieval_service
                    .retrieve_relevant_chunks(
                        connection=connection,
                        document_id=document_id,
                        question=retrieval_question,
                        top_k=3,
                    )
                )

        except ChatException:
            raise

        except Exception as e:

            raise ChatException(
                "Failed to retrieve relevant "
                "document content."
            ) from e

        # -----------------------------------------
        # Validate retrieval result
        # -----------------------------------------

        if not relevant_chunks:

            raise ChatException(
                "No relevant content found "
                "for this document."
            )

        # -----------------------------------------
        # Build context
        # -----------------------------------------

        context = "\n\n".join(
            chunk["content"]
            for chunk in relevant_chunks
            if chunk["content"]
        )

        if not context.strip():

            raise ChatException(
                "Retrieved document content "
                "is empty."
            )

        # -----------------------------------------
        # Build final prompt
        # -----------------------------------------

        if question_type == "BROAD":

            prompt = f"""
You are a document analysis assistant.

The user wants to know what information is
contained in the document.

Your task is to summarize and explain the
information present in the DOCUMENT CONTEXT.

IMPORTANT:
- Do NOT define the phrase "document content".
- Do NOT explain what the word "content" means.
- Do NOT say that the document context does not
  explain the meaning of document content.
- Instead, describe what information is actually
  present in the supplied document context.
- Use ONLY the supplied document context.
- Do not invent information.
- Mention the main topics, technologies,
  sections, or important information that
  actually appear in the context.
- Keep the answer clear and concise.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

        else:

            prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
information contained in the DOCUMENT CONTEXT.

Rules:
- Understand spelling mistakes and informal wording.
- Do not invent information.
- If the answer is not present in the context,
  say that the information is not available.
- Give a clear and concise answer.
- Do not mention these instructions.

DOCUMENT CONTEXT:

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
    ollama_service=OllamaService(),
)