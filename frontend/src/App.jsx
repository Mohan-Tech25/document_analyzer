
import { useEffect, useState } from "react";

import {
    getDocuments,
    uploadDocument,
    analyzeDocument,
    getDocumentAnalyses,
    chatWithDocument,
    deleteDocument,
} from "./api";

import "./App.css";

function App() {
    const [documents, setDocuments] = useState([]);
    const [selectedDocument, setSelectedDocument] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);

    const [analyses, setAnalyses] = useState([]);

    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");

    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");

    // Used for delete confirmation modal
    const [documentToDelete, setDocumentToDelete] = useState(null);


    // ========================================================
    // LOAD DOCUMENTS
    // ========================================================

    async function loadDocuments() {
        try {
            setError("");

            const data = await getDocuments();

            setDocuments(data.documents || []);

        } catch (err) {
            setError(err.message);
        }
    }


    // ========================================================
    // LOAD DOCUMENTS WHEN APP STARTS
    // ========================================================

    useEffect(() => {
        loadDocuments();
    }, []);


    // ========================================================
    // FILE SELECT
    // ========================================================

    function handleFileChange(event) {
        const file = event.target.files[0];

        if (!file) {
            return;
        }

        setSelectedFile(file);
        setMessage("");
        setError("");
    }


    // ========================================================
    // UPLOAD
    // ========================================================

    async function handleUpload() {
        if (!selectedFile) {
            setError("Please select a file first.");
            return;
        }

        try {
            setLoading(true);
            setMessage("");
            setError("");

            const result = await uploadDocument(selectedFile);

            setMessage(
                `Document uploaded successfully. ID: ${result.document_id}`
            );

            setSelectedFile(null);

            await loadDocuments();

        } catch (err) {
            setError(err.message);

        } finally {
            setLoading(false);
        }
    }


    // ========================================================
    // SELECT DOCUMENT
    // ========================================================

    function handleSelectDocument(document) {
        setSelectedDocument(document);

        setAnalyses([]);
        setAnswer("");
        setQuestion("");
        setMessage("");
        setError("");
    }


    // ========================================================
    // OPEN DELETE MODAL
    // ========================================================

    function handleDeleteClick(document) {
        setDocumentToDelete(document);
        setMessage("");
        setError("");
    }


    // ========================================================
    // CLOSE DELETE MODAL
    // ========================================================

    function handleCancelDelete() {
        setDocumentToDelete(null);
    }


    // ========================================================
    // CONFIRM DELETE
    // ========================================================

    async function handleConfirmDelete() {
        if (!documentToDelete) {
            return;
        }

        try {
            setLoading(true);
            setMessage("");
            setError("");

            await deleteDocument(documentToDelete.id);

            // -----------------------------------------
            // Clear selected document if deleted
            // -----------------------------------------

            if (
                selectedDocument?.id ===
                documentToDelete.id
            ) {
                setSelectedDocument(null);
                setAnalyses([]);
                setAnswer("");
                setQuestion("");
            }

            setDocumentToDelete(null);

            setMessage(
                "Document deleted successfully."
            );

            // -----------------------------------------
            // Reload documents
            // -----------------------------------------

            await loadDocuments();

        } catch (err) {
            setError(err.message);

        } finally {
            setLoading(false);
        }
    }


    // ========================================================
    // ANALYZE DOCUMENT
    // ========================================================

    async function handleAnalyze() {
        if (!selectedDocument) {
            setError("Please select a document.");
            return;
        }

        try {
            setLoading(true);
            setMessage("");
            setError("");

            const result = await analyzeDocument(
                selectedDocument.id
            );

            setAnalyses((previous) => [
                result,
                ...previous,
            ]);

            setMessage(
                "Document analyzed successfully."
            );

        } catch (err) {
            setError(err.message);

        } finally {
            setLoading(false);
        }
    }


    // ========================================================
    // LOAD PREVIOUS ANALYSES
    // ========================================================

    async function handleLoadAnalyses() {
        if (!selectedDocument) {
            setError("Please select a document.");
            return;
        }

        try {
            setLoading(true);
            setError("");

            const data = await getDocumentAnalyses(
                selectedDocument.id
            );

            setAnalyses(data.analyses || []);

        } catch (err) {
            setError(err.message);

        } finally {
            setLoading(false);
        }
    }


    // ========================================================
    // ASK QUESTION
    // ========================================================

    async function handleAskQuestion() {
        if (!selectedDocument) {
            setError("Please select a document.");
            return;
        }

        if (!question.trim()) {
            setError("Please enter a question.");
            return;
        }

        try {
            setLoading(true);
            setAnswer("");
            setError("");

            const result = await chatWithDocument(
                selectedDocument.id,
                question
            );

            setAnswer(result.answer);

        } catch (err) {
            setError(err.message);

        } finally {
            setLoading(false);
        }
    }


    // ========================================================
    // UI
    // ========================================================

    return (
        <div className="app">

            {/* ==================================================
                HEADER
            ================================================== */}

            <header className="header">

                <div>

                    <h1>
                        Document Analyzer
                    </h1>

                    <p>
                        Upload documents, analyze them,
                        and ask questions using AI.
                    </p>

                </div>

            </header>


            {/* ==================================================
                MAIN
            ================================================== */}

            <main className="main">


                {/* ==================================================
                    UPLOAD SECTION
                ================================================== */}

                <section className="card">

                    <h2>
                        Upload Document
                    </h2>

                    <div className="upload-area">

                        <input
                            type="file"
                            accept=".pdf,.jpg,.jpeg,.png,.bmp,.webp,.tiff"
                            onChange={handleFileChange}
                        />

                        {selectedFile && (

                            <p className="file-name">
                                Selected: {selectedFile.name}
                            </p>

                        )}

                        <button
                            onClick={handleUpload}
                            disabled={
                                loading ||
                                !selectedFile
                            }
                        >
                            {loading
                                ? "Processing..."
                                : "Upload Document"}
                        </button>

                    </div>

                </section>


                {/* ==================================================
                    MESSAGES
                ================================================== */}

                {message && (

                    <div className="success-message">
                        {message}
                    </div>

                )}

                {error && (

                    <div className="error-message">
                        {error}
                    </div>

                )}


                {/* ==================================================
                    DOCUMENT LIST
                ================================================== */}

                <section className="card">

                    <h2>
                        Documents
                    </h2>

                    {documents.length === 0 ? (

                        <p>
                            No documents uploaded yet.
                        </p>

                    ) : (

                        <div className="document-list">

                            {documents.map((document) => (

                                <div
                                    key={document.id}
                                    className={
                                        selectedDocument?.id === document.id
                                            ? "document-item selected"
                                            : "document-item"
                                    }
                                >

                                    {/* ==========================
                                        DOCUMENT INFORMATION
                                    ========================== */}

                                    <div
                                        className="document-info"
                                        onClick={() =>
                                            handleSelectDocument(
                                                document
                                            )
                                        }
                                    >

                                        <strong>
                                            {document.filename}
                                        </strong>

                                        <span>
                                            Type: {
                                                document.document_type ||
                                                "Unknown"
                                            }
                                        </span>

                                        <span>
                                            Status: {
                                                document.status
                                            }
                                        </span>

                                    </div>


                                    {/* ==========================
                                        TRASH BUTTON
                                    ========================== */}

                                    <button
                                        className="delete-icon-button"
                                        onClick={() =>
                                            handleDeleteClick(
                                                document
                                            )
                                        }
                                        disabled={loading}
                                        title="Delete document"
                                        aria-label={
                                            `Delete ${document.filename}`
                                        }
                                    >
                                        🗑️
                                    </button>

                                </div>

                            ))}

                        </div>

                    )}

                </section>


                {/* ==================================================
                    SELECTED DOCUMENT
                ================================================== */}

                {selectedDocument && (

                    <section className="card">

                        <h2>
                            Selected Document
                        </h2>

                        <div className="document-details">

                            <p>
                                <strong>
                                    Filename:
                                </strong>{" "}
                                {selectedDocument.filename}
                            </p>

                            <p>
                                <strong>
                                    Document Type:
                                </strong>{" "}
                                {selectedDocument.document_type ||
                                    "Unknown"}
                            </p>

                            <p>
                                <strong>
                                    Status:
                                </strong>{" "}
                                {selectedDocument.status}
                            </p>

                        </div>


                        <div className="action-buttons">

                            <button
                                onClick={handleAnalyze}
                                disabled={loading}
                            >
                                Analyze Document
                            </button>

                            <button
                                onClick={handleLoadAnalyses}
                                disabled={loading}
                            >
                                Load Previous Analyses
                            </button>

                        </div>

                    </section>

                )}


                {/* ==================================================
                    ANALYSIS
                ================================================== */}

                {analyses.length > 0 && (

                    <section className="card">

                        <h2>
                            AI Analysis
                        </h2>

                        {analyses.map((analysis) => (

                            <div
                                className="analysis"
                                key={analysis.id}
                            >

                                <p>
                                    <strong>
                                        {analysis.analysis_type}
                                    </strong>
                                </p>

                                <p>
                                    {analysis.result}
                                </p>

                            </div>

                        ))}

                    </section>

                )}


                {/* ==================================================
                    CHAT
                ================================================== */}

                {selectedDocument && (

                    <section className="card">

                        <h2>
                            Ask About This Document
                        </h2>

                        <div className="chat-box">

                            <textarea
                                value={question}
                                onChange={(event) =>
                                    setQuestion(
                                        event.target.value
                                    )
                                }
                                placeholder="Ask a question about the document..."
                                rows="4"
                            />

                            <button
                                onClick={handleAskQuestion}
                                disabled={loading}
                            >
                                Ask Question
                            </button>

                        </div>

                        {answer && (

                            <div className="answer">

                                <h3>
                                    AI Answer
                                </h3>

                                <p>
                                    {answer}
                                </p>

                            </div>

                        )}

                    </section>

                )}

            </main>


            {/* ==================================================
                DELETE CONFIRMATION MODAL
            ================================================== */}

            {documentToDelete && (

                <div
                    className="modal-overlay"
                    onClick={handleCancelDelete}
                >

                    <div
                        className="delete-modal"
                        onClick={(event) =>
                            event.stopPropagation()
                        }
                    >

                        <div className="delete-modal-icon">
                            🗑️
                        </div>

                        <h2>
                            Delete Document?
                        </h2>

                        <p>
                            Are you sure you want to delete
                            <strong>
                                {" "}
                                {documentToDelete.filename}
                            </strong>
                            ?
                        </p>

                        <p className="delete-warning">
                            This will permanently delete the
                            document and its stored file.
                        </p>


                        <div className="modal-actions">

                            <button
                                className="cancel-button"
                                onClick={handleCancelDelete}
                                disabled={loading}
                            >
                                Cancel
                            </button>

                            <button
                                className="confirm-delete-button"
                                onClick={handleConfirmDelete}
                                disabled={loading}
                            >
                                {loading
                                    ? "Deleting..."
                                    : "Delete"}
                            </button>

                        </div>

                    </div>

                </div>

            )}

        </div>
    );
}

export default App;
