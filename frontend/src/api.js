const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "/api";

// ============================================================
// GET ALL DOCUMENTS
// ============================================================

export async function getDocuments() {
    const response = await fetch(
        `${API_BASE_URL}/documents/`
    );

    if (!response.ok) {
        throw new Error("Failed to fetch documents.");
    }

    return response.json();
}


// ============================================================
// GET DOCUMENT BY ID
// ============================================================

export async function getDocument(documentId) {
    const response = await fetch(
        `${API_BASE_URL}/documents/${documentId}`
    );

    if (!response.ok) {
        throw new Error("Failed to fetch document.");
    }

    return response.json();
}

// ============================================================
// DELETE DOCUMENT
// ============================================================

export async function deleteDocument(documentId) {
    const response = await fetch(
        `${API_BASE_URL}/documents/${documentId}`,
        {
            method: "DELETE",
        }
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
            errorData?.detail || "Failed to delete document."
        );
    }

    return response.json();
}

// ============================================================
// UPLOAD DOCUMENT
// ============================================================

export async function uploadDocument(file) {
    const formData = new FormData();

    formData.append("file", file);

    const response = await fetch(
        `${API_BASE_URL}/documents/upload`,
        {
            method: "POST",
            body: formData,
        }
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
            errorData?.detail || "Failed to upload document."
        );
    }

    return response.json();
}


// ============================================================
// ANALYZE DOCUMENT
// ============================================================

export async function analyzeDocument(documentId) {
    const response = await fetch(
        `${API_BASE_URL}/documents/${documentId}/analyze`,
        {
            method: "POST",
        }
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
            errorData?.detail || "Failed to analyze document."
        );
    }

    return response.json();
}


// ============================================================
// GET DOCUMENT ANALYSES
// ============================================================

export async function getDocumentAnalyses(documentId) {
    const response = await fetch(
        `${API_BASE_URL}/documents/${documentId}/analysis`
    );

    if (!response.ok) {
        throw new Error("Failed to fetch document analyses.");
    }

    return response.json();
}


// ============================================================
// CHAT WITH DOCUMENT
// ============================================================

export async function chatWithDocument(documentId, question) {
    const response = await fetch(
        `${API_BASE_URL}/chat`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                document_id: documentId,
                question: question,
            }),
        }
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => null);

        throw new Error(
            errorData?.detail || "Failed to get answer."
        );
    }

    return response.json();
}