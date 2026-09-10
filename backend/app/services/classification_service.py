class ClassificationService:
    """
    Service responsible for identifying
    the type of an uploaded document.
    """

    def classify_document(self, text: str) -> str:
        """
        Identify the likely document type
        from extracted text.
        """

        if not text or not text.strip():
            return "unknown"

        text_lower = text.lower()

        # ========================================================
        # DOCUMENT INDICATORS
        # ========================================================

        scores = {
            "aadhaar": 0,
            "pan": 0,
            "passport": 0,
            "voter_id": 0,
            "driving_license": 0,
            "marksheet": 0,
            "resume": 0,
        }

        # ========================================================
        # AADHAAR
        # ========================================================

        aadhaar_keywords = [
            "aadhaar",
            "aadhar",
            "unique identification authority of india",
            "uidai",
            "enrolment no",
            "enrollment no",
        ]

        for keyword in aadhaar_keywords:
            if keyword in text_lower:
                scores["aadhaar"] += 2

        # ========================================================
        # PAN
        # ========================================================

        pan_keywords = [
            "permanent account number",
            "income tax department",
            "income tax department government of india",
            "pan card",
            "pan no",
            "pan number",
        ]

        for keyword in pan_keywords:
            if keyword in text_lower:
                scores["pan"] += 2

        # ========================================================
        # PASSPORT
        # ========================================================

        passport_keywords = [
            "passport",
            "passport no",
            "passport number",
            "place of issue",
            "date of issue",
            "date of expiry",
            "nationality",
        ]

        for keyword in passport_keywords:
            if keyword in text_lower:
                scores["passport"] += 2

        # ========================================================
        # VOTER ID
        # ========================================================

        voter_keywords = [
            "election commission of india",
            "elector photo identity card",
            "elector photo identity",
            "epic no",
            "epic number",
            "voter id",
            "voter identity card",
        ]

        for keyword in voter_keywords:
            if keyword in text_lower:
                scores["voter_id"] += 2

        # ========================================================
        # DRIVING LICENSE
        # ========================================================

        driving_keywords = [
            "driving licence",
            "driving license",
            "driving licence number",
            "driving license number",
            "transport department",
            "licence to drive",
            "license to drive",
        ]

        for keyword in driving_keywords:
            if keyword in text_lower:
                scores["driving_license"] += 2

        # ========================================================
        # RESUME
        # ========================================================

        resume_keywords = [
            "resume",
            "curriculum vitae",
            "professional summary",
            "professional experience",
            "work experience",
            "employment history",
            "career objective",
            "technical skills",
            "skills",
            "projects",
            "certifications",
            "education",
        ]

        for keyword in resume_keywords:
            if keyword in text_lower:
                scores["resume"] += 1

        # Strong resume indicators
        if "work experience" in text_lower:
            scores["resume"] += 3

        if "professional experience" in text_lower:
            scores["resume"] += 3

        if "technical skills" in text_lower:
            scores["resume"] += 3

        if "projects" in text_lower and "skills" in text_lower:
            scores["resume"] += 2

        # ========================================================
        # MARKSHEET
        # ========================================================

        marksheet_keywords = [
            "marksheet",
            "mark sheet",
            "statement of marks",
            "marks obtained",
            "grade obtained",
            "total marks",
            "percentage",
            "register number",
            "hall ticket number",
            "academic transcript",
            "transcript of marks",
        ]

        for keyword in marksheet_keywords:
            if keyword in text_lower:
                scores["marksheet"] += 2

        # Strong marksheet indicators
        if "marks obtained" in text_lower:
            scores["marksheet"] += 3

        if "statement of marks" in text_lower:
            scores["marksheet"] += 3

        if "total marks" in text_lower:
            scores["marksheet"] += 3

        if "grade obtained" in text_lower:
            scores["marksheet"] += 3

        # ========================================================
        # RESUME PRIORITY
        # ========================================================

        # A resume normally contains education/university/
        # examination information, so strong resume indicators
        # should override weak academic keywords.

        if scores["resume"] >= 4:
            scores["marksheet"] = min(
                scores["marksheet"],
                3
            )

        # ========================================================
        # FIND BEST MATCH
        # ========================================================

        best_document = max(
            scores,
            key=scores.get
        )

        best_score = scores[best_document]

        if best_score == 0:
            return "unknown"

        return best_document


classification_service = ClassificationService()