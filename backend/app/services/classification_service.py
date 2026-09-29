import re


class ClassificationService:
    """
    Service responsible for identifying
    the type of an uploaded document.
    """

    def classify_document(
        self,
        text: str,
    ) -> str:
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
            "voter_list": 0,
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
        # VOTER LIST
        # ========================================================

        voter_list_keywords = [
            # English
            "electoral roll",
            "electoral rolls",
            "voter list",
            "voters list",
            "part number",
            "section number",
            "serial number",
            "elector name",

            # Tamil
            "சட்டமன்றத் தொகுதியின் எண்",
            "சட்டமன்றத் தாகுதியின் எண்",
            "பாகம் எண்",
            "பிரிவு எண் மற்றும் பெயர்",
            "பெயர்",
            "தந்தையின் பெயர்",
            "தாயின் பெயர்",
            "கணவர் பெயர்",
            "வீட்டு எண்",
            "வயது",
            "பாலினம்",
            "மொத்தப் பக்கங்கள்",
            "பட்டியல் வெளியிடப்பட்ட நாள்",
        ]

        for keyword in voter_list_keywords:

            if keyword in text_lower:
                scores["voter_list"] += 2

        # --------------------------------------------------------
        # EPIC NUMBER PATTERN
        #
        # Typical EPIC numbers look like:
        # ABC1234567
        #
        # A voter list contains many such numbers.
        # --------------------------------------------------------

        epic_pattern = r"\b[A-Z]{3}\d{7}\b"

        epic_matches = re.findall(
            epic_pattern,
            text.upper(),
        )

        if len(epic_matches) >= 5:

            scores["voter_list"] += 5

        elif len(epic_matches) >= 2:

            scores["voter_list"] += 2

        # --------------------------------------------------------
        # STRONG VOTER-LIST STRUCTURE
        # --------------------------------------------------------

        voter_structure_indicators = [
            "பெயர்",
            "வயது",
            "பாலினம்",
        ]

        structure_matches = sum(
            1
            for keyword in voter_structure_indicators
            if keyword in text_lower
        )

        if structure_matches == 3:

            scores["voter_list"] += 5

        # --------------------------------------------------------
        # MULTIPLE VOTER SERIAL NUMBERS
        # --------------------------------------------------------

        serial_number_matches = re.findall(
            r"(?m)^\s*\d+\s*$",
            text,
        )

        if len(serial_number_matches) >= 10:

            scores["voter_list"] += 5

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

        if (
            "projects" in text_lower
            and "skills" in text_lower
        ):

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
                3,
            )

        # ========================================================
        # VOTER LIST PRIORITY
        # ========================================================

        # A voter list contains many voter records.
        # If strong voter-list evidence exists, prefer
        # voter_list over a generic voter_id classification.

        if scores["voter_list"] >= 5:

            scores["voter_id"] = min(
                scores["voter_id"],
                2,
            )

        # ========================================================
        # FIND BEST MATCH
        # ========================================================

        best_document = max(
            scores,
            key=scores.get,
        )

        best_score = scores[
            best_document
        ]

        if best_score == 0:

            return "unknown"

        return best_document


classification_service = ClassificationService()