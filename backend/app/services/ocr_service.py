import math
import os
import re
import statistics
from typing import Any, Optional

from paddleocr import PaddleOCR


class OCRService:

    SERIAL_PATTERN = re.compile(
        r"^\s*(\d{1,4})\s*$"
    )

    EPIC_PATTERN = re.compile(
        r"[A-Z]{3}\d{7}",
        re.IGNORECASE,
    )

    # A normal document may contain numbers.
    # We only consider voter-grid detection when
    # several serial-number anchors are present.
    MIN_SERIAL_ANCHORS_FOR_VOTER = 3

    COLUMN_X_TOLERANCE_FACTOR = 2.5
    ROW_Y_TOLERANCE_FACTOR = 1.5

    MIN_COLUMN_SHARE = 0.25
    COLUMN_LATTICE_TOLERANCE = 0.15

    MIN_ROW_SPACING_RATIO = 0.6
    FALLBACK_ROW_PITCH_FACTOR = 6.0

    TOP_TOLERANCE_FACTOR = 0.5
    CARD_HEIGHT_RATIO = 0.92

    LEFT_EDGE_SEARCH_RATIO = 0.5
    LEFT_EDGE_CLUSTER_RATIO = 0.04
    LEFT_EDGE_FALLBACK_RATIO = 0.30

    COLUMN_PAD_RATIO = 0.04
    BANNER_WIDTH_RATIO = 1.05

    LINE_TOLERANCE_FACTOR = 0.6
    MIN_ORPHAN_LINES = 3

    def __init__(
        self,
        debug_grouping: Optional[bool] = None,
    ):
        self._ocr_instances: dict[str, PaddleOCR] = {}

        if debug_grouping is None:
            debug_grouping = (
                os.getenv(
                    "OCR_DEBUG_GROUPING",
                    "1",
                ).strip() != "0"
            )

        self.debug_grouping = debug_grouping

    # ==========================================================
    # OCR INSTANCE
    # ==========================================================

    def _get_ocr(
        self,
        language: str = "en",
    ) -> PaddleOCR:

        if language not in self._ocr_instances:

            self._ocr_instances[language] = PaddleOCR(
                lang=language,
                device="cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

        return self._ocr_instances[language]

    # ==========================================================
    # SINGLE-PASS SCANNED PAGE OCR
    # ==========================================================

    def extract_scanned_page(
        self,
        image_path: str,
        language: str = "en",
        page_number: int = 1,
    ) -> dict:

        ocr = self._get_ocr(language)

        self._log(
            f"Starting OCR for scanned page "
            f"{page_number}."
        )

        # PaddleOCR runs ONLY ONCE.
        results = ocr.predict(
            image_path
        )

        all_items = []

        for result in results:

            data = self._get_result_data(
                result
            )

            items = self._result_to_items(
                data
            )

            if not items:
                continue

            all_items.extend(
                items
            )

        if not all_items:

            self._log(
                f"No OCR text detected on page "
                f"{page_number}."
            )

            return {
                "is_voter_list": False,
                "text": "",
                "records": [],
            }

        # Assign IDs and dimensions.
        for index, item in enumerate(
            all_items
        ):

            item["id"] = index

            item.setdefault(
                "w",
                item["x2"] - item["x1"],
            )

            item.setdefault(
                "h",
                item["y2"] - item["y1"],
            )

        self._log(
            f"OCR completed for page "
            f"{page_number}. "
            f"Boxes={len(all_items)}"
        )

        # ------------------------------------------------------
        # FIRST:
        # Build normal OCR text.
        # ------------------------------------------------------

        ordered_items = self._reading_order(
            all_items,
            all_items,
        )

        normal_text = "\n".join(
            item["text"]
            for item in ordered_items
            if item["text"].strip()
        ).strip()

        # ------------------------------------------------------
        # SECOND:
        # Quick voter-list detection.
        # ------------------------------------------------------

        anchors = self._find_serial_anchors(
            all_items,
            debug=False,
        )

        if (
            len(anchors)
            < self.MIN_SERIAL_ANCHORS_FOR_VOTER
        ):

            self._log(
                f"Normal scanned document detected "
                f"on page {page_number}. "
                f"Serial anchors="
                f"{len(anchors)}"
            )

            return {
                "is_voter_list": False,
                "text": normal_text,
                "records": [],
            }

        # ------------------------------------------------------
        # POSSIBLE VOTER LIST
        # ------------------------------------------------------

        self._log(
            f"Possible voter-list structure detected "
            f"on page {page_number}. "
            f"SerialAnchors={len(anchors)}"
        )

        records = (
            self._build_voter_records_from_items(
                items=all_items,
                anchors=anchors,
                page_number=page_number,
                debug_header=True,
            )
        )

        if records:

            self._log(
                f"Voter-list extraction successful. "
                f"Records={len(records)}"
            )

            return {
                "is_voter_list": True,
                "text": "",
                "records": records,
            }

        # ------------------------------------------------------
        # GRID FAILED
        # ------------------------------------------------------

        self._log(
            "Possible serial numbers were found, "
            "but voter-grid extraction failed. "
            "Using normal OCR text."
        )

        return {
            "is_voter_list": False,
            "text": normal_text,
            "records": [],
        }

    # ==========================================================
    # NORMAL SCANNED DOCUMENT OCR
    # ==========================================================

    def extract_document_text(
        self,
        image_path: str,
        language: str = "en",
    ) -> str:

        ocr = self._get_ocr(
            language
        )

        results = ocr.predict(
            image_path
        )

        page_text_parts = []

        for result in results:

            data = self._get_result_data(
                result
            )

            items = self._result_to_items(
                data
            )

            if not items:
                continue

            ordered_items = self._reading_order(
                items,
                items,
            )

            text = "\n".join(
                item["text"]
                for item in ordered_items
                if item["text"].strip()
            ).strip()

            if text:

                page_text_parts.append(
                    text
                )

        return "\n\n".join(
            page_text_parts
        ).strip()

    # ==========================================================
    # VOTER RECORD OCR
    # ==========================================================

    def extract_voter_records(
        self,
        image_path: str,
        language: str = "en",
        include_page_text: bool = False,
    ) -> list[Any]:

        ocr = self._get_ocr(
            language
        )

        results = ocr.predict(
            image_path
        )

        records = []

        for page_number, result in enumerate(
            results,
            start=1,
        ):

            data = self._get_result_data(
                result
            )

            items = self._result_to_items(
                data
            )

            if not items:
                continue

            page_records = (
                self._build_voter_records(
                    items=items,
                    include_page_text=include_page_text,
                    page_number=page_number,
                )
            )

            records.extend(
                page_records
            )

        return records

    # ==========================================================
    # RESULT DATA EXTRACTION
    # ==========================================================

    def _get_result_data(
        self,
        result: Any,
    ) -> dict:

        if isinstance(
            result,
            dict,
        ):
            return result

        if hasattr(
            result,
            "json",
        ):

            try:

                value = result.json()

                if isinstance(
                    value,
                    dict,
                ):

                    return value

            except Exception:
                pass

        if hasattr(
            result,
            "to_json",
        ):

            try:

                value = result.to_json()

                if isinstance(
                    value,
                    str,
                ):

                    import json

                    value = json.loads(
                        value
                    )

                if isinstance(
                    value,
                    dict,
                ):

                    return value

            except Exception:
                pass

        return {}

    # ==========================================================
    # CONVERT PADDLE RESULT TO ITEMS
    # ==========================================================

    def _result_to_items(
        self,
        data: dict,
    ) -> list[dict]:

        items = []

        rec_texts = data.get(
            "rec_texts",
            [],
        )

        rec_scores = data.get(
            "rec_scores",
            [],
        )

        rec_polys = data.get(
            "rec_polys",
            [],
        )

        if (
            rec_texts is None
            or len(rec_texts) == 0
        ):

            return items

        for index, text in enumerate(
            rec_texts
        ):

            if text is None:
                continue

            text = str(
                text
            ).strip()

            if not text:
                continue

            score = None

            if (
                rec_scores is not None
                and index < len(rec_scores)
            ):

                try:

                    score = float(
                        rec_scores[index]
                    )

                except Exception:

                    score = None

            polygon = None

            if (
                rec_polys is not None
                and index < len(rec_polys)
            ):

                polygon = rec_polys[
                    index
                ]

            if polygon is None:
                continue

            try:

                xs = [
                    float(point[0])
                    for point in polygon
                ]

                ys = [
                    float(point[1])
                    for point in polygon
                ]

                if not xs or not ys:
                    continue

                x1 = min(xs)
                y1 = min(ys)
                x2 = max(xs)
                y2 = max(ys)

            except Exception:

                continue

            items.append(
                {
                    "text": text,
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                }
            )

        return items

    # ==========================================================
    # BUILD VOTER RECORDS
    # ==========================================================

    def _build_voter_records(
        self,
        items: list[dict],
        include_page_text: bool = False,
        page_number: int = 1,
    ) -> list[Any]:

        for index, item in enumerate(
            items
        ):

            item["id"] = index

            item.setdefault(
                "w",
                item["x2"] - item["x1"],
            )

            item.setdefault(
                "h",
                item["y2"] - item["y1"],
            )

        self._log()

        self._log(
            "=" * 72
        )

        self._log(
            f"VOTER GRID DEBUG - page {page_number} "
            f"({len(items)} OCR boxes)"
        )

        self._log(
            "=" * 72
        )

        anchors = self._find_serial_anchors(
            items,
            debug=False,
        )

        if (
            len(anchors)
            < self.MIN_SERIAL_ANCHORS_FOR_VOTER
        ):

            self._log(
                "Not enough serial-number boxes "
                "to consider this a voter page."
            )

            if include_page_text:

                ordered_items = self._reading_order(
                    items,
                    items,
                )

                page_text = "\n".join(
                    item["text"]
                    for item in ordered_items
                ).strip()

                if page_text:

                    return [
                        page_text
                    ]

            return []

        return self._build_voter_records_from_items(
            items=items,
            anchors=anchors,
            page_number=page_number,
            debug_header=False,
        )

    # ==========================================================
    # BUILD VOTER RECORDS FROM OCR ITEMS
    # ==========================================================

    def _build_voter_records_from_items(
        self,
        items: list[dict],
        anchors: list[dict],
        page_number: int = 1,
        debug_header: bool = True,
    ) -> list[dict]:

        if not items:
            return []

        if not anchors:
            return []

        for index, item in enumerate(
            items
        ):

            item["id"] = index

            item.setdefault(
                "w",
                item["x2"] - item["x1"],
            )

            item.setdefault(
                "h",
                item["y2"] - item["y1"],
            )

        if debug_header:

            self._log()

            self._log(
                "=" * 72
            )

            self._log(
                f"VOTER GRID DEBUG - page {page_number} "
                f"({len(items)} OCR boxes)"
            )

            self._log(
                "=" * 72
            )

        grid = self._detect_grid(
            items,
            anchors,
        )

        if not grid:

            self._log(
                "Grid detection failed."
            )

            return []

        cells = self._assign_items_to_cells(
            items,
            grid,
        )

        records = self._format_cards(
            cells,
            grid,
        )

        self._log_grid(
            grid
        )

        self._log_summary(
            records
        )

        return records

    # ==========================================================
    # SERIAL ANCHOR DETECTION
    # ==========================================================

    def _find_serial_anchors(
        self,
        items: list[dict],
        debug: bool = False,
    ) -> list[dict]:

        anchors = []

        if debug:

            self._log()

            self._log(
                "---------- SERIAL ANCHOR DEBUG ----------"
            )

        for item in sorted(
            items,
            key=lambda item: (
                item["y"],
                item["x"],
            ),
        ):

            text = item["text"]

            match = self.SERIAL_PATTERN.match(
                text
            )

            if debug:

                self._log(
                    f"TEXT={text!r} "
                    f"X={item['x']:.1f} "
                    f"Y={item['y']:.1f} "
                    f"W={item['w']:.1f} "
                    f"H={item['h']:.1f} "
                    f"MATCH="
                    f"{'YES' if match else 'NO'}"
                )

            if not match:
                continue

            serial = int(
                match.group(1)
            )

            if 1 <= serial <= 9999:

                item["serial"] = serial

                anchors.append(
                    item
                )

        if debug:

            self._log(
                f"Serial anchors detected: "
                f"{len(anchors)}"
            )

            self._log(
                "------------------------------------------"
            )

            self._log()

        return anchors

    # ==========================================================
    # GRID DETECTION
    # ==========================================================

    def _detect_grid(
        self,
        items: list[dict],
        anchors: list[dict],
    ) -> Optional[dict]:

        if not anchors:
            return None

        anchor_heights = [
            anchor["h"]
            for anchor in anchors
            if anchor["h"] > 0
        ]

        if not anchor_heights:
            return None

        median_height = statistics.median(
            anchor_heights
        )

        x_values = [
            anchor["x"]
            for anchor in anchors
        ]

        y_values = [
            anchor["y"]
            for anchor in anchors
        ]

        columns = self._cluster_values(
            x_values,
            median_height
            * self.COLUMN_X_TOLERANCE_FACTOR,
            axis="x",
        )

        if not columns:
            return None

        rows = self._cluster_values(
            y_values,
            median_height
            * self.ROW_Y_TOLERANCE_FACTOR,
            axis="y",
        )

        if not rows:
            return None

        columns = [
            column
            for column in columns
            if len(column)
            >= max(
                1,
                int(
                    len(anchors)
                    * self.MIN_COLUMN_SHARE
                ),
            )
        ]

        if not columns:
            return None

        column_centers = [
            statistics.mean(
                value["x"]
                for value in column
            )
            for column in columns
        ]

        column_centers.sort()

        row_centers = [
            statistics.mean(
                value["y"]
                for value in row
            )
            for row in rows
        ]

        row_centers.sort()

        row_pitch = self._estimate_row_pitch(
            row_centers,
            median_height,
        )

        if row_pitch <= 0:

            row_pitch = (
                median_height
                * self.FALLBACK_ROW_PITCH_FACTOR
            )

        card_width = self._estimate_card_width(
            column_centers,
            items,
        )

        card_height = (
            row_pitch
            * self.CARD_HEIGHT_RATIO
        )

        # ------------------------------------------------------
        # NEW:
        #
        # The serial anchors are normally positioned near the
        # LEFT side of each voter card.
        #
        # Therefore we must not treat the serial position as
        # the physical center of the card.
        #
        # Build horizontal boundaries between neighboring
        # serial columns instead.
        # ------------------------------------------------------

        column_boundaries = (
            self._build_column_boundaries(
                column_centers,
                items,
            )
        )

        row_boundaries = (
            self._build_row_boundaries(
                row_centers,
                row_pitch,
            )
        )

        return {
            "columns": column_centers,
            "rows": row_centers,
            "row_pitch": row_pitch,
            "card_width": card_width,
            "card_height": card_height,
            "column_boundaries": column_boundaries,
            "row_boundaries": row_boundaries,
            "median_anchor_height": median_height,
        }

    # ==========================================================
    # COLUMN BOUNDARIES
    # ==========================================================

    def _build_column_boundaries(
        self,
        column_centers: list[float],
        items: list[dict],
    ) -> list[tuple[float, float]]:

        if not column_centers:
            return []

        centers = sorted(column_centers)

        if len(centers) == 1:

            widths = [
                item.get("w", 0)
                for item in items
                if item.get("w", 0) > 0
            ]

            if widths:
                estimated_width = (
                    statistics.median(widths) * 12
                )
            else:
                estimated_width = 500.0

            center = centers[0]

            return [
                (
                    center - estimated_width * 0.35,
                    center + estimated_width * 0.65,
                )
            ]

        # ------------------------------------------------------
        # Estimate horizontal distance between serial anchors.
        #
        # The serial number is NOT the center of the voter card.
        # In this voter-list layout it is positioned toward the
        # right side of the card.
        # ------------------------------------------------------

        spacings = [
            centers[index + 1] - centers[index]
            for index in range(len(centers) - 1)
            if centers[index + 1] > centers[index]
        ]

        if not spacings:
            return []

        column_pitch = statistics.median(spacings)

        # ------------------------------------------------------
        # Estimate the physical card position.
        #
        # Observed layout:
        #
        #     card left
        #          |
        #          |------ serial
        #          |
        #
        # The serial anchor is approximately 1/3 of a card
        # width from the physical left edge.
        #
        # This is derived from the repeated OCR geometry rather
        # than using page-specific x coordinates.
        # ------------------------------------------------------

        serial_left_offset = column_pitch * 0.33

        # The remaining part of the card is to the right
        # of the serial anchor.
        serial_right_offset = column_pitch - serial_left_offset

        boundaries = []

        for index, center in enumerate(centers):

            left = (
                center
                - serial_left_offset
            )

            right = (
                center
                + serial_right_offset
            )

            boundaries.append(
                (
                    left,
                    right,
                )
            )

        # ------------------------------------------------------
        # Make neighboring boundaries meet exactly.
        #
        # This prevents gaps/overlaps caused by small OCR
        # variations in serial positions.
        # ------------------------------------------------------

        for index in range(
            len(boundaries) - 1
        ):

            left_boundary = (
                boundaries[index][0]
            )

            right_boundary = (
                boundaries[index + 1][1]
            )

            shared_boundary = (
                boundaries[index][1]
                + boundaries[index + 1][0]
            ) / 2

            boundaries[index] = (
                left_boundary,
                shared_boundary,
            )

            boundaries[index + 1] = (
                shared_boundary,
                right_boundary,
            )

        return boundaries
    # ==========================================================
    # ROW BOUNDARIES
    # ==========================================================

    def _build_row_boundaries(
        self,
        row_centers: list[float],
        row_pitch: float,
    ) -> list[tuple[float, float]]:

        if not row_centers:
            return []

        if len(row_centers) == 1:

            center = row_centers[0]

            half_height = (
                row_pitch * 0.5
            )

            return [
                (
                    center - half_height,
                    center + half_height,
                )
            ]

        internal_boundaries = []

        for index in range(
            len(row_centers) - 1
        ):

            top = row_centers[index]
            bottom = row_centers[index + 1]

            internal_boundaries.append(
                (top + bottom) / 2
            )

        first_spacing = (
            row_centers[1]
            - row_centers[0]
        )

        last_spacing = (
            row_centers[-1]
            - row_centers[-2]
        )

        top_edge = (
            row_centers[0]
            - first_spacing * 0.50
        )

        bottom_edge = (
            row_centers[-1]
            + last_spacing * 0.50
        )

        boundaries = []

        boundaries.append(
            (
                top_edge,
                internal_boundaries[0],
            )
        )

        for index in range(
            len(internal_boundaries) - 1
        ):

            boundaries.append(
                (
                    internal_boundaries[index],
                    internal_boundaries[index + 1],
                )
            )

        boundaries.append(
            (
                internal_boundaries[-1],
                bottom_edge,
            )
        )

        return boundaries

    # ==========================================================
    # CLUSTER VALUES
    # ==========================================================

    def _cluster_values(
        self,
        values: list[float],
        tolerance: float,
        axis: str = "x",
    ) -> list[list[dict]]:

        if not values:
            return []

        if axis not in {"x", "y"}:
            raise ValueError(
                f"Unsupported clustering axis: {axis}"
            )

        sorted_values = sorted(
            values
        )

        clusters = []

        current = [
            {
                axis: sorted_values[0],
            }
        ]

        for value in sorted_values[1:]:

            center = statistics.mean(
                item[axis]
                for item in current
            )

            if abs(
                value - center
            ) <= tolerance:

                current.append(
                    {
                        axis: value
                    }
                )

            else:

                clusters.append(
                    current
                )

                current = [
                    {
                        axis: value
                    }
                ]

        clusters.append(
            current
        )

        return clusters

    # ==========================================================
    # ROW PITCH
    # ==========================================================

    def _estimate_row_pitch(
        self,
        row_centers: list[float],
        median_height: float,
    ) -> float:

        if len(row_centers) < 2:

            return (
                median_height
                * self.FALLBACK_ROW_PITCH_FACTOR
            )

        differences = [
            row_centers[index + 1]
            - row_centers[index]
            for index in range(
                len(row_centers) - 1
            )
            if row_centers[index + 1]
            > row_centers[index]
        ]

        if not differences:

            return (
                median_height
                * self.FALLBACK_ROW_PITCH_FACTOR
            )

        median_difference = statistics.median(
            differences
        )

        valid = [
            difference
            for difference in differences
            if difference
            >= (
                median_difference
                * self.MIN_ROW_SPACING_RATIO
            )
        ]

        if not valid:
            return median_difference

        return statistics.median(
            valid
        )

    # ==========================================================
    # CARD WIDTH
    # ==========================================================

    def _estimate_card_width(
        self,
        column_centers: list[float],
        items: list[dict],
    ) -> float:

        if len(column_centers) >= 2:

            differences = [
                column_centers[index + 1]
                - column_centers[index]
                for index in range(
                    len(column_centers) - 1
                )
            ]

            if differences:

                return statistics.median(
                    differences
                )

        widths = [
            item["w"]
            for item in items
            if item["w"] > 0
        ]

        if widths:

            return (
                statistics.median(
                    widths
                )
                * 10
            )

        return 500.0

    # ==========================================================
    # ASSIGN OCR ITEMS TO CELLS
    # ==========================================================

    def _assign_items_to_cells(
        self,
        items: list[dict],
        grid: dict,
    ) -> dict:

        cells = {}

        columns = grid["columns"]
        rows = grid["rows"]

        column_boundaries = grid.get(
            "column_boundaries",
            [],
        )

        row_boundaries = grid.get(
            "row_boundaries",
            [],
        )

        if not columns or not rows:
            return cells

        # ------------------------------------------------------
        # FALLBACK:
        #
        # If boundaries are unavailable, use the old method.
        # ------------------------------------------------------

        use_boundaries = (
            len(column_boundaries)
            == len(columns)
            and len(row_boundaries)
            == len(rows)
        )

        for item in items:

            x = item["x"]
            y = item["y"]

            column_index = None
            row_index = None

            # ==================================================
            # NEW BOUNDARY-BASED ASSIGNMENT
            # ==================================================

            if use_boundaries:

                for index, (
                    left,
                    right,
                ) in enumerate(
                    column_boundaries
                ):

                    if (
                        x >= left
                        and x < right
                    ):

                        column_index = index
                        break

                for index, (
                    top,
                    bottom,
                ) in enumerate(
                    row_boundaries
                ):

                    if (
                        y >= top
                        and y < bottom
                    ):

                        row_index = index
                        break

                if (
                    column_index is None
                    or row_index is None
                ):

                    continue

            # ==================================================
            # OLD FALLBACK
            # ==================================================

            else:

                column_index = min(
                    range(
                        len(columns)
                    ),
                    key=lambda index: abs(
                        x - columns[index]
                    ),
                )

                row_index = min(
                    range(
                        len(rows)
                    ),
                    key=lambda index: abs(
                        y - rows[index]
                    ),
                )

            cell_key = (
                row_index,
                column_index,
            )

            cells.setdefault(
                cell_key,
                [],
            ).append(
                item
            )

        return cells

        # ==========================================================
    # CLEAN VOTER RECORD OCR TEXT
    # ==========================================================

    def _clean_voter_record_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return text

        lines = []

        for line in text.splitlines():

            cleaned = line.strip()

            if not cleaned:
                continue

            # --------------------------------------------------
            # Remove OCR artifacts that are not voter fields.
            # --------------------------------------------------

            if cleaned.lower() == "photo is":
                continue

            if cleaned.lower() == "available":
                continue

            # --------------------------------------------------
            # Normalize common OCR variations of:
            #
            # வீட்டு எண்
            #
            # This is important because some OCR output was:
            #
            # வீட்டு எrஎன : 146
            # வீட்டு எரென : 146
            # --------------------------------------------------

            cleaned = re.sub(
                r"வீட்டு\s*எ[rR]\s*எ[னn]",
                "வீட்டு எண்",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"வீட்டு\s*எரென",
                "வீட்டு எண்",
                cleaned,
                flags=re.IGNORECASE,
            )

            # --------------------------------------------------
            # Normalize spacing around the house-number colon.
            # --------------------------------------------------

            cleaned = re.sub(
                r"(வீட்டு\s*எண்)\s*[:：]\s*",
                r"\1 : ",
                cleaned,
            )

            lines.append(cleaned)

        return "\n".join(lines).strip()

    # ==========================================================
    # FORMAT VOTER CARDS
    # =========================================================

    def _format_cards(
        self,
        cells: dict,
        grid: dict,
    ) -> list[dict]:

        records = []

        for cell_key in sorted(
            cells.keys()
        ):

            row_index, column_index = (
                cell_key
            )

            cell_items = cells[
                cell_key
            ]

            ordered_items = self._reading_order(
                cell_items,
                cell_items,
            )

            if not ordered_items:
                continue

            serial = None
            serial_item = None

            for item in ordered_items:

                if "serial" in item:

                    serial = item[
                        "serial"
                    ]

                    serial_item = item

                    break

            # --------------------------------------------------
            # A cell without a serial anchor is not a voter.
            # --------------------------------------------------

            if serial is None:
                continue

            lines = [
                item["text"]
                for item in ordered_items
                if item["text"].strip()
            ]

            if not lines:
                continue

            text = "\n".join(
                lines
            ).strip()

            # --------------------------------------------------
            # Clean only known OCR artifacts.
            #
            # This does NOT change grouping or OCR detection.
            # --------------------------------------------------

            text = self._clean_voter_record_text(
                text
            )

            record = {
                "serial_number": serial,
                "text": text,
                "page_number": None,
            }

            # --------------------------------------------------
            # EPIC EXTRACTION
            # --------------------------------------------------

            epic_numbers = []

            for item in ordered_items:

                item_text = item.get(
                    "text",
                    "",
                )

                if not item_text:
                    continue

                matches = (
                    self.EPIC_PATTERN.findall(
                        item_text
                    )
                )

                for match in matches:

                    epic = match.upper()

                    if epic not in epic_numbers:

                        epic_numbers.append(
                            epic
                        )

            if epic_numbers:

                record[
                    "epic_numbers"
                ] = epic_numbers

            records.append(
                record
            )

            self._log_card(
                record
            )

            # --------------------------------------------------
            # TEMPORARY EPIC COORDINATE DEBUG
            #
            # This does NOT change grouping.
            #
            # It only tells us exactly where the serial and
            # EPIC OCR boxes are located inside the assigned
            # voter cell.
            #
            # We especially need cards that currently contain
            # two EPICs.
            # --------------------------------------------------

            if len(epic_numbers) >= 2:

                self._log()

                self._log(
                    "---------- EPIC COORDINATE DEBUG ----------"
                )

                self._log(
                    f"CELL row={row_index} "
                    f"column={column_index} "
                    f"SERIAL={serial}"
                )

                if serial_item is not None:

                    self._log(
                        f"SERIAL BOX: "
                        f"text={serial_item['text']!r} "
                        f"x1={serial_item['x1']:.1f} "
                        f"x2={serial_item['x2']:.1f} "
                        f"y1={serial_item['y1']:.1f} "
                        f"y2={serial_item['y2']:.1f} "
                        f"center=({serial_item['x']:.1f},"
                        f"{serial_item['y']:.1f})"
                    )

                self._log(
                    "CELL BOUNDARY: "
                    f"x={grid['column_boundaries'][column_index][0]:.1f}"
                    f" -> "
                    f"{grid['column_boundaries'][column_index][1]:.1f} "
                    f"y={grid['row_boundaries'][row_index][0]:.1f}"
                    f" -> "
                    f"{grid['row_boundaries'][row_index][1]:.1f}"
                )

                for item in ordered_items:

                    item_text = item.get(
                        "text",
                        "",
                    )

                    if not item_text:
                        continue

                    item_epics = (
                        self.EPIC_PATTERN.findall(
                            item_text
                        )
                    )

                    if not item_epics:
                        continue

                    self._log(
                        f"EPIC BOX: "
                        f"text={item_text!r} "
                        f"x1={item['x1']:.1f} "
                        f"x2={item['x2']:.1f} "
                        f"y1={item['y1']:.1f} "
                        f"y2={item['y2']:.1f} "
                        f"center=({item['x']:.1f},"
                        f"{item['y']:.1f})"
                    )

                self._log(
                    "--------------------------------------------"
                )

        return records

    # ==========================================================
    # READING ORDER
    # ==========================================================

    def _reading_order(
        self,
        items: list[dict],
        reference_items: list[dict],
    ) -> list[dict]:

        if not items:
            return []

        heights = [
            item["h"]
            for item in reference_items
            if item.get("h", 0) > 0
        ]

        if heights:

            median_height = statistics.median(
                heights
            )

        else:

            median_height = 10.0

        line_tolerance = (
            median_height
            * self.LINE_TOLERANCE_FACTOR
        )

        sorted_items = sorted(
            items,
            key=lambda item: (
                item["y"],
                item["x"],
            ),
        )

        lines = []

        for item in sorted_items:

            assigned = False

            for line in lines:

                line_y = statistics.mean(
                    value["y"]
                    for value in line
                )

                if abs(
                    item["y"]
                    - line_y
                ) <= line_tolerance:

                    line.append(
                        item
                    )

                    assigned = True

                    break

            if not assigned:

                lines.append(
                    [item]
                )

        ordered = []

        for line in sorted(
            lines,
            key=lambda line: statistics.mean(
                value["y"]
                for value in line
            ),
        ):

            ordered.extend(
                sorted(
                    line,
                    key=lambda item: item["x"],
                )
            )

        return ordered

    # ==========================================================
    # DEBUG GRID
    # ==========================================================

    def _log_grid(
        self,
        grid: dict,
    ):

        self._log()

        self._log(
            "---------- GRID SUMMARY ----------"
        )

        self._log(
            f"Columns: {len(grid['columns'])}"
        )

        self._log(
            f"Rows: {len(grid['rows'])}"
        )

        self._log(
            f"Row pitch: "
            f"{grid['row_pitch']:.2f}"
        )

        self._log(
            f"Card width: "
            f"{grid['card_width']:.2f}"
        )

        self._log(
            f"Card height: "
            f"{grid['card_height']:.2f}"
        )

        self._log(
            "Column boundaries:"
        )

        for index, boundary in enumerate(
            grid.get(
                "column_boundaries",
                [],
            )
        ):

            self._log(
                f"  COLUMN {index}: "
                f"{boundary[0]:.1f} -> "
                f"{boundary[1]:.1f}"
            )

        self._log(
            "----------------------------------"
        )

    # ==========================================================
    # DEBUG CARD
    # ==========================================================

    def _log_card(
        self,
        record: dict,
    ):

        if not self.debug_grouping:
            return

        serial = record.get(
            "serial_number"
        )

        epic = record.get(
            "epic_numbers",
            [],
        )

        self._log(
            f"CARD "
            f"SERIAL={serial} "
            f"EPIC={epic}"
        )

    # ==========================================================
    # DEBUG SUMMARY
    # ==========================================================

    def _log_summary(
        self,
        records: list[dict],
    ):

        self._log()

        self._log(
            "---------- OCR SUMMARY ----------"
        )

        self._log(
            f"Records detected: "
            f"{len(records)}"
        )

        if records:

            serials = [
                record.get(
                    "serial_number"
                )
                for record in records
                if record.get(
                    "serial_number"
                ) is not None
            ]

            if serials:

                self._log(
                    f"Serial range: "
                    f"{min(serials)} - "
                    f"{max(serials)}"
                )

        self._log(
            "---------------------------------"
        )

    # ==========================================================
    # LOGGING
    # ==========================================================

    def _log(
        self,
        message: str = "",
    ):

        if self.debug_grouping:

            print(
                message,
                flush=True,
            )


ocr_service = OCRService()