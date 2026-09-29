import math
import os
import re
import statistics
from typing import Any, Optional

from paddleocr import PaddleOCR


class OCRService:

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    # A serial number is an OCR box that contains ONLY 1-4 digits.
    SERIAL_PATTERN = re.compile(r"^\s*(\d{1,4})\s*$")

    # Used ONLY to print debug information. It never decides grouping.
    EPIC_PATTERN = re.compile(r"[A-Z]{3}\d{7}", re.IGNORECASE)

    # ------------------------------------------------------------------
    # Tuning constants
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, debug_grouping: Optional[bool] = None):

        self._ocr_instances: dict[str, PaddleOCR] = {}

        if debug_grouping is None:
            debug_grouping = (
                os.getenv("OCR_DEBUG_GROUPING", "1").strip() != "0"
            )

        self.debug_grouping = debug_grouping

    def _get_ocr(self, language: str = "en") -> PaddleOCR:

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

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def _log(self, message: str = "") -> None:

        if not self.debug_grouping:
            return

        try:
            print(message)

        except UnicodeEncodeError:
            print(
                str(message)
                .encode("ascii", "backslashreplace")
                .decode("ascii")
            )

    # ------------------------------------------------------------------
    # PaddleOCR result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_result_data(data: dict) -> dict:

        if (
            isinstance(data, dict)
            and "rec_texts" not in data
            and isinstance(data.get("res"), dict)
        ):
            return data["res"]

        return data

    def _get_result_data(self, result: Any) -> dict:

        if result is None:
            return {}

        if hasattr(result, "json"):

            try:
                data = result.json

                if callable(data):
                    data = data()

                if isinstance(data, dict):
                    return self._unwrap_result_data(data)

                if isinstance(data, list) and data:

                    if isinstance(data[0], dict):
                        return self._unwrap_result_data(data[0])

            except Exception:
                pass

        if isinstance(result, dict):
            return self._unwrap_result_data(result)

        return {}

    # ------------------------------------------------------------------
    # Plain text extraction
    # ------------------------------------------------------------------

    def extract_text(
        self,
        image_path: str,
        language: str = "en",
    ) -> str:

        ocr = self._get_ocr(language)

        results = ocr.predict(image_path)

        lines = []

        for result in results:

            data = self._get_result_data(result)

            texts = data.get("rec_texts", [])

            if texts is None or len(texts) == 0:
                continue

            for text in texts:

                text = str(text).strip()

                if text:
                    lines.append(text)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Voter record extraction
    # ------------------------------------------------------------------

    def extract_voter_records(
        self,
        image_path: str,
        language: str = "en",
        include_page_text: bool = False,
    ) -> list[str]:
        """
        Return one text block per voter card.

        Reading order:
            page text/header
            voter cards row-by-row, left-to-right

        When include_page_text=False:
            Only voter records are returned.

        When include_page_text=True:
            Page-level text such as:
                - constituency
                - part number
                - section name
                - publication date
                - page number
                - footer/header

            is returned as the FIRST block.

        This is important because page metadata must reach the
        document chunking/retrieval pipeline.
        """

        ocr = self._get_ocr(language)

        results = ocr.predict(image_path)

        records: list[str] = []

        # Each result is one page.
        for page_number, result in enumerate(results, start=1):

            data = self._get_result_data(result)

            items = self._result_to_items(data)

            if not items:
                continue

            page_records = self._build_voter_records(
                items=items,
                include_page_text=include_page_text,
                page_number=page_number,
            )

            records.extend(page_records)

        return records

    # ------------------------------------------------------------------
    # OCR result -> list of boxes
    # ------------------------------------------------------------------

    def _result_to_items(
        self,
        data: dict,
    ) -> list[dict]:
        """
        Convert PaddleOCR output into:

            text
            x1
            y1
            x2
            y2
            x
            y
            w
            h
            id
        """

        rec_texts = data.get("rec_texts")
        rec_boxes = data.get("rec_boxes")
        rec_polys = data.get("rec_polys")

        if rec_texts is None or len(rec_texts) == 0:
            return []

        items: list[dict] = []

        for index, text in enumerate(rec_texts):

            text = str(text).strip()

            if not text:
                continue

            box = self._get_box(
                rec_boxes,
                rec_polys,
                index,
            )

            if box is None:
                continue

            x1, y1, x2, y2 = box

            items.append(
                {
                    "id": len(items),
                    "text": text,
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

    @staticmethod
    def _get_box(
        rec_boxes,
        rec_polys,
        index: int,
    ):
        """
        Return:

            (x1, y1, x2, y2)

        Uses rec_boxes first and rec_polys as fallback.
        """

        try:

            if (
                rec_boxes is not None
                and index < len(rec_boxes)
            ):

                box = rec_boxes[index]

                xs = [
                    float(box[0]),
                    float(box[2]),
                ]

                ys = [
                    float(box[1]),
                    float(box[3]),
                ]

                return (
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                )

        except Exception:
            pass

        try:

            if (
                rec_polys is not None
                and index < len(rec_polys)
            ):

                points = rec_polys[index]

                xs = [
                    float(point[0])
                    for point in points
                ]

                ys = [
                    float(point[1])
                    for point in points
                ]

                return (
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                )

        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Small generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_by_value(
        entries: list[dict],
        key: str,
        tolerance: float,
    ):

        clusters: list[list[dict]] = []

        for entry in sorted(
            entries,
            key=lambda e: e[key],
        ):

            if (
                clusters
                and (
                    entry[key]
                    - clusters[-1][-1][key]
                )
                <= tolerance
            ):

                clusters[-1].append(entry)

            else:
                clusters.append([entry])

        return clusters

    @staticmethod
    def _median(values) -> float:

        return float(
            statistics.median(
                list(values)
            )
        )

    # ------------------------------------------------------------------
    # Main grouping pipeline
    # ------------------------------------------------------------------

    def _build_voter_records(
        self,
        items: list[dict],
        include_page_text: bool = False,
        page_number: int = 1,
    ) -> list[str]:

        # Make sure every item has required fields.
        for index, item in enumerate(items):

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

        self._log("=" * 72)

        self._log(
            f"VOTER GRID DEBUG - page {page_number} "
            f"({len(items)} OCR boxes)"
        )

        self._log("=" * 72)

        # ==============================================================
        # STEP 1
        # Find serial-number anchors
        # ==============================================================

        anchors = self._find_serial_anchors(items)

        if not anchors:

            self._log(
                "No serial-number boxes found -> "
                "no voter cards."
            )

            # If this page contains no voter cards, preserve
            # the complete OCR text as page-level text.
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
                    return [page_text]

            return []

        # ==============================================================
        # STEP 2 + 3
        # Detect grid
        # ==============================================================

        grid = self._detect_grid(
            anchors,
            items,
        )

        # ==============================================================
        # STEP 4 + 5
        # Build card rectangles and assign OCR boxes
        # ==============================================================

        cells, page_text = (
            self._assign_items_to_cells(
                items,
                grid,
            )
        )

        # ==============================================================
        # STEP 6
        # Format voter cards
        # ==============================================================

        records = self._format_cards(
            cells,
            grid,
            items,
        )

        # ==============================================================
        # PAGE-LEVEL TEXT
        # ==============================================================
        #
        # IMPORTANT:
        #
        # Page text is inserted BEFORE voter records.
        #
        # Example:
        #
        #     constituency header
        #     part number
        #     section name
        #     publication date
        #
        #     voter 1
        #     voter 2
        #     voter 3
        #
        # This allows ChunkingService to create a dedicated
        # document-level chunk.
        # ==============================================================

        if include_page_text and page_text:

            ordered_page_text = self._reading_order(
                page_text,
                items,
            )

            page_text_block = "\n".join(
                item["text"]
                for item in ordered_page_text
            ).strip()

            if page_text_block:

                records.insert(
                    0,
                    page_text_block,
                )

        # Debug information.
        self._log_summary(
            cells,
            page_text,
        )

        return records

    # ------------------------------------------------------------------
    # STEP 1 - serial anchors
    # ------------------------------------------------------------------

    def _find_serial_anchors(
        self,
        items: list[dict],
    ) -> list[dict]:

        anchors = []

        for item in items:

            match = self.SERIAL_PATTERN.match(
                item["text"]
            )

            if not match:
                continue

            serial = int(
                match.group(1)
            )

            if 1 <= serial <= 9999:

                item["serial"] = serial

                anchors.append(item)

        return anchors

    # ------------------------------------------------------------------
    # STEP 2 + 3 - grid detection
    # ------------------------------------------------------------------

    def _detect_grid(
        self,
        anchors: list[dict],
        items: list[dict],
    ) -> dict:

        median_h = (
            self._median(
                a["h"]
                for a in anchors
            )
            or 1.0
        )

        # --------------------------------------------------------------
        # Columns
        # --------------------------------------------------------------

        col_clusters = (
            self._cluster_by_value(
                anchors,
                "x",
                self.COLUMN_X_TOLERANCE_FACTOR
                * median_h,
            )
        )

        largest = max(
            len(cluster)
            for cluster in col_clusters
        )

        min_members = max(
            1,
            math.ceil(
                self.MIN_COLUMN_SHARE
                * largest
            ),
        )

        col_clusters = [
            cluster
            for cluster in col_clusters
            if len(cluster) >= min_members
        ]

        col_clusters = (
            self._keep_lattice_columns(
                col_clusters
            )
        )

        col_clusters.sort(
            key=lambda c: self._median(
                a["x"]
                for a in c
            )
        )

        columns = [
            {
                "x": self._median(
                    a["x"]
                    for a in cluster
                ),
                "anchors": cluster,
            }
            for cluster in col_clusters
        ]

        grid_anchors = [
            a
            for cluster in col_clusters
            for a in cluster
        ]

        # --------------------------------------------------------------
        # Rows
        # --------------------------------------------------------------

        row_clusters = (
            self._cluster_by_value(
                grid_anchors,
                "y",
                self.ROW_Y_TOLERANCE_FACTOR
                * median_h,
            )
        )

        rows = [
            {
                "y": self._median(
                    a["y"]
                    for a in cluster
                ),
                "anchors": cluster,
            }
            for cluster in row_clusters
        ]

        rows.sort(
            key=lambda r: r["y"]
        )

        fallback_pitch_y = (
            self.FALLBACK_ROW_PITCH_FACTOR
            * median_h
        )

        if len(rows) >= 2:

            pitch_y = self._median(
                rows[i + 1]["y"]
                - rows[i]["y"]
                for i in range(
                    len(rows) - 1
                )
            )

            kept_rows = [rows[0]]

            for row in rows[1:]:

                if (
                    row["y"]
                    - kept_rows[-1]["y"]
                    >= (
                        self.MIN_ROW_SPACING_RATIO
                        * pitch_y
                    )
                ):
                    kept_rows.append(row)

            rows = kept_rows

            if len(rows) >= 2:

                pitch_y = self._median(
                    rows[i + 1]["y"]
                    - rows[i]["y"]
                    for i in range(
                        len(rows) - 1
                    )
                )

            else:
                pitch_y = fallback_pitch_y

        else:
            pitch_y = fallback_pitch_y

        # --------------------------------------------------------------
        # Cell ownership
        # --------------------------------------------------------------

        col_index_of = {
            a["id"]: c_idx
            for c_idx, column in enumerate(columns)
            for a in column["anchors"]
        }

        row_index_of = {
            a["id"]: r_idx
            for r_idx, row in enumerate(rows)
            for a in row["anchors"]
        }

        anchor_by_cell: dict[
            tuple,
            dict,
        ] = {}

        rejected_anchors: list[dict] = []

        for anchor in sorted(
            grid_anchors,
            key=lambda a: (
                a["y"],
                a["x"],
            ),
        ):

            if anchor["id"] not in row_index_of:

                rejected_anchors.append(
                    anchor
                )

                continue

            cell = (
                row_index_of[
                    anchor["id"]
                ],
                col_index_of[
                    anchor["id"]
                ],
            )

            if cell in anchor_by_cell:

                rejected_anchors.append(
                    anchor
                )

            else:

                anchor_by_cell[cell] = (
                    anchor
                )

        used_ids = {
            a["id"]
            for a in anchor_by_cell.values()
        }

        rejected_anchors.extend(
            a
            for a in anchors
            if (
                a["id"] not in used_ids
                and a not in rejected_anchors
            )
        )

        # --------------------------------------------------------------
        # Horizontal geometry
        # --------------------------------------------------------------

        pitch_x = None

        if len(columns) >= 2:

            pitch_x = self._median(
                columns[i + 1]["x"]
                - columns[i]["x"]
                for i in range(
                    len(columns) - 1
                )
            )

        page_width = (
            max(
                item["x2"]
                for item in items
            )
            - min(
                item["x1"]
                for item in items
            )
        )

        reference_width = (
            pitch_x
            if pitch_x
            else page_width
        )

        banner_limit = (
            self.BANNER_WIDTH_RATIO
            * pitch_x
            if pitch_x
            else None
        )

        left_offset, edge_count = (
            self._estimate_left_edge_offset(
                items=items,
                columns=columns,
                anchor_ids=used_ids,
                reference_width=reference_width,
                banner_limit=banner_limit,
            )
        )

        if left_offset is None:

            left_offset = (
                -self.LEFT_EDGE_FALLBACK_RATIO
                * reference_width
            )

            self._log(
                "WARNING: could not measure "
                "the card left edge; "
                "using a default guess."
            )

        pad = (
            self.COLUMN_PAD_RATIO
            * reference_width
        )

        for column in columns:

            column["left"] = (
                column["x"]
                + left_offset
            )

            column["bound_left"] = (
                column["left"]
                - pad
            )

        right_limit = (
            columns[-1]["left"]
            + pitch_x
            if pitch_x
            else float("inf")
        )

        # --------------------------------------------------------------
        # Vertical geometry
        # --------------------------------------------------------------

        top_tolerance = (
            self.TOP_TOLERANCE_FACTOR
            * median_h
        )

        for index, row in enumerate(rows):

            row["top"] = (
                row["y"]
                - top_tolerance
            )

            bottom = (
                row["y"]
                + self.CARD_HEIGHT_RATIO
                * pitch_y
            )

            if index + 1 < len(rows):

                bottom = min(
                    bottom,
                    rows[index + 1]["y"]
                    - top_tolerance,
                )

            row["bottom"] = bottom

        grid = {
            "columns": columns,
            "rows": rows,
            "pitch_x": pitch_x,
            "pitch_y": pitch_y,
            "median_h": median_h,
            "left_offset": left_offset,
            "right_limit": right_limit,
            "banner_limit": banner_limit,
            "anchor_by_cell": anchor_by_cell,
        }

        self._log_grid(
            grid,
            len(anchors),
            len(anchor_by_cell),
            edge_count,
        )

        return grid

    def _keep_lattice_columns(
        self,
        col_clusters: list[list[dict]],
    ):

        largest = max(
            len(c)
            for c in col_clusters
        )

        strong = [
            c
            for c in col_clusters
            if len(c) >= 0.5 * largest
        ]

        if len(strong) < 2:
            return col_clusters

        strong_x = sorted(
            self._median(
                a["x"]
                for a in c
            )
            for c in strong
        )

        pitch = self._median(
            strong_x[i + 1]
            - strong_x[i]
            for i in range(
                len(strong_x) - 1
            )
        )

        if pitch <= 0:
            return col_clusters

        kept = []

        for cluster in col_clusters:

            cx = self._median(
                a["x"]
                for a in cluster
            )

            distance = abs(
                cx - strong_x[0]
            )

            whole = round(
                distance / pitch
            )

            if abs(
                distance
                - whole * pitch
            ) <= (
                self.COLUMN_LATTICE_TOLERANCE
                * pitch
            ):

                kept.append(cluster)

        return (
            kept
            or col_clusters
        )

    def _estimate_left_edge_offset(
        self,
        items: list[dict],
        columns: list[dict],
        anchor_ids: set,
        reference_width: float,
        banner_limit: Optional[float],
    ):

        search = (
            self.LEFT_EDGE_SEARCH_RATIO
            * reference_width
        )

        offsets = []

        for item in items:

            if item["id"] in anchor_ids:
                continue

            if (
                banner_limit is not None
                and item["w"] > banner_limit
            ):
                continue

            for column in columns:

                distance = (
                    item["x1"]
                    - column["x"]
                )

                if (
                    -search
                    <= distance
                    <= 0
                ):
                    offsets.append(
                        distance
                    )

        if not offsets:
            return None, 0

        offsets.sort()

        window = (
            self.LEFT_EDGE_CLUSTER_RATIO
            * reference_width
        )

        best_count = 0
        best_start = 0
        best_end = 0

        start = 0

        for end in range(
            len(offsets)
        ):

            while (
                offsets[end]
                - offsets[start]
                > window
            ):
                start += 1

            count = (
                end
                - start
                + 1
            )

            if count >= best_count:

                best_count = count
                best_start = start
                best_end = end

        return (
            self._median(
                offsets[
                    best_start:
                    best_end + 1
                ]
            ),
            best_count,
        )

    # ------------------------------------------------------------------
    # STEP 4 + 5 - card rectangles and assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_cell(
        item: dict,
        grid: dict,
    ):

        columns = grid["columns"]
        rows = grid["rows"]

        cx = item["x"]
        cy = item["y"]

        if (
            cx < columns[0]["bound_left"]
            or cx >= grid["right_limit"]
        ):
            return None

        column_index = 0

        for index, column in enumerate(
            columns
        ):

            if (
                cx
                >= column["bound_left"]
            ):
                column_index = index

        for index, row in enumerate(
            rows
        ):

            if (
                row["top"]
                <= cy
                < row["bottom"]
            ):
                return (
                    index,
                    column_index,
                )

        return None

    def _assign_items_to_cells(
        self,
        items: list[dict],
        grid: dict,
    ):

        anchor_by_cell = (
            grid["anchor_by_cell"]
        )

        banner_limit = (
            grid["banner_limit"]
        )

        anchor_ids = {
            a["id"]
            for a in anchor_by_cell.values()
        }

        cells = {
            cell: {
                "anchor": anchor,
                "items": [],
            }
            for cell, anchor
            in anchor_by_cell.items()
        }

        page_text: list[dict] = []

        for item in items:

            if item["id"] in anchor_ids:
                continue

            # Too wide for one card.
            # Treat as header/footer/banner.
            if (
                banner_limit is not None
                and item["w"] > banner_limit
            ):

                page_text.append(item)

                continue

            cell = self._locate_cell(
                item,
                grid,
            )

            if cell is None:

                page_text.append(item)

                continue

            cells.setdefault(
                cell,
                {
                    "anchor": None,
                    "items": [],
                },
            )["items"].append(item)

        # Cells without serial anchors are kept only when
        # they contain enough lines to look like real cards.
        for cell in list(cells.keys()):

            data = cells[cell]

            if (
                data["anchor"] is None
                and len(
                    data["items"]
                )
                < self.MIN_ORPHAN_LINES
            ):

                page_text.extend(
                    data["items"]
                )

                del cells[cell]

        return (
            cells,
            page_text,
        )

    # ------------------------------------------------------------------
    # STEP 6 - card text blocks
    # ------------------------------------------------------------------

    def _reading_order(
        self,
        cell_items: list[dict],
        all_items: list[dict],
    ):

        median_item_h = (
            self._median(
                i["h"]
                for i in all_items
            )
            or 1.0
        )

        line_tolerance = (
            self.LINE_TOLERANCE_FACTOR
            * median_item_h
        )

        lines: list[dict] = []

        for item in sorted(
            cell_items,
            key=lambda i: i["y"],
        ):

            if (
                lines
                and abs(
                    item["y"]
                    - lines[-1]["y"]
                )
                <= line_tolerance
            ):

                lines[-1]["items"].append(
                    item
                )

            else:

                lines.append(
                    {
                        "y": item["y"],
                        "items": [item],
                    }
                )

        ordered: list[dict] = []

        for line in lines:

            ordered.extend(
                sorted(
                    line["items"],
                    key=lambda i: i["x1"],
                )
            )

        return ordered

    def _format_cards(
        self,
        cells: dict,
        grid: dict,
        all_items: list[dict],
    ) -> list[str]:

        records: list[str] = []

        # Reading order:
        # row by row, left to right.
        for cell in sorted(
            cells.keys()
        ):

            data = cells[cell]

            ordered_items = (
                self._reading_order(
                    data["items"],
                    all_items,
                )
            )

            lines = []

            if data["anchor"] is not None:

                lines.append(
                    data["anchor"]["text"]
                )

            lines.extend(
                item["text"]
                for item in ordered_items
            )

            record = "\n".join(
                lines
            ).strip()

            if record:
                records.append(record)

            self._log_card(
                cell,
                data,
                ordered_items,
                grid,
            )

        return records

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def _log_grid(
        self,
        grid: dict,
        anchors_found: int,
        anchors_used: int,
        edge_count: int,
    ) -> None:

        if not self.debug_grouping:
            return

        columns = grid["columns"]
        rows = grid["rows"]

        self._log(
            f"Serial boxes found: "
            f"{anchors_found}   "
            f"used as grid anchors: "
            f"{anchors_used}"
        )

        self._log(
            "Columns (serial x): "
            + ", ".join(
                f"{c['x']:.1f}"
                for c in columns
            )
        )

        if grid["pitch_x"]:

            self._log(
                f"Column pitch: "
                f"{grid['pitch_x']:.1f}px"
            )

        self._log(
            "Rows (serial y):    "
            + ", ".join(
                f"{r['y']:.1f}"
                for r in rows
            )
        )

        self._log(
            f"Row pitch:    "
            f"{grid['pitch_y']:.1f}px"
        )

        self._log(
            f"Card left edge is "
            f"{-grid['left_offset']:.1f}px "
            f"LEFT of the serial number "
            f"({edge_count} text lines agree)"
        )

        self._log(
            "Card left edges (x): "
            + ", ".join(
                f"{c['left']:.1f}"
                for c in columns
            )
        )

        self._log()

    def _log_card(
        self,
        cell: tuple,
        data: dict,
        ordered_items: list[dict],
        grid: dict,
    ) -> None:

        if not self.debug_grouping:
            return

        row_index, column_index = cell

        columns = grid["columns"]

        row = grid["rows"][row_index]

        column = columns[column_index]

        if column_index + 1 < len(columns):

            x_end = columns[
                column_index + 1
            ]["left"]

        else:

            x_end = grid["right_limit"]

        anchor = data["anchor"]

        if anchor is not None:

            title = (
                f"SERIAL "
                f"{anchor['serial']}"
            )

        else:

            title = (
                "NO SERIAL "
                "(serial box not detected)"
            )

        self._log(
            f"{title}   "
            f"(row {row_index + 1}, "
            f"column {column_index + 1})"
        )

        if anchor is not None:

            self._log(
                f"  serial box: "
                f"x={anchor['x']:.1f} "
                f"y={anchor['y']:.1f}"
            )

        self._log(
            f"  card area : "
            f"x=[{column['left']:.1f}, "
            f"{x_end:.1f})  "
            f"y=[{row['top']:.1f}, "
            f"{row['bottom']:.1f})"
        )

        epics = [
            item
            for item in ordered_items
            if self.EPIC_PATTERN.search(
                item["text"]
            )
        ]

        if epics:

            for item in epics:

                self._log(
                    f"  EPIC: "
                    f"{item['text']}  "
                    f"(x={item['x']:.1f}, "
                    f"y={item['y']:.1f})"
                )

        else:

            self._log(
                "  EPIC: "
                "(none found inside this card)"
            )

        for item in ordered_items:

            self._log(
                f"    | {item['text']}"
            )

        self._log()

    def _log_summary(
        self,
        cells: dict,
        page_text: list[dict],
    ) -> None:

        if not self.debug_grouping:
            return

        self._log("-" * 72)

        self._log(
            f"Voter cards on this page: "
            f"{len(cells)}"
        )

        # Serial sequence check.
        previous = None

        for cell in sorted(
            cells.keys()
        ):

            anchor = (
                cells[cell]["anchor"]
            )

            if anchor is None:
                continue

            serial = anchor["serial"]

            if (
                previous is not None
                and serial != previous + 1
            ):

                self._log(
                    f"WARNING: serial sequence "
                    f"jumps {previous} -> {serial}"
                )

            previous = serial

        # Compact SERIAL -> EPIC table.
        self._log(
            "SERIAL -> EPIC table:"
        )

        for cell in sorted(
            cells.keys()
        ):

            data = cells[cell]

            serial = (
                data["anchor"]["serial"]
                if data["anchor"] is not None
                else "?"
            )

            found = [
                match.group(0).upper()
                for item in data["items"]
                for match in [
                    self.EPIC_PATTERN.search(
                        item["text"]
                    )
                ]
                if match
            ]

            self._log(
                f"  {serial} -> "
                f"{', '.join(found) if found else '-'}"
            )

        # Page-level text.
        if page_text:

            self._log(
                f"Page text kept OUT of "
                f"voter cards: "
                f"{len(page_text)} boxes"
            )

            for item in sorted(
                page_text,
                key=lambda i: (
                    i["y"],
                    i["x1"],
                ),
            ):

                self._log(
                    f"    x={item['x']:.0f} "
                    f"y={item['y']:.0f} "
                    f"| {item['text']}"
                )

        self._log("=" * 72)

        self._log()


# ----------------------------------------------------------------------
# Global instance
# ----------------------------------------------------------------------

ocr_service = OCRService()