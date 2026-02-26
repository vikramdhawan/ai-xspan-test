"""
PDF Parser for Agentic RAG Ingestion Pipeline
==============================================
Converts a PDF into structured chunks that preserve:
  - Section and subsection titles (via font heuristics)
  - Tables (via PyMuPDF's find_tables() + caption-triggered fallback)
  - Equations (kept in-line as readable text approximations)
  - Page numbers (stored in every chunk's metadata)
  - Figures (caption text extracted and stored)
  - Footnotes (emitted as separate chunks)

Output: a flat list of chunk dicts, one per logical unit of content.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import tiktoken

# ---------------------------------------------------------------------------
# Font & layout constants (calibrated from inspecting the target PDF)
# ---------------------------------------------------------------------------
SECTION_FONT_SUBSTRING = "Medi"          # NimbusRomNo9L-Medi: headings
MATH_FONT_PREFIXES = (                   # Computer Modern math fonts
    "CMMI", "CMR", "CMSY", "CMEX", "CMB", "CMBX", "CMMIB", "CMBSY",
)
SECTION_SIZE_MIN = 11.5                  # pt — level-1 section headings
SUBSECTION_SIZE_MIN = 9.2               # pt — level-2 subsection headings
SUBSECTION_SIZE_MAX = 11.4              # pt — upper bound for subsections

FOOTER_Y_THRESHOLD = 700.0              # blocks below this may be footnotes
PAGE_NUM_Y_THRESHOLD = 730.0            # blocks below this that are bare digits
COPYRIGHT_COLOR = 16711680              # 0xFF0000 — red copyright line, page 1
ARXIV_X_MAX = 50.0                      # left-margin arXiv watermark
AUTHOR_Y_MAX = 400.0                    # on page 1, author block is above this

MAX_FALSE_POS_TABLE_COLS = 10           # attention heatmap tables → skip
MAX_TOKENS = 400                         # max tokens per text chunk
OVERLAP_TOKENS = 50                      # tokens carried into next chunk as overlap

TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+[:\.]", re.IGNORECASE)
FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+[:\.]", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,2}$")
SECTION_NUM_RE = re.compile(r"^\d+(\.\d+)*$")   # bare "1" or "3.2.1" etc.

# The tiktoken encoder used for token counting (same as OpenAI embedding model)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _slugify(text: str) -> str:
    """Turn a section title into a safe identifier fragment."""
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text[:40].strip("_")


def _is_math_font(font_name: str) -> bool:
    return any(font_name.startswith(p) for p in MATH_FONT_PREFIXES)


def _bbox_overlap_ratio(a: tuple, b: tuple) -> float:
    """Return what fraction of bbox `a` is covered by bbox `b`."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    return intersection / area_a if area_a > 0 else 0.0


def _block_plain_text(block: dict) -> str:
    """Extract all text from a block as a single clean string."""
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            t = _clean_text(span["text"])
            if t:
                parts.append(t)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Parses a single PDF file into a list of structured chunk dicts.

    Design philosophy
    -----------------
    Rather than treating the PDF as a flat stream of text (which loses all
    structure), we walk PyMuPDF's *block* representation, which gives us:
      - font name and size per text span → lets us detect headings
      - bounding boxes → lets us identify tables, figures, footnotes
      - block type (0=text, 1=image) → lets us handle figures separately

    We build a list of chunks that are *semantically bounded*: each chunk
    stays within a section, stays under MAX_TOKENS, and carries metadata
    that tells the retrieval layer exactly where the content came from.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ):
        self.pdf_path = Path(pdf_path)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        # Running state (reset per document, but section labels persist
        # across pages because sections often span multiple pages)
        self._current_section = "Preamble"
        self._current_subsection = ""
        self._chunk_seq = 0          # global sequence counter for chunk IDs
        self._page_start = 1         # page where current accumulation began
        self._accumulator: list[str] = []
        self._accum_tokens = 0
        self._overlap_buffer: list[str] = []
        self._has_equation_in_accum = False
        self._in_references = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> list[dict]:
        """
        Open the PDF and return all chunks.

        Returns
        -------
        list[dict]
            Each dict is a chunk with text + rich metadata. See module
            docstring for the full schema.
        """
        doc = fitz.open(str(self.pdf_path))
        all_chunks: list[dict] = []
        last_page_num = 1

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-indexed for humans
            last_page_num = page_num
            chunks = self._process_page(page, page_num)
            all_chunks.extend(chunks)

        # Flush any remaining accumulated text after the last page
        final = self._flush(last_page_num)
        if final:
            all_chunks.append(final)

        doc.close()
        return all_chunks

    # ------------------------------------------------------------------
    # Per-page processing
    # ------------------------------------------------------------------

    def _process_page(self, page: fitz.Page, page_num: int) -> list[dict]:
        chunks: list[dict] = []

        # Step 1 ── detect and extract tables first.
        # We record their bboxes so we can skip those regions during text walk.
        table_chunks, table_bboxes = self._extract_tables(page, page_num)
        chunks.extend(table_chunks)

        # Step 2 ── get structured block list in reading order
        block_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = block_dict["blocks"]

        # Step 3 ── pre-scan to merge split headings like ("1", "Introduction")
        #           that appear as separate consecutive blocks on the same y line.
        heading_map = self._build_heading_map(blocks, page_num)
        # heading_map: {block_idx: merged_heading_text | None}
        #   - truthy string → first block of a merged heading group
        #   - None          → follow-on block (skip it)

        for block_idx, block in enumerate(blocks):

            # Skip image blocks (figures handled via captions in text)
            if block.get("type") == 1:
                continue

            bbox = tuple(block["bbox"])  # (x0, y0, x1, y1)

            # ── Skip: overlaps with a known table region (avoid double-ingestion)
            if any(_bbox_overlap_ratio(bbox, tb) > 0.5 for tb in table_bboxes):
                continue

            # ── Skip: arXiv watermark in left margin
            if bbox[0] < ARXIV_X_MAX and bbox[2] < ARXIV_X_MAX + 30:
                continue

            # ── Skip: page number marker at very bottom
            if bbox[1] > PAGE_NUM_Y_THRESHOLD:
                text = _block_plain_text(block)
                if PAGE_NUMBER_RE.match(text):
                    continue

            # ── Footnotes: small text below footer threshold
            if bbox[1] > FOOTER_Y_THRESHOLD:
                fn = self._try_make_footnote(block, page_num)
                if fn:
                    flushed = self._flush(page_num)
                    if flushed:
                        chunks.append(flushed)
                    chunks.append(fn)
                continue

            # ── Merged heading (first block of a group)
            if block_idx in heading_map:
                merged = heading_map[block_idx]
                if merged is None:
                    continue  # follow-on block already consumed
                if merged:   # non-empty string → this is the heading
                    flushed = self._flush(page_num)
                    if flushed:
                        chunks.append(flushed)
                    self._set_section(merged, page_num)
                    continue

            # ── Walk spans inside this block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = _clean_text(span["text"])
                    if not span_text:
                        continue

                    font = span.get("font", "")
                    size = span.get("size", 10.0)
                    color = span.get("color", 0)

                    # Skip: red copyright block on page 1
                    if color == COPYRIGHT_COLOR and page_num == 1:
                        continue

                    # Level-1 section heading (not already handled via merge)
                    if self._is_section_heading(font, size):
                        # Page 1, above author block boundary → treat as preamble
                        if page_num == 1 and bbox[1] < AUTHOR_Y_MAX:
                            self._append_to_accum(span_text, font, page_num)
                            continue
                        flushed = self._flush(page_num)
                        if flushed:
                            chunks.append(flushed)
                        self._set_section(span_text, page_num)
                        continue

                    # Level-2 subsection heading
                    if self._is_subsection_heading(font, size, page_num, bbox, span_text):
                        self._current_subsection = span_text
                        continue

                    # Figure caption → own chunk (skip trivially short captions)
                    if FIGURE_CAPTION_RE.match(span_text) and len(span_text) > 12:
                        flushed = self._flush(page_num)
                        if flushed:
                            chunks.append(flushed)
                        chunks.append(self._make_figure_chunk(span_text, page_num))
                        continue

                    # Default: body text (includes inline equations)
                    self._append_to_accum(span_text, font, page_num)

                    if self._accum_tokens >= self.max_tokens:
                        flushed = self._flush(page_num)
                        if flushed:
                            chunks.append(flushed)

        return chunks

    # ------------------------------------------------------------------
    # Section state helper
    # ------------------------------------------------------------------

    def _set_section(self, heading: str, page_num: int) -> None:
        self._current_section = heading
        self._current_subsection = ""
        self._page_start = page_num
        self._in_references = heading.strip().lower() == "references"

    # ------------------------------------------------------------------
    # Heading merge pre-pass
    # ------------------------------------------------------------------

    def _build_heading_map(
        self, blocks: list[dict], page_num: int
    ) -> dict[int, Optional[str]]:
        """
        Pre-scan blocks on this page to find *split headings* — cases where
        a section number ("1") and its title ("Introduction") appear as
        separate adjacent text blocks at the same y-coordinate.

        Returns
        -------
        dict mapping block_index → merged heading text (str) or None.
          - str  : first block of a merged group; value is the full heading
          - None : follow-on block in the group (caller should skip it)

        Blocks not in the dict are normal body blocks.
        """
        result: dict[int, Optional[str]] = {}

        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.get("type") != 0:
                i += 1
                continue

            spans = [
                s
                for line in block.get("lines", [])
                for s in line.get("spans", [])
            ]
            if not spans:
                i += 1
                continue

            first = spans[0]
            font = first.get("font", "")
            size = first.get("size", 10.0)
            by0 = block["bbox"][1]

            # Must be a heading-font span
            if SECTION_FONT_SUBSTRING not in font:
                i += 1
                continue
            if size < SUBSECTION_SIZE_MIN:
                i += 1
                continue

            # On page 1, author names share the same font — skip the author block
            if page_num == 1 and by0 < AUTHOR_Y_MAX:
                i += 1
                continue

            block_text = _block_plain_text(block)

            # Reject long blocks that are clearly body text (run-in bold labels,
            # e.g. "Encoder: The encoder is composed of a stack...")
            if len(block_text) > 120:
                i += 1
                continue

            # Look ahead: collect consecutive heading-font blocks at same y
            parts = [block_text]
            j = i + 1
            while j < len(blocks):
                nb = blocks[j]
                if nb.get("type") != 0:
                    break
                nb_spans = [
                    s for line in nb.get("lines", []) for s in line.get("spans", [])
                ]
                if not nb_spans:
                    break
                nf = nb_spans[0]
                nfont = nf.get("font", "")
                nsize = nf.get("size", 10.0)
                ny0 = nb["bbox"][1]

                if (
                    SECTION_FONT_SUBSTRING in nfont
                    and nsize >= SUBSECTION_SIZE_MIN
                    and abs(ny0 - by0) <= 5
                ):
                    parts.append(_block_plain_text(nb))
                    j += 1
                else:
                    break

            if len(parts) > 1:
                # Register the merged group
                merged = " ".join(parts)
                result[i] = merged
                for k in range(i + 1, j):
                    result[k] = None   # follow-on blocks → skip
                i = j
            else:
                # Single-block heading — still register so the main loop
                # can route it correctly (avoids re-detection in span walk).
                result[i] = block_text
                i += 1

        return result

    # ------------------------------------------------------------------
    # Accumulator helpers
    # ------------------------------------------------------------------

    def _append_to_accum(self, text: str, font: str, page_num: int) -> None:
        if not self._accumulator:
            self._page_start = page_num
        self._accumulator.append(text)
        self._accum_tokens += _count_tokens(text)
        if _is_math_font(font):
            self._has_equation_in_accum = True

    def _flush(self, page_end: int) -> Optional[dict]:
        """
        Emit the current accumulator as a chunk dict and reset state.

        Overlap handling
        ----------------
        The last OVERLAP_TOKENS tokens are carried into the next chunk so
        that context is not abruptly cut at chunk boundaries. This means the
        same sentence fragment may appear at the end of one chunk and the
        start of the next — intentional for retrieval recall.
        """
        if not self._accumulator:
            return None

        full_text = _clean_text(" ".join(self._accumulator))
        if not full_text:
            self._accumulator = list(self._overlap_buffer)
            self._accum_tokens = _count_tokens(" ".join(self._overlap_buffer))
            self._has_equation_in_accum = False
            return None

        self._chunk_seq += 1
        chunk = {
            "chunk_id": self._make_chunk_id(self._current_section, self._page_start),
            "text": full_text,
            "type": "text",
            "section": self._current_section,
            "subsection": self._current_subsection,
            "page_number": self._page_start,
            "page_end": page_end,
            "source": self.pdf_path.name,
            "paper_title": self.pdf_path.stem,
            "has_equation": self._has_equation_in_accum,
            "has_image": False,
            "is_reference": self._in_references,
            "table_caption": "",
            "table_page": -1,
            "token_count": _count_tokens(full_text),
        }

        # Build overlap tail from the end of the accumulator
        tail: list[str] = []
        tail_tokens = 0
        for item in reversed(self._accumulator):
            t = _count_tokens(item)
            if tail_tokens + t > self.overlap_tokens:
                break
            tail.insert(0, item)
            tail_tokens += t

        self._overlap_buffer = tail
        self._accumulator = list(tail)
        self._accum_tokens = tail_tokens
        self._has_equation_in_accum = False
        return chunk

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _is_section_heading(self, font: str, size: float) -> bool:
        return SECTION_FONT_SUBSTRING in font and size >= SECTION_SIZE_MIN

    def _is_subsection_heading(
        self, font: str, size: float, page_num: int, bbox: tuple, text: str = ""
    ) -> bool:
        if SECTION_FONT_SUBSTRING not in font:
            return False
        if not (SUBSECTION_SIZE_MIN <= size <= SUBSECTION_SIZE_MAX):
            return False
        if page_num == 1 and bbox[1] < AUTHOR_Y_MAX:
            return False
        # Subsection headings are short labels (e.g. "3.2.1 Scaled Dot-Product Attention").
        # Reject run-in bold labels like "Encoder:" or "Decoder:" that appear inline.
        if text:
            if len(text) > 60:
                return False
            # Run-in labels end with a bare colon and are a single word
            # (e.g. "Encoder:", "Decoder:"). Real subsection headings don't end with ":"
            if text.strip().endswith(":"):
                return False
        return True

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def _extract_tables(
        self, page: fitz.Page, page_num: int
    ) -> tuple[list[dict], list[tuple]]:
        """
        Return (table_chunks, table_bboxes).

        Two strategies:
        1. PyMuPDF find_tables() — works for most tables
        2. Caption-triggered fallback — for tables missed by find_tables()
           (notably Table 2, BLEU scores, which uses mixed math fonts)
        """
        chunks: list[dict] = []
        bboxes: list[tuple] = []

        # Strategy 1: find_tables()
        try:
            tabs = page.find_tables()
        except Exception:
            tabs = None

        if tabs:
            for tab in tabs.tables:
                try:
                    cells = tab.extract()
                except Exception:
                    continue

                # Filter false positives: attention heatmap pages produce
                # very wide tables (30-50 cols) with mostly None/empty cells
                if tab.col_count > MAX_FALSE_POS_TABLE_COLS:
                    continue

                non_empty = sum(
                    1 for row in cells for cell in row
                    if cell is not None and str(cell).strip()
                )
                total = sum(len(row) for row in cells)
                if total > 0 and non_empty / total < 0.4:
                    continue

                caption = self._find_nearby_caption(page, tab.bbox, TABLE_CAPTION_RE)
                markdown = self._cells_to_markdown(cells, caption)

                self._chunk_seq += 1
                chunks.append(self._make_table_chunk(
                    markdown, caption, page_num, tab.bbox
                ))
                bboxes.append(tuple(tab.bbox))

        # Strategy 2: caption-triggered fallback — only for tables NOT already found
        fb_chunks, fb_bboxes = self._caption_fallback_tables(page, page_num, bboxes)
        chunks.extend(fb_chunks)
        bboxes.extend(fb_bboxes)

        # Deduplicate: if both strategies produced a chunk for the same table
        # (same caption text), keep only the first one (find_tables() result).
        seen_captions: set[str] = set()
        deduped: list[dict] = []
        for ch in chunks:
            cap = ch.get("table_caption", "")
            key = cap if cap else ch["chunk_id"]
            if key not in seen_captions:
                seen_captions.add(key)
                deduped.append(ch)
        chunks = deduped

        return chunks, bboxes

    def _find_nearby_caption(
        self, page: fitz.Page, table_bbox: tuple, pattern: re.Pattern, search_px: int = 70
    ) -> str:
        tx0, ty0, tx1, ty1 = table_bbox
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            bx0, by0, bx1, by1 = block["bbox"]
            # Must be horizontally aligned
            if bx1 < tx0 - 60 or bx0 > tx1 + 60:
                continue
            dist_above = ty0 - by1
            dist_below = by0 - ty1
            if 0 <= dist_above <= search_px or 0 <= dist_below <= search_px:
                text = _block_plain_text(block)
                if pattern.match(text):
                    return text
        return ""

    def _cells_to_markdown(self, cells: list[list], caption: str) -> str:
        """Convert extracted table cells into a readable markdown table."""
        if not cells:
            return caption

        def clean_cell(c) -> str:
            if c is None:
                return ""
            return re.sub(r"\s+", " ", str(c)).strip()

        cleaned = [[clean_cell(c) for c in row] for row in cells]
        col_count = max(len(row) for row in cleaned)
        widths = [0] * col_count
        for row in cleaned:
            for i, cell in enumerate(row[:col_count]):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(row: list[str]) -> str:
            cells = [(row[i] if i < len(row) else "") for i in range(col_count)]
            return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        lines = ([caption, ""] if caption else []) + [fmt_row(cleaned[0]), sep]
        for row in cleaned[1:]:
            lines.append(fmt_row(row))
        return "\n".join(lines)

    def _make_table_chunk(
        self, text: str, caption: str, page_num: int, bbox: tuple
    ) -> dict:
        self._chunk_seq += 1
        return {
            "chunk_id": self._make_chunk_id(
                self._current_section, page_num, suffix="table"
            ),
            "text": text,
            "type": "table",
            "section": self._current_section,
            "subsection": self._current_subsection,
            "page_number": page_num,
            "page_end": page_num,
            "source": self.pdf_path.name,
            "paper_title": self.pdf_path.stem,
            "has_equation": False,
            "has_image": False,
            "is_reference": False,
            "table_caption": caption,
            "table_page": page_num,
            "token_count": _count_tokens(text),
        }

    def _caption_fallback_tables(
        self,
        page: fitz.Page,
        page_num: int,
        existing_bboxes: list[tuple],
    ) -> tuple[list[dict], list[tuple]]:
        """
        Fallback for tables not caught by find_tables() (e.g. BLEU score
        table on page 8 whose cells contain mixed math fonts).

        Triggered when a span matches TABLE_CAPTION_RE and is NOT already
        covered by a known table bbox. Collects spans below the caption and
        reconstructs rows by y-coordinate proximity.
        """
        chunks: list[dict] = []
        bboxes: list[tuple] = []
        blocks = page.get_text("dict")["blocks"]

        for block_idx, block in enumerate(blocks):
            if block.get("type") != 0:
                continue
            block_text = _block_plain_text(block)
            if not TABLE_CAPTION_RE.match(block_text):
                continue

            caption_bbox = tuple(block["bbox"])
            # Skip if this caption is already inside a known table region
            if any(_bbox_overlap_ratio(caption_bbox, tb) > 0.3 for tb in existing_bboxes):
                continue

            cap_y1 = caption_bbox[3]
            table_spans: list[dict] = []
            region_x0, region_y0 = float("inf"), cap_y1
            region_x1, region_y1 = 0.0, cap_y1

            for later_block in blocks[block_idx + 1:]:
                if later_block.get("type") != 0:
                    continue
                by0 = later_block["bbox"][1]
                if by0 > cap_y1 + 260:
                    break
                # Stop at a new section heading
                if any(
                    SECTION_FONT_SUBSTRING in s.get("font", "")
                    and s.get("size", 0) >= SECTION_SIZE_MIN
                    for ln in later_block.get("lines", [])
                    for s in ln.get("spans", [])
                ):
                    break
                for ln in later_block.get("lines", []):
                    for s in ln.get("spans", []):
                        t = _clean_text(s["text"])
                        if t:
                            ox, oy = s.get("origin", (0, 0))
                            table_spans.append({"text": t, "x": ox, "y": oy})
                            region_x0 = min(region_x0, ox)
                            region_x1 = max(region_x1, ox + len(t) * 5)
                            region_y1 = max(region_y1, oy)

            if not table_spans:
                continue

            # Group into rows by y (±8pt tolerance)
            rows: dict[int, list[dict]] = {}
            for s in table_spans:
                placed = False
                for key in rows:
                    if abs(s["y"] - key) <= 8:
                        rows[key].append(s)
                        placed = True
                        break
                if not placed:
                    rows[int(s["y"])] = [s]

            table_lines = []
            for _, row_spans in sorted(rows.items()):
                row_spans.sort(key=lambda x: x["x"])
                table_lines.append(" | ".join(sp["text"] for sp in row_spans))

            table_text = block_text + "\n\n" + "\n".join(table_lines)
            chunk = self._make_table_chunk(
                table_text, block_text, page_num,
                (region_x0, region_y0, region_x1, region_y1 + 10)
            )
            chunks.append(chunk)
            if region_x1 > region_x0:
                bboxes.append((region_x0, region_y0, region_x1, region_y1 + 10))

        return chunks, bboxes

    # ------------------------------------------------------------------
    # Figure chunks
    # ------------------------------------------------------------------

    def _make_figure_chunk(self, caption_text: str, page_num: int) -> dict:
        self._chunk_seq += 1
        return {
            "chunk_id": self._make_chunk_id(
                self._current_section, page_num, suffix="figure"
            ),
            "text": f"[FIGURE] {caption_text}",
            "type": "figure",
            "section": self._current_section,
            "subsection": self._current_subsection,
            "page_number": page_num,
            "page_end": page_num,
            "source": self.pdf_path.name,
            "paper_title": self.pdf_path.stem,
            "has_equation": False,
            "has_image": True,
            "is_reference": False,
            "table_caption": "",
            "table_page": -1,
            "token_count": _count_tokens(caption_text),
        }

    # ------------------------------------------------------------------
    # Footnote chunks
    # ------------------------------------------------------------------

    def _try_make_footnote(self, block: dict, page_num: int) -> Optional[dict]:
        """
        Emit substantive footnotes as their own chunk.
        Footnotes in this paper contain important explanations
        (e.g. why dot products are scaled by 1/sqrt(dk)).
        """
        parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 10) < 9.2:
                    t = _clean_text(span["text"])
                    if t:
                        parts.append(t)

        if not parts:
            return None

        text = " ".join(parts)
        if len(text) < 25:
            return None

        self._chunk_seq += 1
        return {
            "chunk_id": self._make_chunk_id(
                self._current_section, page_num, suffix="footnote"
            ),
            "text": text,
            "type": "footnote",
            "section": self._current_section,
            "subsection": self._current_subsection,
            "page_number": page_num,
            "page_end": page_num,
            "source": self.pdf_path.name,
            "paper_title": self.pdf_path.stem,
            "has_equation": False,
            "has_image": False,
            "is_reference": False,
            "table_caption": "",
            "table_page": -1,
            "token_count": _count_tokens(text),
        }

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _make_chunk_id(
        self, section: str, page_num: int, suffix: str = ""
    ) -> str:
        """
        Deterministic chunk ID for ChromaDB upsert idempotency.
        Format: {pdf_stem}_{section_slug}_p{page}_{seq:04d}[_{suffix}]
        """
        stem = _slugify(self.pdf_path.stem)
        sec = _slugify(section)
        parts = [stem, sec, f"p{page_num}", f"{self._chunk_seq:04d}"]
        if suffix:
            parts.append(suffix)
        return "_".join(parts)
