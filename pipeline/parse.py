"""
Stage 1: parsing. Turn a PDF into text a chunker can work with.

Hobby project note: this is a demo of the idea, not a production parser.

Three ways to read a PDF are offered so you can compare them side by side:

  raw        pypdf's plain text extraction. Fast, but you get hard line wraps,
             running headers and footers on every page, and tables flattened
             into a column of loose values. This is the "before" picture.

  pymupdf    pymupdf4llm. Lightweight PDF -> Markdown. Recognises headings by
             font size and rebuilds tables. The default, because it installs
             in seconds and runs fine on free hosting.

  docling    IBM's Docling. Layout-aware, uses ML models, usually the best on
             messy documents. Optional: it pulls in PyTorch, so it's too heavy
             for free hosting. Install it locally with requirements-docling.txt.

After parsing, clean_markdown() removes lines that repeat on most pages
(running headers and footers) and tidies heading markup.
"""

from __future__ import annotations

import io
import re
import tempfile
from collections import Counter
from pathlib import Path

ENGINES = {
    "raw": "Raw text (pypdf), the naive baseline",
    "pymupdf": "PyMuPDF4LLM (lightweight, default)",
    "docling": "Docling (layout-aware, heavy, optional)",
}


def available_engines() -> list[str]:
    """Which parsers are installed. Docling is optional."""
    out = ["raw"]
    try:
        import pymupdf4llm  # noqa: F401
        out.append("pymupdf")
    except Exception:
        pass
    try:
        import docling  # noqa: F401
        out.append("docling")
    except Exception:
        pass
    return out


def raw_text(pdf_bytes: bytes) -> tuple[str, int]:
    """The naive way: pypdf text extraction, page by page. Returns (text, pages)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [pg.extract_text() or "" for pg in reader.pages]
    return "\n\n".join(pages), len(pages)


def _with_temp_file(pdf_bytes: bytes, fn):
    # Both Markdown parsers want a file path, so write the upload to a temp file.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "upload.pdf"
        path.write_bytes(pdf_bytes)
        return fn(str(path))


def to_markdown(pdf_bytes: bytes, engine: str = "pymupdf") -> str:
    """Parse a PDF into Markdown with the chosen engine (before cleanup)."""
    if engine == "raw":
        return raw_text(pdf_bytes)[0]
    if engine == "pymupdf":
        import pymupdf4llm

        return _with_temp_file(pdf_bytes, lambda p: pymupdf4llm.to_markdown(p))
    if engine == "docling":
        from docling.document_converter import DocumentConverter

        def run(p):
            result = DocumentConverter().convert(p)
            return result.document.export_to_markdown()

        return _with_temp_file(pdf_bytes, run)
    raise ValueError(f"Unknown parser: {engine}")


# Lines like "Page 3 | v3.2" differ only by their numbers. Swapping digits for
# "#" lets us count them as the same line when looking for repeats.
def _shape(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_markdown(md: str, pages: int | None = None) -> tuple[str, list[str]]:
    """
    Tidy parser output. Returns (clean_markdown, removed_lines).

    1. Drop running headers and footers: short lines whose shape repeats on
       at least half of the pages. Simple, and good enough for a demo.
    2. Remove bold/italic markers inside headings ("### **_1.1 Vacation_**")
       and bold markers elsewhere (table headers).
    3. Collapse runs of blank lines.
    """
    lines = md.split("\n")
    counts = Counter(_shape(l) for l in lines if l.strip() and not l.lstrip().startswith(("#", "|")))
    if pages is None:
        # Guess the page count from the most repeated short line.
        pages = max((c for s, c in counts.items() if len(s) < 120), default=1)
    threshold = max(2, (pages + 1) // 2)
    noisy = {s for s, c in counts.items() if c >= threshold and len(s) < 120}

    kept, removed = [], []
    for line in lines:
        if line.strip() and _shape(line) in noisy:
            removed.append(line.strip())
            continue
        if line.lstrip().startswith("#"):
            hashes, _, title = line.strip().partition(" ")
            title = re.sub(r"[*_]+", "", title).strip()
            line = f"{hashes} {title}"
        kept.append(line.rstrip())

    text = "\n".join(kept).replace("**", "")  # bold markers add noise to search
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n", sorted(set(removed))


def parse(pdf_bytes: bytes, engine: str = "pymupdf") -> dict:
    """Run one engine and clean the result. Returns everything the UI shows."""
    raw, pages = raw_text(pdf_bytes)
    if engine == "raw":
        return {"engine": engine, "pages": pages, "markdown": raw, "removed": [], "raw": raw}
    md = to_markdown(pdf_bytes, engine)
    clean, removed = clean_markdown(md, pages)
    return {"engine": engine, "pages": pages, "markdown": clean, "removed": removed, "raw": raw}
