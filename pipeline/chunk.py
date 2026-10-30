"""
Stage 2: chunking. Split the parsed Markdown into pieces small enough to
embed and search, two different ways, so you can compare them.

Hobby project note: written for clarity, not speed or completeness.

Structure-aware recursive (off the shelf, via Chonkie)
    First split the document at its headings, so no chunk crosses a section
    boundary. Inside each section, Chonkie's RecursiveChunker tries the
    biggest separator first (blank lines between paragraphs), then sentence
    endings, then smaller ones, until every piece fits the size budget.
    Cheap, predictable, and it respects how the author organised things.

Semantic (hand-written here, about 40 lines, on purpose)
    Ignore the headings. Split the text into sentences, embed each one with a
    small local model, and measure how similar each sentence is to the next.
    Cut where similarity drops the most: that's where the topic changed.
    Chonkie ships a SemanticChunker too, but it smooths the scores over
    windows, which is harder to show on screen. This version exposes every
    score so the demo can display where and why it cut.

Both return the same Chunk shape, so the search stage doesn't care which
strategy made them.

"Context" option: when on, each chunk is stored with its heading path in
front of it ("Time off > Parental leave"). That's the cheapest fix for chunks
that lose meaning once they're cut out of the document.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

# Character counts, not tokens, to keep the demo dependency-free and easy to
# read. Rule of thumb: about 4 characters per token in English.
DEFAULT_CHUNK_CHARS = 800
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "minishlab/potion-base-32M")


@dataclass
class Chunk:
    strategy: str                 # "recursive" or "semantic"
    index: int                    # position in the document, 0-based
    text: str                     # the chunk itself, as displayed
    section: str                  # heading path of the section it starts in
    sections: list[str] = field(default_factory=list)  # every section it touches
    with_context: bool = True     # was the heading path prepended?

    @property
    def content(self) -> str:
        """What gets embedded and keyword-indexed in turbopuffer."""
        if self.with_context and self.section:
            return f"{self.section}\n{self.text}"
        return self.text

    def to_row(self, doc_id: str) -> dict:
        """One turbopuffer row. The id is stable, so re-uploading overwrites."""
        return {
            "id": f"{doc_id}-{self.strategy}-{self.index:04d}",
            "content": self.content,
            "section": self.section,
            "chunk_index": self.index,
            "strategy": self.strategy,
            "doc_id": doc_id,
        }

    def as_dict(self) -> dict:
        d = asdict(self)
        d["content"] = self.content
        d["chars"] = len(self.text)
        return d


# ---------------------------------------------------------------------------
# Shared helper: read the Markdown into sections with their heading paths
# ---------------------------------------------------------------------------
@dataclass
class Section:
    path: str      # "Harbor Lane Employee Handbook > Time off > Parental leave"
    body: str      # the text under that heading (not including sub-sections)


def split_sections(md: str, keep_top_title: bool = False) -> list[Section]:
    """
    Walk the Markdown headings and return each section's text with its path.

    The document title (the single top-level "#") is usually the same for
    every chunk, so it's left out of the path unless keep_top_title is True.
    """
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    body: list[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            names = [t for lvl, t in stack if keep_top_title or lvl > 1]
            if not names and stack:
                names = [stack[0][1]]  # text right under the title, e.g. an intro
            sections.append(Section(" > ".join(names), text))
        body.clear()

    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            # Drop numbering like "1.3" from the path; it adds noise to embeddings.
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    flush()
    return sections


def _add_overlap(pieces: list[str], overlap_chars: int) -> list[str]:
    """Prefix each piece with the tail of the one before it (whole words only)."""
    if overlap_chars <= 0:
        return pieces
    out = [pieces[0]] if pieces else []
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap_chars:]
        tail = tail[tail.find(" ") + 1:] if " " in tail else tail
        out.append(f"…{tail.strip()} {cur}")
    return out


# ---------------------------------------------------------------------------
# Strategy 1: structure-aware recursive splitting
# ---------------------------------------------------------------------------
def recursive_chunks(md: str, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap_chars: int = 100,
                     with_context: bool = True) -> list[Chunk]:
    from chonkie import RecursiveChunker

    splitter = RecursiveChunker(chunk_size=chunk_chars, min_characters_per_chunk=24)
    chunks: list[Chunk] = []
    for sec in split_sections(md):
        pieces = [c.text.strip() for c in splitter.chunk(sec.body) if c.text.strip()]
        for text in _add_overlap(pieces, overlap_chars):
            chunks.append(Chunk("recursive", len(chunks), text, sec.path, [sec.path], with_context))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: semantic chunking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _sentence_model():
    # model2vec "static" embeddings: tiny, CPU-only, no API key. Good enough
    # to spot topic shifts. Downloads about 130 MB from Hugging Face once.
    from model2vec import StaticModel

    return StaticModel.from_pretrained(SEMANTIC_MODEL)


def _units(md: str) -> list[tuple[str, str]]:
    """
    Break the document into (sentence, section_path) units.

    Headings are dropped on purpose: semantic chunking only looks at the
    words, the way it would on a transcript or email with no structure.
    Tables stay whole, because splitting a table into "sentences" is nonsense.
    """
    units = []
    for sec in split_sections(md):
        for block in re.split(r"\n\s*\n", sec.body):
            block = block.strip()
            if not block:
                continue
            if block.startswith("|"):
                units.append((block, sec.path))
                continue
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", " ".join(block.split())):
                if s.strip():
                    units.append((s.strip(), sec.path))
    return units


def semantic_chunks(md: str, breakpoint_percentile: int = 20, max_chars: int = 1500,
                    min_chars: int = 200, with_context: bool = True) -> tuple[list[Chunk], list[dict]]:
    """
    Returns (chunks, gaps). Each gap describes the boundary between two
    neighbouring sentences: their similarity and whether we cut there.

    Cut rule, in plain words:
      - cut where the similarity is in the lowest `breakpoint_percentile`
        percent of all gaps (the biggest topic shifts in this document),
      - but never make a chunk shorter than min_chars,
      - and always cut before a chunk would grow past max_chars.
    """
    units = _units(md)
    if not units:
        return [], []
    texts = [u[0] for u in units]
    vecs = np.asarray(_sentence_model().encode(texts), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)  # cosine similarity of each neighbour pair
    cutoff = float(np.percentile(sims, breakpoint_percentile)) if len(sims) else 0.0

    chunks: list[Chunk] = []
    gaps: list[dict] = []
    cur: list[int] = [0]

    def emit(idxs):
        text = " ".join(texts[i] for i in idxs)
        secs = list(dict.fromkeys(units[i][1] for i in idxs))
        chunks.append(Chunk("semantic", len(chunks), text, secs[0], secs, with_context))

    for i, sim in enumerate(sims):
        cur_len = sum(len(texts[j]) + 1 for j in cur)
        nxt_len = len(texts[i + 1])
        topic_shift = sim <= cutoff
        too_big = cur_len + nxt_len > max_chars
        cut = (topic_shift and cur_len >= min_chars) or too_big
        reason = "topic shift" if topic_shift and cut else ("size limit" if cut else "")
        gaps.append({"after": texts[i][-60:], "before": texts[i + 1][:60], "similarity": round(float(sim), 3),
                     "cut": cut, "reason": reason, "cross_section": units[i][1] != units[i + 1][1]})
        if cut:
            emit(cur)
            cur = []
        cur.append(i + 1)
    emit(cur)
    return chunks, gaps


def summary(chunks: list[Chunk]) -> dict:
    sizes = [len(c.text) for c in chunks] or [0]
    return {
        "chunks": len(chunks),
        "avg_chars": int(np.mean(sizes)),
        "min_chars": int(min(sizes)),
        "max_chars": int(max(sizes)),
        "span_sections": sum(1 for c in chunks if len(c.sections) > 1),
    }
