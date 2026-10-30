"""
Stage 3: store chunks in turbopuffer and search them.

Hobby project note: a demo of the ideas, with minimal error handling.

How the data is laid out
    One namespace per (document, chunking strategy), e.g.
    "tpragdemo-harbor-lane-handbook-recursive". Namespaces are created by
    their first write and cost almost nothing when idle, so giving each
    strategy its own namespace makes side-by-side comparison trivial.

Embedding
    turbopuffer's native embedding does the work: the schema tells it to
    embed the "content" attribute on write, and queries pass ("Embed", text)
    so the question is embedded the same way. No separate embedding service.

Search modes
    vector   ANN over the embeddings: finds passages by meaning
    keyword  BM25 full-text: finds exact words, codes and numbers
    hybrid   both at once via multi_query, fused with reciprocal rank fusion
    rerank   hybrid, then a reranking model re-scores the candidates
             (depends on your plan; falls back to hybrid if unavailable)

"Small to big"
    expand() fetches the chunks either side of a hit with a plain filter on
    chunk_index, so you can search small chunks but hand the LLM more context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

REGION = os.getenv("TURBOPUFFER_REGION", "gcp-us-central1")
PREFIX = os.getenv("NAMESPACE_PREFIX", "tpragdemo")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-8b")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1024"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "zeroentropy/zerank-2")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SCHEMA = {
    "content": {
        "type": "string",
        "full_text_search": True,                               # BM25 keyword index
        "embed": {"model": EMBED_MODEL, "dims": EMBED_DIMS},    # native embedding
    }
}
ATTRS = ["content", "section", "chunk_index", "strategy"]
MODES = ["vector", "keyword", "hybrid", "rerank"]


def client(api_key: str | None = None):
    import turbopuffer

    key = api_key or os.getenv("TURBOPUFFER_API_KEY")
    if not key:
        raise RuntimeError("Set TURBOPUFFER_API_KEY (see .env.example).")
    # TURBOPUFFER_BASE_URL is only for testing against a local fake server.
    base_url = os.getenv("TURBOPUFFER_BASE_URL")
    if base_url:
        return turbopuffer.Turbopuffer(api_key=key, base_url=base_url)
    return turbopuffer.Turbopuffer(api_key=key, region=REGION)


def doc_id_for(filename: str, data: bytes) -> str:
    """A short, readable id: file name slug plus a few characters of its hash."""
    stem = re.sub(r"[^a-z0-9]+", "-", os.path.splitext(filename)[0].lower()).strip("-")[:40]
    return f"{stem}-{hashlib.sha1(data).hexdigest()[:6]}"


def namespace_name(doc_id: str, strategy: str) -> str:
    return f"{PREFIX}-{doc_id}-{strategy}"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def upload(tpuf, doc_id: str, strategy: str, rows: list[dict], batch: int = 200) -> dict:
    """
    Replace the namespace's contents with these chunks.

    The namespace is wiped first, because re-chunking with different settings
    produces a different number of chunks and stale ones would linger.
    """
    ns = tpuf.namespace(namespace_name(doc_id, strategy))
    try:
        ns.delete_all()
    except Exception:
        pass  # didn't exist yet, which is fine
    t0 = time.time()
    for s in range(0, len(rows), batch):
        ns.write(upsert_rows=rows[s:s + batch], distance_metric="cosine_distance", schema=SCHEMA)
    return {"namespace": ns.id if hasattr(ns, "id") else namespace_name(doc_id, strategy),
            "rows": len(rows), "seconds": round(time.time() - t0, 2)}


def delete_demo_namespaces(tpuf) -> int:
    """Remove every namespace this app created (they all share PREFIX)."""
    names = [n.id for n in tpuf.namespaces(prefix=f"{PREFIX}-")]
    for name in names:
        tpuf.namespace(name).delete_all()
    return len(names)


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------
def _row(r) -> dict:
    d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
    score = d.get("$dist", d.get("$score"))
    d["score"] = round(float(score), 4) if isinstance(score, (int, float)) else None
    return d


def search(tpuf, doc_id: str, strategy: str, question: str, mode: str = "hybrid", top_k: int = 5,
           section: str | None = None) -> dict:
    """
    Run one search. Returns rows plus timing and a note if something fell back.
    section: optional prefix such as "Travel" to search only that part of the
    document (a turbopuffer Glob filter on the section attribute).
    """
    ns = tpuf.namespace(namespace_name(doc_id, strategy))
    by_meaning = ("content", "ANN", ("Embed", question))
    by_keyword = ("content", "BM25", question)
    base = {"top_k": top_k, "include_attributes": ATTRS}
    if section:
        base["filters"] = ("section", "Glob", f"{section}*")
    note = ""
    t0 = time.time()
    if mode == "vector":
        rows = ns.query(rank_by=by_meaning, **base).rows
    elif mode == "keyword":
        rows = ns.query(rank_by=by_keyword, **base).rows
    else:
        # Fetch extra candidates from each side, then fuse (or rerank) them.
        sub = dict(base, top_k=top_k * 3)
        queries = [dict(sub, rank_by=by_meaning), dict(sub, rank_by=by_keyword)]
        if mode == "rerank":
            try:
                res = ns.multi_query(queries=queries,
                                     rerank_by={"model": RERANK_MODEL, "attribute": "content", "query": question})
            except Exception as e:
                note = f"Reranking isn't available here ({str(e)[:100]}), so these are fused hybrid results."
                res = ns.multi_query(queries=queries, rerank_by=("RRF",))
        else:
            res = ns.multi_query(queries=queries, rerank_by=("RRF",))
        rows = res.results[0].rows
    out = [_row(r) for r in rows][:top_k]
    return {"rows": out, "ms": round((time.time() - t0) * 1000), "note": note,
            "code": code_for(namespace_name(doc_id, strategy), mode, question, top_k)}


def expand(tpuf, doc_id: str, strategy: str, chunk_index: int, radius: int = 1) -> list[dict]:
    """Small to big: return the hit plus its neighbours, in document order."""
    ns = tpuf.namespace(namespace_name(doc_id, strategy))
    wanted = list(range(max(0, chunk_index - radius), chunk_index + radius + 1))
    res = ns.query(filters=("chunk_index", "In", wanted), rank_by=("chunk_index", "asc"),
                   top_k=len(wanted), include_attributes=ATTRS)
    return [_row(r) for r in res.rows]


def code_for(ns_name: str, mode: str, q: str, top_k: int) -> str:
    """The exact SDK call, shown in the UI so the demo isn't a black box."""
    q_ = json.dumps(q)
    if mode in ("vector", "keyword"):
        rank = f'("content", "ANN", ("Embed", {q_}))' if mode == "vector" else f'("content", "BM25", {q_})'
        return f'ns = tpuf.namespace("{ns_name}")\nres = ns.query(\n    rank_by={rank},\n    top_k={top_k},\n)'
    rr = '("RRF",)' if mode == "hybrid" else f'{{"model": "{RERANK_MODEL}", "attribute": "content", "query": {q_}}}'
    return (f'ns = tpuf.namespace("{ns_name}")\nres = ns.multi_query(\n    queries=[\n'
            f'        {{"rank_by": ("content", "ANN", ("Embed", {q_})), "top_k": {top_k * 3}}},\n'
            f'        {{"rank_by": ("content", "BM25", {q_}), "top_k": {top_k * 3}}},\n'
            f'    ],\n    rerank_by={rr},\n)')


# ---------------------------------------------------------------------------
# Answering (optional)
# ---------------------------------------------------------------------------
SYSTEM = ("You answer questions about a document using only the numbered passages provided. "
          "Cite passages like [1] or [2]. If the passages don't contain the answer, say so plainly "
          "instead of guessing. Keep answers to a few sentences.")


def answer(question: str, rows: list[dict], api_key: str | None = None) -> str:
    """Ask Claude to answer from the retrieved chunks, with citations."""
    import anthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to get a written answer.")
    passages = "\n\n".join(f"[{i}] {r.get('content', '')}" for i, r in enumerate(rows, 1))
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
