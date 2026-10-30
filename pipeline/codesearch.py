"""
Code search: index a Python codebase by function and search it by meaning.

Hobby project note: a teaching demo. Python only, and deliberately simple.

Why chunk by function?
    Text chunkers cut code at arbitrary line counts, which splits functions
    in half. Here Python's own parser (the ast module) finds every function,
    method and class, and each one becomes a chunk with its file path, name,
    line numbers and docstring. For many languages at once you'd use
    tree-sitter instead; turbopuffer's chunking guide suggests the same.

What gets stored per chunk
    content    "path :: kind name" + docstring + the code itself. This is
               what turbopuffer embeds and keyword-indexes.
    path, symbol, kind, start_line, end_line   for filters and for reading
               the exact lines back later.

Sources
    "this demo"  the files of this repo (no network needed)
    GitHub       any public repo, downloaded as a zip (Python files only,
                 capped so a huge repo can't run away with your bill)

grep and read_lines are here too, because the agent comparison pits a
grep-and-read agent (how coding agents worked before search) against one
that can search turbopuffer.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

from . import search

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_FILES = 400
MAX_FILE_BYTES = 200_000
MAX_CHUNKS = 3000
MAX_CODE_CHARS = 6000   # long functions are truncated; embedding models truncate anyway
ATTRS = ["content", "path", "symbol", "kind", "start_line", "end_line"]

EXAMPLES_THIS_REPO = [
    "Where does the app strip page headers and footers out of a PDF?",
    "How does the semantic chunker decide where to cut?",
    "What happens when the agent runs out of tool-call rounds?",
    "How is the password checked?",
]
EXAMPLES_GITHUB = {
    "psf/requests": [
        "How many redirects will requests follow before giving up?",
        "How does requests decide which proxy to use for a URL?",
        "Where are cookies from a response merged back into the session?",
    ],
}


# ---------------------------------------------------------------------------
# Getting files
# ---------------------------------------------------------------------------
def files_from_dir(root: str | Path) -> dict[str, str]:
    """Every .py file under a local folder, e.g. a repo you've cloned."""
    root = Path(root).resolve()
    out = {}
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part.startswith(".") or part in ("venv", "__pycache__", "node_modules") for part in rel.parts):
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        out[rel.as_posix()] = p.read_text(errors="replace")
        if len(out) >= MAX_FILES:
            break
    return out


def files_from_this_repo() -> dict[str, str]:
    return files_from_dir(REPO_ROOT)


def files_from_github(repo: str) -> dict[str, str]:
    """Download a public repo's default branch as a zip and keep its .py files."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Use the form owner/repo, e.g. psf/requests")
    url = f"https://github.com/{repo}/archive/HEAD.zip"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read(60_000_000)  # 60 MB cap
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            name = info.filename.split("/", 1)[-1]   # drop the "repo-sha/" prefix
            if not name.endswith(".py") or info.file_size > MAX_FILE_BYTES:
                continue
            out[name] = z.read(info).decode("utf-8", errors="replace")
            if len(out) >= MAX_FILES:
                break
    if not out:
        raise ValueError(f"No Python files found in {repo}")
    return out


# ---------------------------------------------------------------------------
# Chunking by function
# ---------------------------------------------------------------------------
def _chunk_row(path, symbol, kind, start, end, doc, code) -> dict:
    code = code if len(code) <= MAX_CODE_CHARS else code[:MAX_CODE_CHARS] + "\n# … truncated"
    header = f"{path} :: {kind} {symbol}"
    content = "\n".join(x for x in (header, doc or "", code) if x)
    rid = hashlib.sha1(f"{path}::{symbol}::{start}".encode()).hexdigest()[:24]
    return {"id": rid, "content": content, "path": path, "symbol": symbol, "kind": kind,
            "start_line": start, "end_line": end}


def chunk_file(path: str, source: str) -> list[dict]:
    """One chunk per function, method and class (class = its header and docstring)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out = []
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        out.append(_chunk_row(path, "(module)", "module", 1, 1, mod_doc, ""))

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                code = "\n".join(lines[start - 1:child.end_lineno])
                kind = "method" if prefix else "function"
                out.append(_chunk_row(path, name, kind, start, child.end_lineno, ast.get_docstring(child), code))
                visit(child, f"{name}.")      # nested functions
            elif isinstance(child, ast.ClassDef):
                name = f"{prefix}{child.name}"
                first_def = next((n.lineno for n in child.body
                                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
                                 child.end_lineno + 1)
                head = "\n".join(lines[child.lineno - 1:first_def - 1])
                out.append(_chunk_row(path, name, "class", child.lineno, child.end_lineno,
                                      ast.get_docstring(child), head))
                visit(child, f"{name}.")
    visit(tree)
    return out


def chunk_files(files: dict[str, str]) -> list[dict]:
    rows = []
    for path, src in files.items():
        rows.extend(chunk_file(path, src))
        if len(rows) >= MAX_CHUNKS:
            break
    return rows[:MAX_CHUNKS]


# ---------------------------------------------------------------------------
# turbopuffer
# ---------------------------------------------------------------------------
def namespace_for(source: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")[:40] or "this-demo"
    return f"{search.PREFIX}-code-{slug}"


def load(tpuf, source: str, rows: list[dict], batch: int = 200) -> dict:
    ns = tpuf.namespace(namespace_for(source))
    try:
        ns.delete_all()
    except Exception:
        pass
    t0 = time.time()
    for s in range(0, len(rows), batch):
        ns.write(upsert_rows=rows[s:s + batch], distance_metric="cosine_distance", schema=search.SCHEMA)
    return {"namespace": namespace_for(source), "chunks": len(rows), "seconds": round(time.time() - t0, 2)}


def search_code(tpuf, source: str, query: str, mode: str = "hybrid", top_k: int = 5,
                path_prefix: str | None = None) -> dict:
    ns = tpuf.namespace(namespace_for(source))
    by_meaning = ("content", "ANN", ("Embed", query))
    by_keyword = ("content", "BM25", query)
    base = {"top_k": top_k, "include_attributes": ATTRS}
    if path_prefix:
        base["filters"] = ("path", "Glob", f"{path_prefix}*")
    t0 = time.time()
    if mode == "vector":
        rows = ns.query(rank_by=by_meaning, **base).rows
    elif mode == "keyword":
        rows = ns.query(rank_by=by_keyword, **base).rows
    else:
        sub = dict(base, top_k=top_k * 3)
        rows = ns.multi_query(queries=[dict(sub, rank_by=by_meaning), dict(sub, rank_by=by_keyword)],
                              rerank_by=("RRF",)).results[0].rows
    out = [search._row(r) for r in rows][:top_k]
    return {"rows": out, "ms": round((time.time() - t0) * 1000)}


def hits_text(rows: list[dict]) -> str:
    """Compact result list for an agent: where things are, not the whole code."""
    if not rows:
        return "(no results)"
    out = []
    for r in rows:
        body = (r.get("content") or "").split("\n", 1)[-1]
        out.append(f"{r.get('path')}:{r.get('start_line')}-{r.get('end_line')}  {r.get('kind')} {r.get('symbol')}\n"
                   + "\n".join("    " + l for l in body.splitlines()[:6]))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# The grep side: what a coding agent does without a search index
# ---------------------------------------------------------------------------
def grep(files: dict[str, str], pattern: str, path_prefix: str | None = None, limit: int = 40) -> dict:
    """Case-insensitive regex over every line. Returns matches and how many there were in total."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    t0 = time.time()
    hits, total = [], 0
    for path, src in files.items():
        if path_prefix and not path.startswith(path_prefix):
            continue
        for n, line in enumerate(src.splitlines(), 1):
            if rx.search(line):
                total += 1
                if len(hits) < limit:
                    hits.append(f"{path}:{n}: {line.strip()[:160]}")
    return {"hits": hits, "total": total, "files": len({h.split(':', 1)[0] for h in hits}),
            "ms": round((time.time() - t0) * 1000)}


def read_lines(files: dict[str, str], path: str, start: int, end: int, cap: int = 120) -> str:
    src = files.get(path)
    if src is None:
        return f"No such file: {path}"
    lines = src.splitlines()
    start = max(1, int(start))
    end = min(len(lines), int(end), start + cap - 1)
    return "\n".join(f"{n:5}  {lines[n - 1]}" for n in range(start, end + 1))
