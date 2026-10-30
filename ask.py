"""
ask.py: a tiny command-line "toolbox" for agentic RAG over the sample handbook.

Hobby project note: a demo, not production code.

The idea: an AI agent (Claude in a chat, or Claude Code on your laptop) runs
these commands as its tools. It decides what to search for, reads what comes
back, and searches again until it can answer. No Anthropic API key needed;
the agent is whatever Claude you're already talking to.

  python ask.py load                       parse + chunk the sample, write to turbopuffer
  python ask.py sections                   table of contents (helps the agent plan)
  python ask.py search "query" [--mode hybrid|vector|keyword] [--section "Travel"] [-k 4]
  python ask.py around 7                   chunk 7 plus its neighbours (small to big)

Code search (any local folder of Python code; defaults to this repo):
  python ask.py code-load [--dir PATH]     chunk every function and write to turbopuffer
  python ask.py code-search "query" [--dir PATH] [--mode hybrid|vector|keyword] [-k 5]
  python ask.py grep "pattern" [--dir PATH]
  python ask.py read path/to/file.py 120 160 [--dir PATH]

Needs TURBOPUFFER_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import chunk, codesearch, parse, search

SAMPLE = Path(__file__).parent / "samples" / "harbor-lane-handbook.pdf"
STRATEGY = "recursive"
CHUNK_CHARS = 1200  # ~300 tokens, turbopuffer's suggested starting point


def _doc():
    pdf = SAMPLE.read_bytes()
    return pdf, search.doc_id_for(SAMPLE.name, pdf)


def _print_rows(rows):
    for r in rows:
        score = r.get("score")
        print(f"[chunk {r.get('chunk_index')}] score={score} section={r.get('section')!r}")
        print("   " + (r.get("content") or "").replace("\n", "\n   "))
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("load")
    sub.add_parser("sections")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=search.MODES)
    s.add_argument("--section", help="only search chunks whose section starts with this text")
    s.add_argument("-k", type=int, default=4)
    a = sub.add_parser("around")
    a.add_argument("chunk_index", type=int)
    for name in ("code-load", "code-search", "grep", "read"):
        c = sub.add_parser(name)
        c.add_argument("--dir", default=str(codesearch.REPO_ROOT), help="folder of Python code (default: this repo)")
        if name == "code-search":
            c.add_argument("query")
            c.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "keyword"])
            c.add_argument("-k", type=int, default=5)
        elif name == "grep":
            c.add_argument("pattern")
        elif name == "read":
            c.add_argument("path")
            c.add_argument("start", type=int)
            c.add_argument("end", type=int)
    args = ap.parse_args(argv)

    if args.cmd in ("grep", "read", "code-load", "code-search"):
        return code_cmd(args)

    pdf, doc_id = _doc()
    tpuf = search.client()

    if args.cmd == "load":
        md = parse.parse(pdf, "pymupdf")["markdown"]
        chunks = chunk.recursive_chunks(md, CHUNK_CHARS, overlap_chars=150, with_context=True)
        info = search.upload(tpuf, doc_id, STRATEGY, [c.to_row(doc_id) for c in chunks])
        print(f"Wrote {info['rows']} chunks to {info['namespace']} in {info['seconds']}s")
    elif args.cmd == "sections":
        md = parse.parse(pdf, "pymupdf")["markdown"]
        for sec in chunk.split_sections(md):
            print(sec.path)
    elif args.cmd == "search":
        r = search.search(tpuf, doc_id, STRATEGY, args.query, args.mode, args.k, section=args.section)
        print(f"# {args.mode} search, {r['ms']} ms, {len(r['rows'])} results"
              + (f", section filter {args.section!r}" if args.section else ""))
        if r["note"]:
            print("# note:", r["note"])
        _print_rows(r["rows"])
    elif args.cmd == "around":
        _print_rows(search.expand(tpuf, doc_id, STRATEGY, args.chunk_index))


def code_cmd(args):
    files = codesearch.files_from_dir(args.dir)
    source = Path(args.dir).resolve().name
    if args.cmd == "grep":
        g = codesearch.grep(files, args.pattern)
        print(f"# {g['total']} matching lines" + (" (showing 40)" if g["total"] > 40 else ""))
        print("\n".join(g["hits"]))
    elif args.cmd == "read":
        print(codesearch.read_lines(files, args.path, args.start, args.end))
    elif args.cmd == "code-load":
        rows = codesearch.chunk_files(files)
        info = codesearch.load(search.client(), source, rows)
        print(f"Wrote {info['chunks']} chunks from {len(files)} files to {info['namespace']} in {info['seconds']}s")
    elif args.cmd == "code-search":
        r = codesearch.search_code(search.client(), source, args.query, args.mode, args.k)
        print(f"# {args.mode} search, {r['ms']} ms")
        print(codesearch.hits_text(r["rows"]))


if __name__ == "__main__":
    sys.exit(main())
