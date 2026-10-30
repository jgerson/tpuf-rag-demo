"""
Stage 5: classic RAG vs agentic RAG, with a full trace of every call.

Hobby project note: a teaching demo. The loop is deliberately short and
explicit so you can read exactly what happens, not a production agent.

Classic RAG (your code decides everything)
    1. one hybrid search with the question exactly as typed
    2. the top chunks go into one prompt
    3. one Claude call writes the answer

Agentic RAG (Claude decides what to search for)
    1. Claude gets the question plus three tool descriptions, no chunks
    2. Claude replies with stop_reason "tool_use" and one or more tool calls
    3. this code runs them (turbopuffer queries, or a local table of contents)
       and sends the results back as "tool_result" blocks
    4. repeat until Claude replies with stop_reason "end_turn" (the answer)
       or the step cap is hit, at which point Claude must answer with what it has

The API is stateless, so every turn resends the whole conversation so far.
That's why the trace shows input tokens growing turn by turn.

Every request and response is recorded as a trace step, with the text Claude
wrote alongside its tool calls (its stated reason for them), so the app can
show the run step by step.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from . import chunk, search

MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "6"))

TOOLS = [
    {
        "name": "sections",
        "description": "List the document's section headings (its table of contents). Cheap; use it first "
                       "to plan which sections a multi-part question needs.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_docs",
        "description": "Search the document in turbopuffer. mode: 'keyword' (BM25) for exact codes, names or "
                       "numbers; 'vector' for meaning when wording may differ; 'hybrid' (both, fused) when unsure. "
                       "section: optional heading prefix such as 'Travel' or 'Expenses > Approvals' to search only "
                       "that part. Ask several narrow searches rather than one broad one.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string", "enum": ["hybrid", "vector", "keyword"]},
                "section": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_around",
        "description": "Return a chunk plus the chunks just before and after it, when a result looks cut off.",
        "input_schema": {"type": "object", "properties": {"chunk_index": {"type": "integer"}},
                         "required": ["chunk_index"]},
    },
]

AGENT_SYSTEM = (
    "You answer questions about one document using the tools provided. For questions with several parts, "
    "look at the sections first, then run one narrow search per part, filtered to the right section. "
    "Before each tool call, say in one short sentence why you're making it. Answer only from text the tools "
    "returned, cite chunks like [chunk 7], and say plainly if something isn't in the document. "
    "Keep the final answer to a few sentences."
)


@dataclass
class Step:
    """One row in the trace the app displays."""
    actor: str          # "user", "app → turbopuffer", "app → Claude", "Claude → app", "tool (local)"
    title: str
    sent: str = ""
    received: str = ""
    why: str = ""
    ms: int | None = None


@dataclass
class Run:
    kind: str                       # "classic" or "agent"
    question: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    llm_turns: int = 0
    tpuf_queries: int = 0
    tpuf_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    note: str = ""


def _claude(api_key=None):
    import anthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to run this comparison.")
    return anthropic.Anthropic(api_key=key)


def _rows_text(rows: list[dict]) -> str:
    """How search results are handed to Claude: chunk number, section, text."""
    if not rows:
        return "(no results)"
    return "\n\n".join(f"[chunk {r.get('chunk_index')}] {r.get('content', '')}" for r in rows)


def _short(s: str, n: int = 700) -> str:
    return s if len(s) <= n else s[:n] + " …"


def _count_usage(run: Run, msg):
    run.llm_turns += 1
    u = getattr(msg, "usage", None)
    run.input_tokens += int(getattr(u, "input_tokens", 0) or 0)
    run.output_tokens += int(getattr(u, "output_tokens", 0) or 0)


# ---------------------------------------------------------------------------
# Classic RAG
# ---------------------------------------------------------------------------
def classic(tpuf, doc_id: str, strategy: str, question: str, top_k: int = 3, client=None) -> Run:
    run = Run("classic", question)
    t0 = time.time()
    run.steps.append(Step("user", "Question arrives", sent=question,
                          why="Classic RAG always searches with the question exactly as typed."))

    r = search.search(tpuf, doc_id, strategy, question, "hybrid", top_k)
    run.tpuf_queries += 1
    run.tpuf_ms += r["ms"]
    found = "\n".join(f"chunk {x.get('chunk_index')}  score {x.get('score')}  {x.get('section')}" for x in r["rows"])
    run.steps.append(Step("app → turbopuffer", "One hybrid search with the whole question", sent=r["code"],
                          received=found, ms=r["ms"],
                          why=f"The pipeline takes the top {top_k}, whatever they are. It can't tell whether "
                              "every part of the question was covered."))

    passages = "\n\n".join(f"[{i}] {x.get('content', '')}" for i, x in enumerate(r["rows"], 1))
    user_msg = f"Passages:\n{passages}\n\nQuestion: {question}"
    client = client or _claude()
    msg = client.messages.create(model=search.CLAUDE_MODEL, max_tokens=500, system=search.SYSTEM,
                                 messages=[{"role": "user", "content": user_msg}])
    _count_usage(run, msg)
    run.answer = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    run.steps.append(Step("app → Claude", "Chunks pasted into one prompt",
                          sent=f"system: {search.SYSTEM}\n\nuser: {_short(user_msg, 1200)}",
                          received=f"stop_reason: {msg.stop_reason}\n\n{run.answer}",
                          why="That's the only LLM call. Whatever the one search returned is all Claude ever sees."))
    run.seconds = round(time.time() - t0, 2)
    return run


# ---------------------------------------------------------------------------
# Agentic RAG
# ---------------------------------------------------------------------------
def _run_tool(name: str, args: dict, ctx: dict, run: Run) -> tuple[str, Step]:
    """Execute one tool call. Returns (text for Claude, trace step)."""
    tpuf, doc_id, strategy, md = ctx["tpuf"], ctx["doc_id"], ctx["strategy"], ctx["md"]
    if name == "sections":
        text = "\n".join(s.path for s in chunk.split_sections(md))
        return text, Step("tool (local)", "sections: table of contents",
                          sent="Reads the parsed Markdown headings locally. No turbopuffer call.", received=text)
    if name == "search_docs":
        mode = args.get("mode") or "hybrid"
        if mode not in ("hybrid", "vector", "keyword"):
            mode = "hybrid"
        k = max(1, min(int(args.get("k") or 3), 5))
        r = search.search(tpuf, doc_id, strategy, args.get("query", ""), mode, k, section=args.get("section"))
        run.tpuf_queries += 1
        run.tpuf_ms += r["ms"]
        code = r["code"]
        if args.get("section"):
            code += f"\n# plus filters=(\"section\", \"Glob\", \"{args['section']}*\") on each query"
        text = _rows_text(r["rows"])
        sec = f" · {args['section']}" if args.get("section") else ""
        return text, Step("app → turbopuffer", f"search_docs · {mode}{sec}", sent=code,
                          received=_short(text), ms=r["ms"])
    if name == "read_around":
        rows = search.expand(tpuf, doc_id, strategy, int(args.get("chunk_index", 0)))
        run.tpuf_queries += 1
        text = _rows_text(rows)
        return text, Step("app → turbopuffer", f"read_around · chunk {args.get('chunk_index')}",
                          sent=f"filters=(\"chunk_index\", \"In\", [neighbours]), rank_by=(\"chunk_index\", \"asc\")",
                          received=_short(text))
    return f"Unknown tool {name}", Step("tool (local)", f"unknown tool {name}")


def agent(tpuf, doc_id: str, strategy: str, question: str, md: str, client=None,
          max_rounds: int = MAX_TOOL_ROUNDS) -> Run:
    run = Run("agent", question)
    t0 = time.time()
    client = client or _claude()
    ctx = {"tpuf": tpuf, "doc_id": doc_id, "strategy": strategy, "md": md}
    messages: list = [{"role": "user", "content": question}]
    run.steps.append(Step("app → Claude", "Question plus tool definitions, no chunks",
                          sent="tools: " + ", ".join(t["name"] for t in TOOLS) + f"\n\nuser: {question}",
                          why="Claude gets tools instead of results, so it decides what to search for."))

    rounds = 0
    while True:
        force_answer = rounds >= max_rounds
        kwargs = {"tool_choice": {"type": "none"}} if force_answer else {}
        msg = client.messages.create(model=search.CLAUDE_MODEL, max_tokens=800, system=AGENT_SYSTEM,
                                     tools=TOOLS, messages=messages, **kwargs)
        _count_usage(run, msg)
        said = " ".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        calls = [b for b in msg.content if getattr(b, "type", "") == "tool_use"]

        if msg.stop_reason != "tool_use" or not calls:
            run.answer = said
            if force_answer:
                run.note = f"Hit the {max_rounds}-round cap, so Claude was told to answer with what it had."
            run.steps.append(Step("Claude → app", "Final answer", received=f"stop_reason: {msg.stop_reason}\n\n{said}",
                                  why="end_turn ends the loop: Claude judged every part covered (or ran out of rounds)."))
            break

        listed = "\n".join(f"{c.name}({json.dumps(c.input)})" for c in calls)
        run.steps.append(Step("Claude → app", f"Asks for {len(calls)} tool call{'s' if len(calls) > 1 else ''}",
                              received=f"stop_reason: tool_use\n\n{listed}",
                              why=said or "Claude didn't explain this call."))
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for c in calls:
            text, step = _run_tool(c.name, dict(c.input or {}), ctx, run)
            step.why = step.why or "The result goes back to Claude as a tool_result matched to this call's id."
            run.steps.append(step)
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": text})
        messages.append({"role": "user", "content": results})
        run.steps.append(Step("app → Claude", f"Returns {len(results)} tool result{'s' if len(results) > 1 else ''}",
                              sent=f"The whole conversation so far ({len(messages)} messages) plus the new results. "
                                   "The API is stateless, so every turn resends everything.",
                              why="Claude decides again: search more, read around a chunk, or answer."))
        rounds += 1

    run.seconds = round(time.time() - t0, 2)
    return run
