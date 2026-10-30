"""
Permission-aware RAG: the same question, four people, different answers.

Hobby project note: a teaching demo with made-up documents and people.
Harbor Lane Coffee Roasters, its staff, pay bands and plans are all fictional.

The idea
    Each document carries a "groups" attribute listing who may read it.
    Each user belongs to some groups. A search should only ever consider
    documents whose groups overlap the user's. In turbopuffer that's one
    filter on the query:  ("groups", "ContainsAny", user_groups)

Three ways to enforce it, so the demo can show why the first is right:

  native      Filter inside turbopuffer. Only allowed documents are ranked,
              so the user gets a full top-k of things they may see.
  post        Search everything, then drop disallowed results in the app.
              Nothing leaks, but restricted documents still take up top-k
              slots, so the user gets fewer (sometimes zero) results.
              turbopuffer's "native filtering" post explains why
              post-filtering hurts recall at scale.
  none        No filter at all. Restricted documents reach the user and the
              LLM. This is the bug you're demoing against.

One namespace holds every document here. For separate customers you'd more
often give each tenant its own namespace; groups filters are for access
rules *within* a tenant.
"""

from __future__ import annotations

import time

from . import search

NAMESPACE = f"{search.PREFIX}-permissions"
ATTRS = ["title", "content", "groups"]

USERS = {
    "Sam · barista": ["everyone"],
    "Priya · roastery manager": ["everyone", "managers"],
    "Dana · finance lead": ["everyone", "managers", "finance"],
    "Lee · people team": ["everyone", "people-team"],
}

# (id, title, groups, text). All fictional.
DOCS = [
    ("pay-schedule", "Pay schedule", ["everyone"],
     "Everyone is paid every other Friday by direct deposit. Pay reviews happen once a year in March. "
     "Ask your manager if you have questions about your own pay."),
    ("reporting-concerns", "Reporting a concern", ["everyone"],
     "If you experience or see harassment, discrimination or anything unsafe, tell your manager or email "
     "people-team@harborlane.example. You can also use the anonymous hotline. Retaliation is never tolerated."),
    ("cafe-locations", "Our cafes", ["everyone"],
     "Harbor Lane runs three cafes: Dock Street, Mill Yard and the roastery cafe on Pier Road. "
     "Opening hours are posted on the staff noticeboard in each cafe."),
    ("expense-approvals", "Expense approvals", ["everyone"],
     "Expenses under $500 need your manager's approval. Expenses of $500 or more also need the finance lead."),
    ("salary-bands", "Salary bands 2026 (managers only)", ["managers"],
     "Barista: $19 to $23 per hour. Shift lead: $23 to $27 per hour. Roastery supervisor: $68,000 to $82,000. "
     "Cafe manager: $72,000 to $88,000. The 2027 pay review budget is 4% of payroll."),
    ("performance-plans", "Managing underperformance (managers only)", ["managers"],
     "Start with a documented conversation. If nothing changes in 30 days, open a 60-day performance plan "
     "with the people team. Never discuss a plan with other staff."),
    ("restructuring-draft", "Q1 2027 restructuring plan (draft, confidential)", ["managers", "finance"],
     "Draft: close the Dock Street cafe at the end of March 2027 because the lease cost doubled. Six roles are "
     "affected. Offer every affected person a transfer to Mill Yard or Pier Road before any redundancy."),
    ("q3-results", "Q3 2026 results (finance only)", ["finance"],
     "Q3 revenue was $4.2 million, up 11% on last year. Gross margin fell to 58% because green coffee prices rose. "
     "Wholesale grew fastest; Dock Street was the only cafe that lost money."),
    ("bank-changes", "Supplier bank detail changes (finance only)", ["finance"],
     "Never change a supplier's bank details from an email. Call the supplier on the number already on file, "
     "and get a second approver in finance before saving the change."),
    ("investigations", "Investigation procedure (people team only)", ["people-team"],
     "When a concern is reported, open a case within two business days, interview the people involved "
     "separately, and keep notes in the case system only. Outcomes are shared only with those who need to know."),
]

# Questions that land differently depending on who asks.
EXAMPLES = [
    "What's the pay range for a roastery supervisor?",
    "Are any of our cafes closing?",
    "What happens after someone reports harassment?",
    "How did the company do financially last quarter?",
    "Who approves a $700 expense?",
]

POLICIES = {
    "native": "Filter in turbopuffer (correct)",
    "post": "Filter in the app after search",
    "none": "No filter (unsafe)",
}


def rows() -> list[dict]:
    return [{"id": i, "title": t, "content": f"{t}\n{text}", "groups": g} for i, t, g, text in DOCS]


def load(tpuf) -> dict:
    ns = tpuf.namespace(NAMESPACE)
    try:
        ns.delete_all()
    except Exception:
        pass
    t0 = time.time()
    ns.write(upsert_rows=rows(), distance_metric="cosine_distance", schema=search.SCHEMA)
    return {"namespace": NAMESPACE, "documents": len(DOCS), "seconds": round(time.time() - t0, 2)}


def allowed(doc_groups, user_groups) -> bool:
    return bool(set(doc_groups or []) & set(user_groups))


def search_as(tpuf, user: str, question: str, policy: str = "native", top_k: int = 3) -> dict:
    """
    Hybrid search as one user under one enforcement policy.
    Each returned row gets "allowed" (may this user see it?) so the UI can
    highlight leaks. Also reports how many results the post-filter dropped.
    """
    groups = USERS[user]
    ns = tpuf.namespace(NAMESPACE)
    base = {"top_k": top_k * 3, "include_attributes": ATTRS}
    if policy == "native":
        base["filters"] = ("groups", "ContainsAny", groups)
    queries = [dict(base, rank_by=("content", "ANN", ("Embed", question))),
               dict(base, rank_by=("content", "BM25", question))]
    t0 = time.time()
    res = ns.multi_query(queries=queries, rerank_by=("RRF",))
    ms = round((time.time() - t0) * 1000)
    found = [search._row(r) for r in res.results[0].rows][:top_k]
    dropped = 0
    if policy == "post":
        kept = [r for r in found if allowed(r.get("groups"), groups)]
        dropped = len(found) - len(kept)
        found = kept
    for r in found:
        r["allowed"] = allowed(r.get("groups"), groups)
    code = ('ns.multi_query(queries=[\n'
            f'    {{"rank_by": ("content", "ANN", ("Embed", q)), "top_k": {top_k * 3}'
            + (f', "filters": ("groups", "ContainsAny", {groups!r})' if policy == "native" else "") + '},\n'
            f'    {{"rank_by": ("content", "BM25", q), "top_k": {top_k * 3}'
            + (f', "filters": ("groups", "ContainsAny", {groups!r})' if policy == "native" else "") + '},\n'
            '], rerank_by=("RRF",))')
    return {"rows": found, "ms": ms, "dropped": dropped, "code": code, "groups": groups}
