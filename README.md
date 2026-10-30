# Parse → chunk → search

> **A personal hobby project.** I built this to learn how document retrieval pipelines work end to end. It isn't affiliated with or endorsed by turbopuffer, it isn't production code, and the sample company and handbook are fictional.

Most "chat with your docs" failures happen before the database is involved: the PDF was parsed badly, or the chunks were cut in the wrong places. This small Streamlit app makes those upstream steps visible. It walks one document through four stages and lets you compare choices side by side:

| Tab | What it shows |
|---|---|
| **1 · Parse** | Naive PDF text extraction next to a structured parser's Markdown. Headers and footers removed, tables kept, headings recognised. |
| **2 · Chunk** | The same Markdown split two ways, colour-coded: **structure-aware recursive** (cuts at headings, then paragraphs, then sentences) and **semantic** (embeds every sentence and cuts where the topic shifts). A chart shows each similarity score and why it cut. |
| **3 · Store** | Both sets of chunks written to [turbopuffer](https://turbopuffer.com), one namespace per strategy. turbopuffer embeds the text itself (native embedding) and builds a BM25 index. |
| **4 · Search** | Vector, keyword (BM25), hybrid and reranked search side by side, or the same mode across both chunking strategies. Shows the exact SDK call, fetches neighbouring chunks ("small to big"), and optionally has Claude write an answer with citations. |

There's also `walkthrough.ipynb`, which runs the same pipeline one step at a time.

## How it works

```
PDF ──parse──▶ Markdown ──chunk──▶ chunks ──write──▶ turbopuffer ──query──▶ results ──▶ Claude (optional)
     pypdf /             Chonkie recursive      native embedding     vector / BM25 /
     PyMuPDF4LLM /       or hand-written        + BM25 index         hybrid / rerank
     Docling             semantic chunker
```

- `pipeline/parse.py`: three parsers to compare, plus a cleanup step that drops lines repeated on most pages
- `pipeline/chunk.py`: recursive splitting via [Chonkie](https://github.com/chonkie-inc/chonkie), and a ~40-line semantic chunker written by hand so every decision is visible. Optional heading-path prefix ("Time off > Parental leave") so chunks keep their context.
- `pipeline/search.py`: writes, four search modes, neighbour expansion, and the Claude answer
- `scripts/make_sample_pdf.py`: builds the fictional handbook, deliberately messy (running headers, a two-column page, a table)

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # then set APP_PASSWORD and your keys
set -a; source .env; set +a   # load them into this shell
streamlit run app.py
```

The app asks for a password first: set `APP_PASSWORD` in `.env` (it won't open without one, since it writes to your turbopuffer account). Parsing and chunking work without any API keys. Writing and searching need `TURBOPUFFER_API_KEY`; the written answer needs `ANTHROPIC_API_KEY`. The semantic chunker downloads a small (~130 MB) embedding model from Hugging Face the first time it runs.

**Better parsing, optional:** `pip install -r requirements-docling.txt` adds IBM's Docling as a third parser. It's noticeably better on messy layouts but pulls in PyTorch, so it's left out of the default install and won't fit on free hosting.

**Deploying (optional):** on Streamlit Community Cloud, create an app from this repo with `app.py` as the main file and add the keys from `.streamlit/secrets.toml.example` under Advanced settings → Secrets, including `APP_PASSWORD`. The password is a single shared one kept for the browser session: fine for a hobby demo, not a real login system.

## Demo script

Questions that each show something specific. All answers are in the sample handbook.

| Question | What to look for |
|---|---|
| How many weeks of parental leave does a secondary caregiver in the office get? | The answer (8 weeks) lives in a table. Compare the raw and parsed text on tab 1: raw extraction scatters the table into loose values. |
| What is policy HL-SEC-07? | An exact code. **Keyword** search finds it immediately; **vector** search may not. |
| I left my work computer on the train. Who do I tell? | No words in common with the answer ("laptop or phone is lost or stolen"). **Vector** search wins on meaning. |
| Can I carry unused vacation days into next year? | The handbook says "roll over". Another win for meaning over keywords; hybrid gets it either way. |
| Who has to sign off on a $700 expense? | Needs reasoning over a threshold ($500). Retrieval finds the right chunk; the Claude answer applies the rule. |
| How much can I spend on dinner during a work trip to London? | "London" isn't in the text, "United Kingdom" is. Compare strategies: semantic chunking may split the meals section in two, so check which chunk comes back. |
| How often is the espresso machine cleaned? | The parser misreads the two-column page and files this paragraph under Travel. A good reminder that parsing errors look like search errors. |

Then try the toggles on tab 2: turn off **Prepend heading path**, re-write on tab 3, and see which questions get worse.

## Notes

- Everything is written to namespaces starting with `tpragdemo-`. The sidebar button deletes them.
- Defaults match the turbopuffer setup I used elsewhere: region `gcp-us-central1`, embedding model `nvidia/nemotron-3-embed-8b`. Override with the environment variables in `.env.example`. Native embedding and reranking depend on your turbopuffer plan.
- Chunk sizes are in characters, not tokens, to keep things simple (roughly 4 characters per token in English).
- Costs: parsing and chunking are free and local. turbopuffer usage for a document this size is tiny, but turbopuffer has a monthly minimum on paid plans. Claude answers cost a fraction of a cent each.

## License

MIT. See [LICENSE](LICENSE).
