"""
The three stages of the demo, one module each:

  parse.py   PDF -> clean Markdown (and the naive raw-text version, for contrast)
  chunk.py   Markdown -> chunks, two ways: structure-aware recursive and semantic
  search.py  chunks -> turbopuffer, then vector / keyword / hybrid search

Hobby project: written to learn and to demo the ideas, not for production use.
"""
