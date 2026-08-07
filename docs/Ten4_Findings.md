# TEN-4: Anisotropy Investigation - Findings

**Status:** Investigation complete
**Ticket:** TEN-4
**Dependencies satisfied for:** TEN-5 (entity resolution threshold)

## Question

TEN-1 flagged a concern: `intfloat/multilingual-e5-base` produced similarity scores that were high across the board, even for unrelated pairs (Eminem/cooking pan scored 0.7357). This ticket exists to answer: is raw cosine similarity usable directly for entity resolution and correlation, or does it need a calibration step first?

## Method

Built `compare_embeddings.py`, a standalone script (does not modify `validate_embeddings.py`) that runs (anchor, related, unrelated) triplets through a model and reports:
- Similarity score for anchor vs. related
- Similarity score for anchor vs. unrelated
- **Gap** = related score - unrelated score (the actual signal a threshold would use)
- Mean vector norm across a diverse text sample, as an aggregate anisotropy measure (lower = more isotropic)

8 triplets were run (16 comparisons total) across three categories: rapper/album, movie/actor, band/album. Each was tested against two models, and against two embedding strategies (bare string vs. category-labeled string, e.g. `"Tool"` vs. `"band: Tool"`).

## Finding 1: The original model has an anisotropy problem

`intfloat/multilingual-e5-base`, bare strings: mean vector norm **0.9139**. Gaps across all 8 triplets ranged from +0.024 to +0.165, with one case slipping negative once a harder album name was tested. Every score, related or not, clustered in the 0.77–0.93 range. This confirms the TEN-1 concern: raw scores from this model are too compressed to threshold reliably.

## Finding 2: A candidate model reduces anisotropy substantially, but isn't automatically better

`Snowflake/snowflake-arctic-embed-l-v2.0`, bare strings: mean vector norm **0.5126**, roughly half. Most gaps widened significantly (up to +0.52). But bare-string testing surfaced a new failure mode: two triplets whose "related" term was an ambiguous phrase - "OK Computer" (reads as a sentence fragment) and "Tool" (a common noun) - produced near-zero or negative gaps, worse than the old model on those specific cases.

## Finding 3: The failure was a missing-context problem, not a model problem

Root-caused by testing multiple unrelated words against the same ambiguous anchors (garden hose, mosquito) - the gap stayed broken regardless of what the unrelated term was, which ruled out semantic overlap with the unrelated word as the cause. The actual cause: bare strings like "OK Computer" and "Tool" have no signal distinguishing them from ordinary English phrases.

Fix tested: prepend a category label derived from list context (`"band: Tool"`, `"album: OK Computer"`) instead of embedding bare item text. Result: every triplet cleared a wide, positive gap. Worst case went from **-0.0419 to +0.2229**. Band/album category average gap improved from 0.1953 (context off) to 0.3497 (context on).

## Score distribution summary

| Model | Context | Gap range | Category averages |
|---|---|---|---|
| e5-base | off | -0.037 to +0.165 | rapper/album +0.10–0.17, movie/actor +0.10–0.12, band/album +0.02–0.13 |
| arctic-embed-l-v2.0 | off | -0.064 to +0.519 | rapper/album +0.40, movie/actor +0.44, band/album -0.06 to +0.52 (unstable) |
| arctic-embed-l-v2.0 | **on** | **+0.223 to +0.499** | rapper/album +0.42, movie/actor +0.49, band/album +0.35 |

## Conclusion

Raw cosine similarity **is usable directly** for entity resolution and correlation - but only under two conditions, both now decisions rather than open questions:

1. **Model:** `Snowflake/snowflake-arctic-embed-l-v2.0` replaces `intfloat/multilingual-e5-base`. Confirmed lower anisotropy at the aggregate level (0.51 vs. 0.91 mean vector norm) and better category-level separation once context is applied.
2. **Context is required, not optional.** Items must be embedded with a category-derived label (e.g. `"band: X"`) rather than as bare strings. Without this, specific ambiguous item names produce false near-collapse regardless of model choice.

No additional calibration/normalization layer is needed on top of raw cosine similarity, provided both conditions above are met in the production embedding pipeline.

## Open item carried forward (not part of TEN-4)

Context labels require knowing an item's category, but list categories are free user text ("Top 10 Movies" vs. "Top 10 films"). This needs its own normalization step (embedding the category string against a canonical taxonomy) before context labels can be applied reliably in production. Logged as a future design item, not yet scoped as a ticket.

## Testing notes

This was an investigation ticket; the deliverable is this document and the data behind it, not a pass/fail test. `compare_embeddings.py` remains in the repo as a reusable comparison tool for any future model evaluation.