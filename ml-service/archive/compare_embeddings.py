# ARCHIVED — not part of the active build.
#
# Standalone benchmarking script from TEN-4. Ran (anchor, related, unrelated)
# triplets through intfloat/multilingual-e5-base and Snowflake/snowflake-arctic-embed-l-v2.0
# to compare embedding quality and surface anisotropy issues with the old model.
# Findings are documented in docs/Ten4_Findings.md; the arctic model with
# category labels was adopted as a result and now lives in ml-service/main.py.

"""
compare_embeddings.py

Side-by-side anisotropy + similarity comparison between:
  1. intfloat/multilingual-e5-base   (what Ten currently uses)
  2. Snowflake/snowflake-arctic-embed-l-v2.0   (candidate replacement)

This does NOT touch validate_embeddings.py or anything else in ml-service.
It's a standalone script — drop it anywhere in ml-service/ and run it.

Requires: torch, transformers (already in your requirements.txt)

Usage:
    python compare_embeddings.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Test triplets — (category, anchor, related, unrelated). For each anchor we
# compute both similarities and report the GAP between them (related score
# minus unrelated score). A small gap is the anisotropy symptom: the model
# can't tell related from unrelated apart, even if the raw scores look
# reasonable in isolation.
#
# The first run (Eminem, Terminator, original Radiohead) showed one category
# behaving worse on the new model: band name vs. a generic noun (as opposed
# to band name vs. an actual album name, e.g. Eminem/Slim Shady). This batch
# adds five more band/album triplets to check whether that's a real pattern
# in this model or just noise from one pair.
# ---------------------------------------------------------------------------
# Context labels per category — mirrors what your real pipeline would have
# available (a list's category/type), unlike the bare strings validate_embeddings.py
# currently embeds. Anchor and related both get labeled; unrelated stays bare,
# since it's the control — nothing to disambiguate there.
CATEGORY_LABELS = {
    "rapper/album": ("rapper", "album"),
    "movie/actor":  ("movie", "actor"),
    "band/album":   ("band", "album"),
}

TEST_TRIPLETS = [
    ("rapper/album", "Eminem",          "Slim Shady",                  "cooking pan"),
    ("movie/actor",  "Terminator movie","Arnold Schwarzenegger movie", "banana bread recipe"),
    ("band/album",   "Radiohead",       "OK Computer",                 "mosquito"),
    ("band/album",   "Nirvana",         "Nevermind",                   "lawn mower"),
    ("band/album",   "The Beatles",     "Abbey Road",                  "dish soap"),
    ("band/album",   "Portishead",      "Dummy",                       "tax form"),
    ("band/album",   "Tool",            "Lateralus",                   "birthday cake"),
    ("band/album",   "Wu-Tang Clan",    "Enter the Wu-Tang",           "grocery store"),
]

# Set to True to prepend category context (e.g. "band: Tool", "album: Lateralus")
# instead of embedding bare strings. Compare both runs to see whether context
# fixes the Radiohead/Tool-style ambiguous-string failures.
USE_CONTEXT = True

# A larger, more diverse sample used only to estimate each model's mean
# embedding vector norm — this is the actual anisotropy number, computed
# on your own hardware rather than taken from a paper.
NORM_SAMPLE_TEXTS = [
    "Eminem", "Slim Shady", "cooking pan", "Terminator movie",
    "Arnold Schwarzenegger movie", "banana bread recipe", "Radiohead",
    "OK Computer", "milk", "hiking boots", "jazz trio",
    "Seattle rain", "cat named Genji", "linear algebra", "Wu-Tang Clan",
    "espresso machine", "urban exploration", "GitHub pull request",
    "Portland Oregon", "vintage synthesizer", "Nirvana", "Nevermind",
    "lawn mower", "The Beatles", "Abbey Road", "dish soap", "Portishead",
    "Dummy", "tax form", "Tool", "Lateralus", "Christmas Tree",
    "Enter the Wu-Tang", "grocery store",
]


# ---------------------------------------------------------------------------
# Model 1: intfloat/multilingual-e5-base — mean pooling + attention mask
# (this mirrors what's already in validate_embeddings.py)
# ---------------------------------------------------------------------------
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Model:
    name = "intfloat/multilingual-e5-base"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name)
        self.model.eval()

    def embed(self, texts, is_query=True):
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        batch = self.tokenizer(
            prefixed, max_length=512, padding=True,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# ---------------------------------------------------------------------------
# Model 2: Snowflake/snowflake-arctic-embed-l-v2.0 — CLS token pooling,
# query prefix only (documents get no prefix). This follows Snowflake's
# documented usage pattern, not the E5 pattern — don't copy-paste one
# model's calling convention onto the other.
# ---------------------------------------------------------------------------
class ArcticModel:
    name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    query_prefix = "query: "

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name, add_pooling_layer=False)
        self.model.eval()

    def embed(self, texts, is_query=True):
        prefixed = [self.query_prefix + t for t in texts] if is_query else list(texts)
        batch = self.tokenizer(
            prefixed, max_length=8192, padding=True,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = outputs[0][:, 0]  # CLS token, not mean pooling
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def mean_vector_norm(model):
    """The actual anisotropy proxy from the paper: embed a diverse sample,
    average the embeddings, report the norm of that average. A large
    narrow-cone collapse shows up as a mean vector with a large norm,
    since unrelated texts still point in roughly the same direction."""
    embeddings = model.embed(NORM_SAMPLE_TEXTS, is_query=False)
    mean_vec = embeddings.mean(dim=0)
    return torch.norm(mean_vec).item()


def run_pairs(model, label):
    print(f"\n--- {label} ({model.name}) [context={'ON' if USE_CONTEXT else 'OFF'}] ---")
    gaps_by_category = {}
    for category, anchor, related, unrelated in TEST_TRIPLETS:
        if USE_CONTEXT and category in CATEGORY_LABELS:
            anchor_label, related_label = CATEGORY_LABELS[category]
            anchor_text = f"{anchor_label}: {anchor}"
            related_text = f"{related_label}: {related}"
        else:
            anchor_text = anchor
            related_text = related

        emb_anchor = model.embed([anchor_text], is_query=False)[0]
        emb_related = model.embed([related_text], is_query=False)[0]
        emb_unrelated = model.embed([unrelated], is_query=False)[0]

        sim_related = cosine_similarity(emb_anchor, emb_related)
        sim_unrelated = cosine_similarity(emb_anchor, emb_unrelated)
        gap = sim_related - sim_unrelated

        print(f"  [{category}] {anchor_text}")
        print(f"      related:   {related_text:30s} {sim_related:.4f}")
        print(f"      unrelated: {unrelated:30s} {sim_unrelated:.4f}")
        print(f"      gap:       {gap:+.4f}")

        gaps_by_category.setdefault(category, []).append(gap)

    print("\n  -- average gap by category (bigger = better separation) --")
    for category, gaps in gaps_by_category.items():
        avg_gap = sum(gaps) / len(gaps)
        flag = "  <-- weak separation" if avg_gap < 0.05 else ""
        print(f"      {category:15s} avg gap: {avg_gap:+.4f}{flag}")

    norm = mean_vector_norm(model)
    print(f"\n  mean vector norm (lower = less anisotropic): {norm:.4f}")
    return norm


if __name__ == "__main__":
    print("Loading intfloat/multilingual-e5-base ...")
    e5 = E5Model()
    e5_norm = run_pairs(e5, "CURRENT")

    print("\nLoading Snowflake/snowflake-arctic-embed-l-v2.0 ...")
    arctic = ArcticModel()
    arctic_norm = run_pairs(arctic, "CANDIDATE")

    print("\n=== Side by side ===")
    print(f"  {'model':45s} mean vector norm")
    print(f"  {e5.name:45s} {e5_norm:.4f}")
    print(f"  {arctic.name:45s} {arctic_norm:.4f}")
    print(
        "\n  (Lower norm = more isotropic. This is measured on your own "
        "sample, not copied from the paper — compare it to that as a "
        "sanity check, not as ground truth.)"
    )