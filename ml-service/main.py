from fastapi import FastAPI, Depends
from pydantic import BaseModel
from functools import lru_cache
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"


@lru_cache
def get_embedder():
    """
    Returns (tokenizer, model), loading them from disk/network only on
    the first real call. lru_cache makes this safe under concurrent
    requests — the model loads exactly once, even if multiple requests
    hit /embed at the same time before it's finished loading.

    FastAPI can override this function in tests to hand back a fake
    tokenizer/model instead, so tests that don't care about the actual
    model never pay the cost of loading it.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


class EmbedRequest(BaseModel):
    text: str
    category: str


def average_pool(last_hidden_states, attention_mask):
    """
    Combine per-token vectors into a single vector representing the whole
    phrase, using the attention mask to correctly ignore padding tokens.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed(text, category, tokenizer, model):
    """
    Convert a piece of text into a single normalized vector representing
    its meaning. The category label is prepended so ambiguous terms
    (e.g. "Tool" the band vs. "tool" the object) embed distinctly,
    per the TEN-4 finding.
    """
    labeled_text = f"{category}: {text}"
    inputs = tokenizer(labeled_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
    normalized = F.normalize(pooled, p=2, dim=1)
    return normalized.squeeze()


@app.post("/embed")
def embed_endpoint(request: EmbedRequest, embedder=Depends(get_embedder)):
    tokenizer, model = embedder
    vector = embed(request.text, request.category, tokenizer, model)
    return {"embedding": vector.tolist()}