# ARCHIVED — not part of the active build.
#
# This was the first working proof that the embedding pipeline worked,
# using intfloat/multilingual-e5-base with no category labels. It's kept
# here to show that early thinking, not because it's still in use.
#
# The current implementation lives in ml-service/main.py: it uses
# Snowflake/snowflake-arctic-embed-l-v2.0 and prepends a category label
# to each text before embedding (see docs/Ten4_Findings.md for why).

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
model.eval()


"""
    Convert a piece of text into a single normalized vector representing
    its meaning.

    Args:
        text: The input string to embed.

    Returns:
        Tensor: A length-1 (L2-normalized) vector.
    """
def embed(text):
    prefixed_text = f"query: {text}"
    inputs = tokenizer(prefixed_text, return_tensors="pt", truncation=True, padding=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = outputs[0][:, 0]  # CLS token, not mean pooling
    normalized = F.normalize(pooled, p=2, dim=1)
    return normalized.squeeze()

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

pairs = [
    ("Eminem", "Slim Shady", "should be similar"),
    ("Eminem", "cooking pan", "should be dissimilar"),
    ("Terminator", "Arnold Schwarzenegger movie", "should be similar"),
]

for a, b, expectation in pairs:
    emb_a = embed(a)
    emb_b = embed(b)
    score = cosine_similarity(emb_a, emb_b)
    print(f"{a!r} vs {b!r} ({expectation}): {score:.4f}")