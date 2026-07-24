import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

"""
    Combine per-token vectors into a single vector representing the whole
    phrase, using the attention mask to correctly ignore padding tokens.

    Args:
        last_hidden_states: The model's raw output — one vector per token.
        attention_mask: Tensor of 1s and 0s marking real tokens (1) vs.
            padding filler (0).

    Returns:
        Tensor: A single averaged vector, with padding excluded from the
            calculation.
    """
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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
    inputs = tokenizer(prefixed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
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