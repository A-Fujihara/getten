import torch
from fastapi.testclient import TestClient
from main import app, get_embedder


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        # Pretend every input is 3 real tokens, no padding.
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        # 1 sequence, 3 tokens, 4-dim vectors (small, fast, arbitrary size)
        fake_hidden_states = torch.rand(1, 3, 4)
        return FakeModelOutput(fake_hidden_states)


def fake_get_embedder():
    """
    Stands in for the real get_embedder(). Returns fake objects instead
    of loading the actual arctic model, so these tests run instantly and
    never touch the network or disk.
    """
    return FakeTokenizer(), FakeModel()


app.dependency_overrides[get_embedder] = fake_get_embedder

client = TestClient(app)


def test_embed_rejects_missing_category():
    """
    A request missing the required 'category' field should be
    rejected by FastAPI's validation before embed() ever runs.
    """
    response = client.post("/embed", json={"text": "Tool"})
    assert response.status_code == 422


def test_embed_rejects_missing_text():
    """
    A request missing the required 'text' field should be
    rejected the same way.
    """
    response = client.post("/embed", json={"category": "band"})
    assert response.status_code == 422


def test_embed_rejects_wrong_type():
    """
    A request where 'category' is not a string should be rejected.
    """
    response = client.post("/embed", json={"text": "Tool", "category": 123})
    assert response.status_code == 422


def test_embed_accepts_valid_request():
    """
    A valid request should succeed and return an 'embedding' field
    that's a list of numbers. Uses the fake tokenizer/model, so this
    checks the endpoint's wiring, not the real model's output quality.
    """
    response = client.post("/embed", json={"text": "Tool", "category": "band"})
    assert response.status_code == 200
    body = response.json()
    assert "embedding" in body
    assert isinstance(body["embedding"], list)