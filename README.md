# Ten

A social platform built around ranked Top 10 lists — on any topic, from any category.

## What makes this different

Most "ranking" apps compare you to one specific other person — a match score, a compatibility percentage. Ten deliberately avoids that mechanic entirely. There is no "you vs. one person" comparison feature.

Instead, Ten surfaces **aggregate correlation**: population-level patterns across everyone's lists. A statement like *"75% of people who ranked Adele also ranked Oppenheimer"* — never a claim about one specific pairing. This can cross categories entirely (rappers to movies) or stay within one.

Every list counts toward this aggregate math regardless of its visibility setting. Visibility controls what can ever be *displayed*, never what gets *computed*.

## Project status

Early development. This is a portfolio project — the goal is demonstrating range across ML, service architecture, async processing, and graph data modeling, not shipping a production app.

## Architecture

```
React Frontend → Node/Express (Product API) → FastAPI (ML Service, internal only)
                        |
                  Postgres (Supabase)      Redis + BullMQ      Neo4j (correlation graph)
```

The client never talks to the ML service directly — only through the Node API.

## Tech stack

- **Frontend:** React
- **Product API:** Node / Express
- **ML Service:** Python / FastAPI, internal-only, never exposed to the client directly
- **Embeddings:** PyTorch + HuggingFace Transformers, using `intfloat/multilingual-e5-base`, implemented with raw PyTorch (manual tokenization, pooling, normalization) rather than the `sentence-transformers` wrapper
- **Social graph data:** Postgres (Supabase)
- **Correlation graph:** Neo4j (deferred past MVP)
- **Async processing:** Redis + BullMQ

## Repository structure

```
getten/
├── ml-service/     # FastAPI embedding + correlation service
├── product-api/    # Node/Express API
├── frontend/       # React client
└── docs/           # Spec and design documents
```

## Getting started

### ML service

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 archive/validate_embeddings.py
```

`validate_embeddings.py` is the original standalone sanity check, it embeds a handful of test phrase pairs and prints their cosine similarity scores, to confirm the embedding pipeline behaves as expected before it's wired into the actual service.