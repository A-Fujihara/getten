# Contributing

This is currently a solo portfolio project with no active external contributors, but this file exists so the conventions are written down from the start rather than reconstructed later.

## Project structure

```
getten/
├── ml-service/     # FastAPI embedding + correlation service (Python)
├── product-api/    # Node/Express API
├── frontend/       # React client
└── docs/           # Spec and design documents
```

Each service is self-contained with its own dependency file (`requirements.txt` for `ml-service`, `package.json` for `product-api`/`frontend`). Don't share dependencies across services — the ML service in particular pulls in heavy packages (PyTorch) that have no reason to touch the Node or frontend dependency trees.

## Setting up a local environment

See the "Getting started" section in `README.md` for the ML service. Instructions for `product-api` and `frontend` will be added once those services have real content.

## Commit conventions

- Write commit messages in the imperative mood ("add embedding validation script", not "added" or "adds")
- Keep commits scoped to one logical change — avoid bundling unrelated changes (e.g. a dependency bump and a new feature) into a single commit
- Reference the relevant part of the spec (`docs/`) in the commit body when a change implements or deviates from something documented there

## Branching

- `main` should stay in a working state
- Feature work happens on branches off `main`, named descriptively (e.g. `ml-service-fastapi-endpoint`, not `fix` or `update`)

## Code style

- Python: standard PEP 8 conventions. Docstrings on all functions (see `ml-service/main.py` for the expected format).
- No dependency should be added without a documented reason — this project deliberately favors raw implementations (e.g. manual PyTorch pooling/normalization over the `sentence-transformers` wrapper) where the added understanding is worth more than the convenience, per the project's goals in `docs/`.

## Environment variables and secrets

Never commit `.env` files or hardcoded credentials. `.env` is already excluded via `.gitignore` — if you add a new service that needs secrets, follow the same pattern rather than committing example values "temporarily."