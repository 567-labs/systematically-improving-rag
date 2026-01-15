# Repository Guidelines

## Project Structure & Module Organization
- `latest/`: Current course code and the WildChat case study (`latest/case_study/{core,pipelines}`).
- `cohort_1/`, `cohort_2/`: Earlier cohort materials kept for reference.
- `docs/`: MkDocs book sources; site config in `mkdocs.yml`.
- `docs/workshops/`: Chapter content `chapterN.md` and subparts `chapterN-M.md`; entrypoint is `docs/workshops/index.md`.
- `docs/slides/`: Slide decks `chapterN-slides.md` for workshop chapters.
- `md/`: Markdown exports of notebooks; images in `images/`.
- `scripts/`, `build_book.sh`: Utilities for diagrams and building the PDF/ebook.

## Build, Test, and Development Commands
- Install deps: `uv install` (recommended) or `pip install -e .`.
- Lint/fix: `uv run ruff check --fix --unsafe-fixes .`.
- Format: `uv run ruff format .`.
- Tests: `uv run pytest -q` (supports `pytest-asyncio`).
- Docs (local): `mkdocs serve`  •  Docs (build): `mkdocs build`.
- Book (optional): `bash build_book.sh` (requires `pandoc`; optional `tectonic` + `@mermaid-js/mermaid-cli`).

## Docs & Workshops Authoring
- Front matter: include `title`, `description`, optional `authors`, `date`, `tags` (see `docs/workshops/chapter0.md`).
- Naming: `chapter4-2.md` pattern for multi-part chapters; keep headings starting with a single `#`.
- Admonitions: use MkDocs blocks like `!!! info` and `!!! success`; keep copy concise and actionable.
- Diagrams: use fenced `mermaid` blocks; book build renders via Mermaid CLI when available.
- Linking: link chapters relatively as in `docs/workshops/index.md`; avoid hard-coded site URLs.
- Section order: place `### Key Insight` before `## Learning Objectives`. To fix inconsistencies, run `python scripts/normalize_workshops.py`.

## Coding Style & Naming Conventions
- Python 3.11 required (`requires-python >=3.11,<3.12`). Use type hints throughout.
- Indentation: 4 spaces; max line length per Ruff formatter.
- Naming: `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Imports: standard → third‑party → local; prefer explicit exports.
- Notebooks: keep outputs minimal; prefer moving reusable logic into importable modules under `latest/case_study/`.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` for async code.
- Location: create `tests/` at repo root; name files `test_*.py`.
- Conventions: one behavior per test; use fixtures for external services; mock APIs.
- Examples: `uv run pytest -q`, with coverage `uv run pytest --cov latest/case_study`.

## Commit & Pull Request Guidelines
- Commits: imperative, present tense, concise (e.g., `fix: handle empty queries`).
- PRs: include problem statement, summary of changes, and before/after screenshots for docs/UI.
- Link issues when applicable; note follow‑ups and known limitations.
- Checklist before opening PR: run Ruff (check + format), run tests, verify `mkdocs build` passes.

## Security & Configuration Tips
- Do not commit secrets. Store API keys in `.env` and load via `python-dotenv`.
- Common vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `LOGFIRE_*`.
- Prefer configuration via environment variables over hard‑coding.
