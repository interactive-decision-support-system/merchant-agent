# merchant-agent

Merchant-agent is an in-progress backend for merchant-owned shopping agents.
It exposes merchant catalog ingestion, product search, enrichment, knowledge
graph projection, and admin/inspection workflows around a live merchant data
plane.

## Project Status

This repository is early and backend-first. The current focus is the
merchant-agent API and enrichment pipeline; a dedicated merchant admin UI can
build on the existing API and inspector surfaces.

## Repository Layout

```text
.
|-- apps/
|   `-- backend/
|       |-- merchant_agent/       # FastAPI app, search, ingestion, enrichment
|       |-- migrations/           # Merchant schema and catalog migrations
|       |-- scripts/              # Operational scripts and inspector
|       |-- tests/                # Backend test suite
|       |-- .env.example          # Committed environment template
|       `-- pyproject.toml
|-- packages/
|   `-- contract/                 # Placeholder for generated API contracts
|-- justfile
|-- package.json
|-- pnpm-workspace.yaml
`-- turbo.json
```

## Environment

The backend expects real service connections rather than local stand-ins.
Create `apps/backend/.env` from the committed template and fill the relevant
values:

```bash
cp apps/backend/.env.example apps/backend/.env
```

At minimum, `DATABASE_URL` should point at the live Supabase/Postgres
connection. The template also covers Supabase REST, OpenAI, Langfuse,
Upstash/Redis, Neo4j, tracing, and runtime toggles.

`apps/backend/.env` is git-ignored and must not be committed.

## Backend

Install backend dependencies from the backend app directory:

```bash
cd apps/backend
python -m pip install -e .
```

Run the API:

```bash
just dev
```

By default this starts the FastAPI app at `http://127.0.0.1:8000`.

Useful endpoints:

```text
GET  /health
GET  /merchant
POST /merchant
POST /merchant/search
POST /merchant/{merchant_id}/search
GET  /merchant/{merchant_id}/health
```

## Inspector

The current inspection surface is a Streamlit app for enrichment traces and
catalog/debug workflows:

```bash
cd apps/backend
streamlit run scripts/enrichment_inspector.py
```

## Validation

Run the backend tests:

```bash
just test
```

Compile-check the backend package:

```bash
just compile
```

Run focused smoke checks:

```bash
cd apps/backend
PYTHONPATH=. python -m pytest \
  tests/test_default_merchant_collapse.py \
  tests/test_merchant_registry.py \
  tests/test_merchant_admin_http.py \
  tests/test_enrichment_inspector.py
```

## Roadmap

- Add production-grade auth and tenant isolation for merchant/admin routes.
- Promote the inspector/admin workflows into a dedicated merchant admin UI.
- Tighten enrichment evaluation, traceability, and Langfuse run mapping.
- Remove legacy compatibility code paths that are no longer used.
- Generate and publish API contracts for downstream clients.
