# merchant-agent

Standalone merchant-agent repository.

This repository was seeded from `interactive-decision-support-system/idss-backend`
and is now the implementation home for the merchant agent backend and future
merchant admin UI.

Initial scope:

- FastAPI merchant/admin/search backend in `apps/backend`
- enrichment and ingestion pipeline
- per-merchant catalog isolation
- contract package placeholder for generated OpenAPI TypeScript types

Environment:

- Committed template: `apps/backend/.env.example`
- Local secrets file: `apps/backend/.env` (git-ignored)
- `DATABASE_URL` should point at the live Supabase/Postgres connection; the
  backend does not fall back to a local database.

Out of scope:

- legacy IDSS paper artifacts
- legacy `/chat` surface
- vehicle domain
- OpenClaw/UCP/ACP compatibility layers
- `idss-web`, which remains a separate repo
