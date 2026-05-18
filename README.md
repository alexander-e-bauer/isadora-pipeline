# Option Overlays Engine

FastAPI data pipeline + engine-side AI subagents for the Option Overlays multi-tenant SaaS.

> **v1 wedge shipped 2026-05-18.** This worktree (`engine-tenant-wt` on branch `tenant-integration`) is the active engine. The legacy `../engine/` directory on `master` is the pre-Chunk-2 pipeline, tagged `pre-chunk2-engine-snapshot`.
>
> See the **[North Star Spec](../NORTH_STAR_SPEC.md)** for the architecture, the **[v1 wedge plan](../docs/superpowers/plans/2026-05-15-v1-wedge.md)** for Chunk 2 + Chunk 4, and the **[demo walkthrough](../docs/demo/2026-05-15-v1-walkthrough.md)** to run the full stack.

## v1 wedge inventory

- `xyz/finazon_service/` — pre-existing equity market-data pipeline (Finazon, forecasts, embeddings → Pinecone). Untouched in v1 except for the tenant-Base + Polygon model registration changes.
- `xyz/polygon_service/` — Polygon Options data layer (Chunk 2.2): `OptionHistoricalEod`, `OptionChains`, `OptionIvSurface`, `OptionQuotes`. `xyz/polygon_service/options_client.py` is the raw `httpx` wrapper (no SDK).
- `xyz/tenant/` — tenant-schema mirror of `server-fastapi-wt/app/models/*`. Engine READS firms/users/clients/accounts/strategies/deployments; WRITES events via `xyz/tenant/events.py::emit_event` (byte-for-byte identical hash to the server).
- `xyz/agents/` — engine-side subagents:
  - `research.py` — RESEARCH (Polygon IV + DB documents + Claude) → emits `research.artifact`
  - `author.py` — AUTHOR (Claude + Strategy DSL JSON-Schema validation) → emits `strategy.draft`
  - `backtest.py` + `xyz/backtest/` — BACKTEST (daily-bar replay, deterministic content_hash) → emits `backtest.result`
  - `propose.py` — PROPOSE (deterministic; evaluates ACTIVE deployment trigger) → emits `ticket.proposed`
- `xyz/dsl/` — Strategy DSL JSON-Schema (`schema.py`) + validator (`validate.py`).
- `xyz/observability/` — `RequestIdMiddleware` + JSON log formatter; correlation id stamped into every emitted event payload as `_request_id` (stripped from the hash for determinism).

80 pytest tests. Run with `python -m pytest tests/ -q`.

## Endpoints

All bearer-authed via the `KEY` env var (mirrors `/run-pipeline`):

```
POST /agents/research       body: {firm_id, symbol?, brief?, actor_user_id?}
POST /agents/author         body: {firm_id, brief, target_account_ids?, actor_user_id?}
POST /agents/backtest       body: {strategy_id, strategy_version, firm_id, start_date, end_date, dsl}
POST /agents/propose        body: {firm_id, deployment_id, actor_user_id?}
POST /agents/validate-dsl   body: {dsl}
POST /run-pipeline          (legacy — Finazon equity pipeline)
GET  /health
GET  /monitoring/ticker/{symbol}
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py            # uvicorn on http://0.0.0.0:8900
```

OpenAPI docs at [http://localhost:8900/docs](http://localhost:8900/docs).

## Deployment

Google Cloud Run via `deploy.sh` (`gcloud builds submit` → `gcloud run deploy`). Uses `env.yaml` for env vars and attaches the Cloud SQL instance. Project: `trusty-spanner-446802-p2`, region `us-central1`, service `isadora-pipeline`. `cloudbuild.yaml` is the Cloud Build trigger equivalent.

## Schema parity

The tenant tables on this worktree mirror server's. Parity is enforced by:
- `tests/test_tenant_read.py::test_engine_tenant_schema_matches_server` (column-level metadata diff)
- `tests/test_tenant_read.py::test_engine_event_hash_matches_server_implementation` (cross-app hash function parity)
- `scripts/dump_tenant_schema.py` + `../scripts/check-schema-parity.sh` (workspace-level CI)
