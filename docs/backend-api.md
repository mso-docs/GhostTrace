# Local engine API

Base URL `http://127.0.0.1:8000/api/v1`. Run `ghosttrace serve` and `ghosttrace worker` in separate terminals. No endpoint implicitly downloads weights. The built root `dist/` is served at `/` if present. Interactive contract documentation: `/docs`; generated schema: `/openapi.json`. Experiment request DTOs are strict Pydantic contracts shared with authoritative engine validation.

Every JSON response includes `schema_version` and `request_id` except framework-generated OpenAPI responses. List responses use `{items, next_cursor}` with limit 1–100. Fetch `/health` first and send its `mutation_token` in `X-GhostTrace-Token` on every POST. The token changes when the API restarts. Host and Origin are restricted to localhost and explicit development origins; same-origin requests also work on a custom local service port. Vite proxy uses `changeOrigin:true`. Actual streamed request bytes are bounded even without Content-Length: 5 MB generally, 2 GB for evidence ZIP uploads.

## Queued operations

`POST` preflight, experiment execution, analysis, export, import, and replay return HTTP 202 with `{id,job_id,kind,state:"queued",total_units}`. They enqueue work; they do not claim it completed. An explicitly running worker claims one job at a time. Poll `GET /jobs/{id}` and `/jobs/{id}/events?after=0`.

A completed evidence job exposes `result`:

```json
{"resource_type":"analysis","resource_id":"<uuid>","analysis_id":"<uuid>"}
```

Exports return `export_id`; imports return `import_id`, `experiment_id`, `analysis_id`, `raw_activations_available` and `recomputed`; replay returns `analysis_id` and verification. Failed jobs retain structured `error_code` and `error_message`. Analysis/export/replay accept an optional `Idempotency-Key`; experiment execution requires it. Reusing a key with another body returns 409. Cancellation of a bounded evidence operation takes effect before that operation; model execution cancels between inference units.

## Routes

- `GET /health`: API/worker status, offline flag and mutation token; never loads a model.
- `GET /models`: Qwen profile capabilities and validated local provenance, if available.
- `POST /preflights`: `{profile_id, revision, device, compute_dtype}`; revision must be 40 lowercase hex characters. Model must already exist in the local cache. Queueing does not download it.
- `POST /datasets/validate`: UTF-8 JSONL or JSON array, max 5 MB.
- `POST /experiments`: typed experiment config. Returns frozen manifest, SHA-256, estimated_units/bytes, ordered units and review/preflight warnings. Identical content returns the existing experiment.
- `GET /experiments` and `GET /experiments/{id}`: immutable protocol records.
- `POST /experiments/{id}/jobs`: `Idempotency-Key` and optional `{}`. Requires a validated model profile.
- `GET /jobs/{id}`; `GET /jobs/{id}/events?after=0`; `POST /jobs/{id}/cancel`; `POST /jobs/{id}/resume`. Resume creates an attempt and reuses checksum-verified completed units; it never replaces completed outputs.
- `GET /experiments/{id}/runs`: filters `scenario`, `condition`, `template`, `mode`, `status`.
- `GET /runs/{id}` and `/activations?layer_start=0&layer_end=1&token_start=0&token_end=2`: norms by default; `raw=true` requests a bounded float slice. Summary bundles return unavailable-capture errors for omitted arrays.
- `POST /experiments/{id}/analyses`: `{split:"development"|"holdout"|"all"}`. Holdout requires matching human-review and frozen preregistration receipts. `GET /analyses/{id}` returns L2 primary curves, cosine/norm-ratio secondary curves, scenario bootstrap intervals, lexical/length/truncation/NLL summaries, raw pairing provenance, ratings and missingness.
- `POST /exports`: `{analysis_id,include_activations:false}`. `GET /exports/{id}` returns completed metadata; `GET /exports/{id}/download` downloads the ZIP. Export streams arrays, includes Markdown/CSV/JSON evidence and ratings, and declares raw-array omission. Bounded jobs.jsonl/events.jsonl snapshots and attempts/<job UUID>/<unit hash>/record.json retain earlier failures and recovery history. Earlier-attempt raw arrays are explicitly omitted; latest canonical arrays remain optional.
- `POST /imports?verify_metrics=true`: raw `application/zip` bytes, max 2 GB compressed/expanded. Returns an import job; `GET /imports/{id}` exposes its verification record. No multipart form or arbitrary local paths.
- `POST /experiments/{id}/replay`: `{analysis_id?:"<uuid>",verify_metrics:true}`. Uses stored arrays and recorded settings; never imports Transformers, loads weights or accesses the network.
- `GET /experiments/{id}/rating-tasks?rater_id=<id>`: shared balanced cohort of up to 36 generated responses, rater-specific display ordering, no model/condition/template/activation metadata. Repeated requests return the same unfinished cohort. Stylistic wording may still reveal condition.
- `POST /ratings`: rubric `expressed-behavior-v1`, `run_unit_id`, issued `blinded_display_id`, `rater_id`, four strict integer 0–2 scores (`regret`, `responsibility`, `repair`, `justification_deflection`) plus refusal/malformed flags. Identical submissions return the original; changing an independent score returns 409. Originals and disagreements are retained.
- `GET /experiments/{id}/ratings`: saved independent ratings. `GET /experiments/{id}/rating-summary`: per-rater counts, same-unit/rubric paired agreement and quadratic weighted Cohen's kappa per dimension, retained disagreements, small-sample and one-rater limitations. Zero expected disagreement yields null kappa with a reason.

## Evidence safety and offline replay

Imports reject unknown files, path traversal, symlinks, encrypted or duplicate entries, oversized files/archives, checksum errors, inconsistent manifest/unit IDs, invalid schemas/arrays and numerical replay mismatches. NumPy headers are checked before allocation; arrays use `allow_pickle=False`. Raw arrays are independently recomputed in an isolated staging store before registration, within absolute tolerance 1e−6 and relative tolerance 1e−5. Failed imports expose no partial experiment in the application. Original records/hashes remain unchanged; SQLite stores relocated local array paths separately. Imported jobs, events, and attempt records are inert provenance files and never enter the active queue; re-export preserves that history.

Summary-only bundles preserve recorded metrics for viewing but explicitly cannot independently recompute them. Checksums establish integrity, not authorship or scientific validity. A dataset review or model preflight contained in an imported bundle never promotes it to a trusted local inference profile.

```bash
ghosttrace --data-dir artifacts/replay import evidence.zip
ghosttrace --data-dir artifacts/replay replay --experiment <uuid> --analysis <uuid>
python scripts/verify_export.py evidence.zip
```

Shared PCA, probe and patching remain optional unimplemented capabilities. Human review and independent human ratings must be performed by people; software support does not substitute those release gates.
