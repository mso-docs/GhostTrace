# GhostTrace

A polished local-first research engine and visual workbench for controlled prompt-condition comparisons. React/TypeScript/Vite, Python/FastAPI, a single SQLite-backed worker, and direct Transformers residual hooks.

Read the project write-up on DEV: [GhostTrace: A Local Workbench for Looking Inside Language Models](https://dev.to/mackenzie-techdocs/ghosttrace-a-local-workbench-for-looking-inside-language-models-45p).

## Open the workbench

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:5173` and choose **Open recorded smoke**. The bundled evidence is a genuine local Qwen2.5-0.5B CPU/FP32 measurement: three development scenarios, two template families, three instruction conditions, greedy responses, aligned residual captures, and repeat/label-only/wording controls. All 42 planned units completed. It is a tooling diagnostic, **not a confirmatory pilot**. No model, API or network is required for recorded replay after setup. A plain empty state appears until evidence is explicitly loaded; no simulated measurements are shown.

`npm run build` creates `dist/`; `npm run preview` serves the production static replay.

## New to GhostTrace?

Open **Help & docs** in the sidebar (or append `#help` to the app URL) for a captioned two-minute video of the running project, a restartable guided demo, a first-session checklist and searchable research explanations. No model setup is required for the recorded case. See [the researcher getting-started guide](docs/GETTING_STARTED.md).

## Mechanistic Research Lab (v2)

Choose **Research lab → Recorded research case** to inspect the genuine pinned Qwen2.5-0.5B CPU/FP32 copying diagnostic. The linked workspace contains the validated component DAG and GQA groups; raw signed activation tiles for residual, normalized attention/MLP inputs, head contributions, gated MLP channels and bounded attention patterns; target-token logit-lens readouts; exact patch/ablation controls; gradient attribution with exact validation; and a declared two-stage receiver-path experiment.

The measured case also contains two dictionary projections with visible reconstruction error, an original-model feature-coordinate edit, first-order feature attribution, and a separately trained predictive-KL affine lens fitted on 24 development prompts and evaluated on 8 held-out prompts. The coordinate-identity dictionary is explicitly a **diagnostic basis, not a learned SAE**. The small-corpus lens is not a claim of corpus-general prediction quality. Replacing the entire final residual is an endpoint positive control by construction, **not a discovered circuit**. Negative effects and unclamped recovery are retained.

Locally, the v2 API supports arbitrary named role-message conditions, fixed target metrics, explicit component/token budgets, queued capture/intervention/artifact jobs, safe dictionary/lens imports, development-only lens fitting, feature-coordinate edits and annotations, external circuit import, hypothesis records and replayable research ZIPs. Viewing saved results never loads model weights. The separate v1 persona protocol and its review/holdout gates remain unchanged. See [the implementation and evidence status](docs/MECH_INTERP_IMPLEMENTATION.md).

## GitHub Pages preparation (not published)

```bash
npm run build:pages
npm run preview:pages -- --port 4174
# Open http://127.0.0.1:4174/Ghost-Trace/
```

This dedicated static mode automatically opens the measured research case, keeps graph/token/tile/readout/control inspection interactive, and removes API, inference and rating-save controls. Browser imports remain local. The manual workflow only builds, tests and uploads a review artifact; it does **not** deploy. See [Pages setup and subpath verification](docs/GITHUB_PAGES_SETUP.md).

## Run the full local stack

Use Python 3.11 or3.12 and Node 22. Dependencies are pinned in `uv.lock` and `package-lock.json`.

```bash
uv sync --frozen --extra test
npm run dev:full
```

This starts the localhost API, one explicit model worker and the workbench. **Experiments** provides model preflight, frozen configs, progress/events, cancellation, verified resume, actual outputs, queued analysis/export and evidence import/recomputation. **Behavior ratings** provides a balanced blinded human review cohort with original scores, disagreement and weighted-kappa summaries. API documentation: `http://127.0.0.1:8000/docs`.

## Explicit live-model setup

Model weights are never bundled or downloaded by startup, replay, API health, or inference. For the CPU runtime measured in this workspace:

```bash
uv pip sync --python .venv/bin/python requirements-cpu.lock.txt
HF_HOME="$PWD/artifacts/huggingface" .venv/bin/python scripts/download_model.py \
  --repo Qwen/Qwen2.5-0.5B-Instruct
HF_HOME="$PWD/artifacts/huggingface" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m ghosttrace.cli doctor --profile qwen25-05b-cpu \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

The explicit download is approximately 1 GB and resolves/inventories immutable Safetensors/tokenizer files. Inference is local-only with `trust_remote_code=False`. The measured CPU environment has a separate exact lock; the general model extra (`uv sync --frozen --extra model`) resolves device-specific wheels separately and needs its own preflight. Model primary/fallback choices are explicit; no backend or dtype is silently substituted.

## Automated validation and agentic evaluation

```bash
.venv/bin/python scripts/validate_release.py --mode ci --browser
```

The runner executes unit, integration, scientific-evidence evaluation, generated OpenAPI-contract freshness, lock/build/audit and production Playwright checks, writing hashed logs plus `artifacts/qa/release-validation.json`. Install browser test runtime once with `npx playwright install --with-deps chromium`. Missing real-model, replay, independent-agent or human signoff evidence is reported as pending, never passing. Release mode requires those gates.

Opt-in real-model acceptance tests never download weights:

```bash
HF_HOME="$PWD/artifacts/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GHOSTTRACE_REAL_MODEL=1 \
  GHOSTTRACE_MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775 \
  .venv/bin/python -m pytest tests/real_model -q \
  --junitxml=artifacts/real-model-tests.xml
```

Ten real-model checks cover the original paired alignment/capture/replay protocol plus v2 component/logit parity, head/GQA bookkeeping, selective interventions, gradient-versus-path semantics, candidate scoring and hook/model teardown. Agent review tooling prepares an evidence-hashed rubric packet and validates a structured independent review; it supports an explicitly configured trusted external reviewer command without a shell. See [QA](docs/QA.md) and [agent evaluation](docs/evaluation.md).

## Human review and release

Start with [the release handoff](docs/RELEASE.md). The prepared workspace includes an offline individual-review page at `artifacts/human-review/review.html`; scripts can regenerate it on a clean checkout. A named human reviews all 49 source items, applies hash-bound decisions to a new dataset version, freezes a matching preregistration, and validates the pinned tokenizer before the balanced pilot. Canonical holdout execution remains blocked without those receipts.

Real smoke, tokenizer and offline raw metric-recomputation gates have run. Human review, full confirmatory pilot, independent human ratings and final human release QA are intentionally not self-approved by an agent. Source/P0 release status and remaining limits are in [acceptance evidence](docs/acceptance.md), [protocol](docs/protocol.md), and [limitations](docs/limitations.md).

Original code: Apache-2.0. Synthetic dataset: CC0-1.0. Model and bundled font/other dependency licenses remain separate; see [third-party notices](THIRD_PARTY_NOTICES.md).

---
