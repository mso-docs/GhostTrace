# Human QA and release handoff

The shipped demo is a **real recorded development tooling diagnostic**, not a confirmatory research pilot. It uses three development scenarios, both template families, all three conditions, fixed-continuation capture, greedy generation and repeat/label-only controls. It must not be described as emotion detection. Human review and the confirmatory pilot remain deliberate release gates.

## Start the complete application

```bash
npm ci
uv sync --frozen --extra test
npm run dev:full
```

Open `http://127.0.0.1:5173`. This starts the API, one worker and the workbench. No weights are downloaded by startup. Click **Open recorded smoke** to inspect genuine outputs, traces, uncertainty and provenance without loading a model or connecting to the API. The same static demo works from `npm run build` followed by `npm run preview`.

For live CPU inference on the validated Linux/Python 3.12 profile, install the measured exact environment with `uv pip sync --python .venv/bin/python requirements-cpu.lock.txt`. This is an explicit optional runtime install. The regular `uv.lock` also locks the general model extra (`uv sync --frozen --extra model`), whose device-specific wheels differ from the validated CPU environment. Record a separate preflight for any runtime/device change.

## Review the text before running the pilot

A ready-to-use offline review page is generated at `artifacts/human-review/review.html` in the prepared workspace. A fresh checkout can generate it:

```bash
uv run --frozen python scripts/review_dataset.py prepare \
  --output artifacts/human-review/review-packet.json \
  --html artifacts/human-review/review.html
```

Open the HTML locally. Individually review all 49 scenarios, factual controls and templates, add notes, enter your reviewer identity, and download the decisions. The page has no network access. There is deliberately no “approve all” shortcut. Human attestation must be made by the human reviewer; an agent must not create one on their behalf.

Apply the decisions to a **new** dataset version. Existing source/frozen experiments are never overwritten:

```bash
uv run --frozen python scripts/review_dataset.py apply \
  --decisions PATH_TO_DOWNLOADED_DECISIONS.json \
  --output datasets/synthetic-v1-reviewed \
  --reviewer YOUR_ID --attest-human
```

Any item marked needs_revision blocks application; revise a new source version and repeat review. Review receipts bind exact source hashes. After text review, freeze the protocol with an explicit model choice:

```bash
uv run --frozen python scripts/freeze_protocol.py \
  --dataset datasets/synthetic-v1-reviewed \
  --model-profile qwen25-05b-cpu \
  --output-protocol preregistrations/pilot-v1-reviewed.md \
  --output-config configs/pilot-v1-reviewed.yaml \
  --reviewer YOUR_ID --attest-human
```

Validate continuation token identity against the actual immutable tokenizer before any pilot run:

```bash
HF_HOME="$PWD/artifacts/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/validate_protocol.py \
  --dataset datasets/synthetic-v1-reviewed \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output artifacts/reviewed-tokenizer-validation.json
```

## Real computation and human ratings

The fallback smoke has already run in the prepared workspace. A fresh machine must explicitly download and preflight its own model. See README for commands. Keep CPU/GPU/dtype environments separate; never reuse another machine’s numerical tolerance as proof of local determinism.

Start the full stack and select your reviewed dataset and frozen preregistration in **Experiments → Advanced**. Alternatively:

```bash
.venv/bin/python -m ghosttrace.cli run --config configs/pilot-v1-reviewed.yaml
.venv/bin/python -m ghosttrace.cli worker
```

A canonical run containing holdout scenarios is blocked until matching review and preregistration receipts exist. Retain every failure, refusal and malformed response. Do not select scenarios or alter instructions based on apparent separation. Queue development/holdout analyses explicitly after completion.

In **Behavior ratings**, two independent human raters use distinct stable IDs and the same deterministic balanced blinded cohort (at least36 outputs). Condition, model, charts and provenance are hidden during rating; text may still reveal the instruction. Original ratings remain immutable. Agreement and weighted kappa become available with paired ratings; one-rater results carry an explicit limitation. Do not substitute agent/LLM ratings for the required human review.

## Execute release checks

See [QA.md](QA.md) for the executable validation command and [evaluation.md](evaluation.md) for independent agent review. The release runner emits machine-readable statuses and logs, verifies real evidence, and refuses to label pending human gates as passing. It can run unit/integration/evaluation/browser suites, generated OpenAPI type freshness, lock/build/audit checks, real-model JUnit evidence, tokenizer/preflight evidence, numeric replay evidence, and independent agent-review packets.

Review the desktop/mobile screenshots, keyboard flow, offline bundle, raw export/recompute result, and cancellation/resume behavior before signing `docs/human-qa-template.json`. Copy it into `artifacts/qa/human-qa.json`; fill it as a named human with evidence links. Release mode stays pending/fails until this explicit signoff is supplied. Never edit automated report statuses to make release_ready true.

## Release artifacts

- `dist/`: static UI, fonts/licenses and a small genuine read-only recorded smoke bundle.
- `web/public/demo/evidence.zip`: checksummed summary replay, raw arrays explicitly omitted.
- `artifacts/measured-smoke-raw.zip`: local raw recomputation bundle (ignored by Git).
- `artifacts/qa/`: tests, replay verification, agent evaluation and screenshots.
- `artifacts/human-review/review.html`: local protocol review handoff.
- `docs/acceptance.md`: evidence and limitations; `THIRD_PARTY_NOTICES.md`: licenses.

Model weights, virtual environments, caches and local metadata databases are not release assets. Host the static replay only if desired; real inference remains a localhost service. The user must not be told that a successful smoke/QA run validates remorse, consciousness, or a causal psychological mechanism.
