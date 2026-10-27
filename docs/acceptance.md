# Acceptance evidence and human release gates

GhostTrace is a release candidate for human QA. Automated implementation checks, real fallback-model smoke, and recorded replay have run. An agent must not claim that humans reviewed the protocol, collected ratings, or approved a confirmatory pilot.

## Automated and measured evidence — October 27, 2026

| Gate | Evidence / status |
| --- | --- |
| Python contracts | Core unit/integration/evaluation suites pass; exact current counts are in the machine-readable report. Five real-model tests are intentionally skipped by default. Covers strict types/options/hashes, splits/alignment, metric identities and bootstrap, malformed activation headers before allocation, SQLite cancellation/resume/checksums, loopback/token/request limits, queued task lifecycle, typedOpenAPI, safe bundle import/recompute, raters/cohorts/kappa and evidence/agent-review validators. |
| Actual model | Separately opted-in real-model suite:5passed. Pinned Qwen/Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775, CPU FP32, Python 3.12.15, PyTorch 2.7.0+cpu, Transformers 4.51.3. `artifacts/real-model-tests.xml`. |
| Capture validation | Actual block shapes, exact aligned continuation IDs, deterministic forwards, correct pre-final-norm last-block site, p−1 continuation NLL, greedy output repeat and hook cleanup after a forced failure pass. |
| Hardware preflight | Successful3 generations, repeat maximum absolute error 0.0 (FP32 tolerance 1e−5), measured memory/throughput, unload/reload. Actual values/environment: `artifacts/preflight-output.json`. |
| Tokenizer protocol | All 30 main +12 factual-control scenarios checked with the pinned local tokenizer, both template families and3 conditions; continuations16–32 tokens and corresponding IDs verified. Absolute-position differences remain disclosed. `artifacts/protocol-tokenizer-validation.json`. Human review remains false. |
| Real smoke matrix | A-only smoke24/24 units complete, retained separately. Preregistered release A/B diagnostic42/42 units complete, no failures, only3 development scenarios. `preregistrations/release-smoke-ab.md`. No holdout opened. |
| Recorded demo | `web/public/demo/evidence.zip`, descriptor and README contain actual measured42-unit smoke. No model weights or raw arrays in the small browser bundle. Exact zeros, missing values, controls, unit IDs/masks, responses and provenance are retained. |
| Offline numeric replay | Clean-store raw import independently recomputes metrics without a model. `artifacts/measured-smoke-raw.zip`; `artifacts/qa/raw-replay-validation.json` documents matched values and tolerance. Summary-only replay is not relabeled numerical recomputation. |
| Browser acceptance | The production Playwright acceptance suite passes, including real offline demo, keyboard/mobile navigation, live API failure, corrupt/unsafe archives, token/block selection, zero/missing values and no invented production measurements. Automated axe checks report zero WCAG A/AA violations across all 6dark views and mobile/light fallback. Latest logs/screenshots: `artifacts/qa/`. Automated checks do not replace human accessibility QA. |
| Build/dependencies | TypeScript/Vite production build passes; localfonts/licenses bundled; npm audit reports0 vulnerabilities. `uv.lock`, `package-lock.json`, and the exact measured LinuxCPU runtime `requirements-cpu.lock.txt` are pinned. Generated OpenAPI TS contracts are checked for drift. |
| Agentic evaluation | Independent review has already found and triggered fixes for archive/header bounds, shared blinded-rater cohorts, chart legend/readability, repeated polling, stale review evidence, and dark/light contrast. Hash-bound structured agent packet/review and release gate reports live in `artifacts/qa/`; final status is explicit, never inferred from a rubric alone. |

The executable current verdict is `artifacts/qa/release-validation.json` (or the newer filename named in QA output); consult it together with hashed logs rather than treating this document as a substitute for tests. Release-mode validation refuses pending human gates.

## Human work intentionally not self-approved

- Review all 49 scenario/control/template items in `artifacts/human-review/review.html` (regenerable with `scripts/review_dataset.py`). Approve individually; export decisions and apply to a new dataset version. No agent-generated human attestations exist.
- Freeze the reviewed protocol/config with `scripts/freeze_protocol.py` before the 444-unit balanced pilot. Canonical holdout execution and holdout analysis require matching dataset/template/control/human-review/preregistration receipts.
- Run the balanced pilot on the intended device and preserve failures/exclusions. The existing3-scenario development smoke is not a substitute for this pilot.
- Collect at least36 balanced blinded responses with independent human raters; examine original scores, disagreements, agreement and weighted kappa. One-rater use is explicitly limited and does not establish inter-rater reliability.
- Perform human keyboard/accessibility/responsive/interpretation/recovery/replay QA and fill `docs/human-qa-template.json` into a separate named-human evidence artifact. The release runner does not permit a missing attestation to pass.

[RELEASE.md](RELEASE.md) gives the exact short workflow. These steps are human decisions and empirical research gates, not hidden unimplemented buttons.

## Scope and remaining optional work

P0 engine, API/CLI queue/recovery, capture, paired metrics/controls, visual compare/explorer, method/provenance and safe offline replay/export are implemented and exercised. P1 blinded rating collection/reliability is implemented; original scores remain immutable. Shared PCA, optional sampled/paraphrase experiments, and stretch probes/patching are not enabled or represented as measured. The spec describes PCA as optional and probe/patching as stretch; they are outside the release-critical path.

A single upstream Starlette/AnyIO deprecation warning remains in contract tests. Transformers 4.51.3 emits a sliding-window constructor warning for the pinned config even though use_sliding_window=false; the actual attention path was inspected and the setting is recorded in model provenance. No runtime/model config is silently altered to hide the warning.

Scientific interpretation remains bounded: instruction-conditioned behavior/representation differences are descriptive. They do not demonstrate remorse, consciousness, guilt, deception, a moral center or causal emotional mechanisms. The measured label-only/wording controls must be shown even when they weaken persona-specific attribution.
