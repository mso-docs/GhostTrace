# Release QA runbook

The automated suite is executable evidence, not a declaration that human research review has happened. Test tensors and browser fixtures are marked synthetic and never bundled as measured demo data.

## Clean setup and one-command validation

Use Python 3.11 or 3.12 and Node 22. Install from the committed locks:

```bash
uv sync --frozen --extra test
npm ci
npx playwright install --with-deps chromium
uv run --frozen python scripts/validate_release.py --browser
```

The runner writes `artifacts/qa/release-validation.json` and a separate log with SHA-256 for every automated gate. It runs the unit, integration and evidence-evaluation suites, frozen-lock check, production build, local advisory-cache audit, and optional production browser suite. An advisory-cache audit must also be refreshed online before publication: `npm audit --audit-level=moderate`.

`--mode ci` exits successfully when automated checks pass while clearly retaining pending real-model/independent-review gates. `--mode release` exits nonzero for either a failed or pending gate. Never relabel a CI success as a release approval.

## Real-model evidence

Model downloads are explicit and are not part of default tests. Install `--extra model`, download an allowlisted fallback or primary model, resolve its immutable commit, and run doctor + smoke through the same CLI/worker used by the API. Confirm all three conditions have generation and capture units. Inspect token identity, shape/site, sixteen-token pooling, repeat tolerance, throughput, peak memory, and raw artifact checksums. Export an evidence ZIP with raw activations for replay recomputation.

```bash
uv run python scripts/evaluate_evidence.py path/to/real-evidence.zip --require-measured --output artifacts/qa/evidence-evaluation.json
```

The evaluator independently checks bounded archive paths, checksums, fixture separation, immutable provenance, all three conditions, capture metadata/masks, matched token IDs, controls, missingness, family splits, and versioned bootstrap settings. It never executes code from an archive or fetches a model. Numeric replay is validated by the engine import/replay contract tests; this structural evaluator complements those tests.

## Agentic review

```bash
uv run python scripts/agent_review.py prepare --validation-report artifacts/qa/release-validation.json
```

Give an independent review agent the generated packet and `docs/evaluation.md`. The agent must inspect evidence, reproduce checks, and produce the exact structured review schema. It must not substitute a confidence score or a general code impression for evidence. The validator checks review completeness, immutable packet hash, evidence file hashes/locators, path boundaries, explicit blockers and recommendation consistency.

```bash
uv run python scripts/agent_review.py validate --packet artifacts/qa/agent-review-packet.json --review artifacts/qa/agent-review.json
uv run python scripts/validate_release.py --mode release --browser --bundle path/to/real-evidence.zip --review-packet artifacts/qa/agent-review-packet.json --agent-review artifacts/qa/agent-review.json
```

Keep the validation report used by the reviewer immutable. Run a subsequent release report with a new filename to avoid changing the evidence the review referenced.

## Human review before release

1. Review every scenario and prompt template, then create a new reviewed dataset version and freeze the preregistration **before** running/unsealing holdout. Do not claim independent human review from an automated text check.
2. Run the smoke flow on the intended device/backend and inspect a few aligned token IDs and residual slices directly. Verify same-device repeat floor and genuine zero are distinct from missing/excluded results.
3. Execute a worker interruption/restart, cancel/resume and checksum-corruption recovery scenario. Confirm completed units do not change.
4. Use the production build with keyboard only: setup → monitor → comparison → chart table → method → export/import replay. Check focus visibility, dialog escape/focus return, narrow viewport, reduced motion, and high contrast/light fallback.
5. Disconnect the network and open the measured read-only evidence ZIP. Confirm persistent recorded provenance, identical aggregate metrics, and no model installation or weight fetch.
6. Independently rate the required blinded subset. Keep original scores and disagreements; if one rater is available, state that and omit inter-rater claims.
7. Inspect report prose and screenshots for unsupported emotion/causal claims; retain null results, failed units, exclusions and planned denominators.
8. Review license notices, archive inventory, absence of weights/credentials/private paths, clean-install instructions and current package audit. Record device, date, command and evidence path against every acceptance gate.

A release candidate can be ready for this checklist while the release itself remains blocked. Publish only after the evidence packet names the remaining blockers as resolved.

The release runner also accepts `--model-report artifacts/real-model-tests.xml --preflight artifacts/preflight-output.json` for opt-in model acceptance and hardware evidence. Skipped tests never satisfy the model gate.

Record explicit human approval as a JSON artifact passed with `--human-qa`. The artifact requires `schema_version: "1.0"`, `reviewer: {"kind":"human","name":"..."}`, `reviewed_at`, `release_approved: true`, and a `checks` object containing exactly these IDs: `dataset_templates`, `token_capture`, `recovery`, `keyboard_responsive`, `offline_replay`, `blinded_ratings`, `interpretation`, `release_inventory`. Each value requires `status: "pass"` and an `evidence` path or description. This is an inspectable attestation, not identity authentication; an agent must never fill it in on a human’s behalf.

Pass the actual raw-array replay/import result with `--replay-report artifacts/qa/raw-replay-validation.json`. The runner requires `recomputed: true`, raw arrays available, a `matched` verification status, and numeric tolerance. A summary-only replay is useful for offline viewing but cannot satisfy numeric reproducibility by itself.

Pinned-tokenizer protocol evidence is supplied with `--protocol-report artifacts/protocol-tokenizer-validation.json`. The report must cover all 30 main scenarios plus 12 factual controls with 16–32 matching continuation tokens. Human content review is a separate gate even when tokenizer checks pass.

The automated runner also exercises the real `dev:full` launcher against an isolated temporary metadata directory, checks API/worker/UI health without loading a model, sends SIGTERM only to the launcher, and verifies both ports close. Stop an existing local stack on ports8000/5173 before running validation; the checker refuses to signal or modify a pre-existing service. The launcher uses direct Vite invocation and owned Unix process groups so npm grandchildren do not survive shutdown.

`tests/integration/test_process_recovery.py` additionally kills a real worker OS process after one synthetic unit commits, interrupts the stale attempt, and verifies resume reuses that exact immutable record. These synthetic arrays test recovery behavior and are never research/demo evidence.

The startup checker also owns a stand-in service on5173 and verifies the launcher fails on that occupied port, cleans up its API/worker children, and leaves the unrelated stand-in alive. Vite uses `--strictPort`; the application never silently moves to a port outside the documented origin allowlist.

CI pins the supported Ubuntu24.04 runner instead of a moving `ubuntu-latest` image. The prepared Ubuntu26.04 workspace required Playwright’s official Ubuntu24.04 browser fallback during installation (`PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 npx playwright install chromium`); that headless runtime was actually exercised by the acceptance suite. Other machines should use their Playwright-supported native platform.
