# GhostTrace pilot protocol · draft v1

Created 2026-10-27. **Status: draft, not a completed preregistration. No experimental results exist in this document.** This file must be reviewed, frozen and hashed together with a resolved model profile, dataset and config before opening holdout results. The bundled scenarios and prompts are AI-assisted synthetic drafts awaiting human review. Their presence on disk is not evidence that holdout has been analyzed.

The question is whether fixed persona instructions are associated with differences in expressed output behavior and matched-continuation transformer block representations. Persona labels describe instructions, not psychological states.

## Frozen design to review

- Dataset: synthetic-v1, 30 scenarios in five families. Development: accidental_breakage, missed_commitments, credit_taking (18). Holdout: avoidable_inconvenience, careless_communication (12). All derived texts inherit family/split.
- Treatments: neutral, remorseful, unremorseful, each in prompt families A and B. Shared base instruction, scenario, sentence count, renderer and chat template. Wording and absolute token position are acknowledged confounds.
- Canonical capture: teacher-forced common continuation, unpadded batch 1, no KV cache, model eval/inference mode, block outputs before final norm. First 16 verified identical continuation token IDs plus last-scenario/final-prompt anchors. Reject context overflow or token mismatch; retain exclusions.
- Canonical generation: greedy, max 96 new tokens, model EOS stop. No active temperature or top-p. Save stop/truncation flags and all failures/refusals.
- Canonical model: explicitly chosen local Qwen2.5-1.5B-Instruct or named 0.5B fallback. Resolve immutable model/tokenizer revision, hashes, environment, dtype and backend before freeze. No quantization or silent fallback.
- Fixed randomized condition order seed: 73. Reset history per unit. Seed Python, NumPy and torch, record deterministic algorithm exceptions.

## Budget

Smoke: three development scenarios × three conditions × one template × two modes = 18 main units, plus separately identified controls. Balanced: 180 capture + 180 generation + 30 exact-neutral-repeat + 30 label-only + 24 factual-task captures = 444 planned units. The 12 separate factual-task fixtures in factual-controls.jsonl must be human-reviewed before the full 444-unit plan is frozen. Optional sampled generation, PCA, probes and patching are outside the core pilot.

## Endpoints and analysis

Primary endpoint: holdout scenario-level remorseful–neutral normalized L2 distance at block floor(L/2)−1 (zero-based), first-16 continuation-token mean. Report matched neutral A/B wording distance alongside it. Normalized L2 = ||a−b|| / (0.5(||a||+||b||)+1e−8). Average template distances inside each scenario first. Bootstrap whole scenarios, retaining paired conditions and templates, 2,000 draws, seed 20261027; report mean, 95% percentile interval, scenario n, missingness, exclusions and family sensitivity. No confirmatory p-values.

Secondary: unremorseful–neutral, remorseful–unremorseful, cosine distance (missing if norm ≤1e−8), norm ratio, common-continuation NLL and expressed-behavior ratings. All layer scans and token heatmaps are exploratory. No selecting a layer after seeing separation. Tokens, layers, seeds and templates are never independent subjects.

Controls: saved-tensor identity (zero), exact same-backend neutral repeat (preflight target maximum absolute difference ≤1e−5 in FP32; measured reduced-precision floor recorded), neutral wording A/B, label-only neutral, factual tasks and whole-scenario condition permutation for optional probes. Failed repeat blocks canonical pilot until investigated or explicitly reclassified exploratory.

Ratings: at least 36 blinded responses balanced across conditions/templates, two independent raters, 0–2 regret/responsibility/repair/deflection. Preserve disagreements, report agreement and weighted kappa with small-n caveat. If one rater, disclose and omit inter-rater reliability. Human annotation is not claimed complete.

## Exclusions and stopping

Exclude primary pairs only for missing/corrupt artifacts, incompatible backend/site, context overflow or nonidentical continuation IDs/mask; report each reason. Retain failed and malformed units in run inventory. Stop on repeat failure, integrity failure or exhausted preregistered quota. Never rerun selectively to obtain separation. A null result is valid. A smaller descriptive pilot requires an amendment **before** execution and drops inferential language.

## Freeze checklist

- [ ] Human review of all dataset/template text, including the separate factual task fixtures.
- [ ] Exact tokenizer verifies common continuation token count and alignment.
- [ ] Hardware preflight, deterministic repeats and actual throughput/memory recorded.
- [ ] Immutable model/tokenizer revisions, file hashes and environment frozen.
- [ ] Dataset/config/preregistration hashes stored in experiment manifest.
- [ ] Holdout access logged only after the preceding freeze.
