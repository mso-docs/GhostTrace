# General mechanistic research API (v2)

The v2 namespace `/api/v2/research` is separate from the immutable v1 persona protocol. It supports arbitrary named conditions and role messages, explicit next-token logit-difference or fixed-candidate likelihood targets, selected component capture, controlled interventions, feature dictionaries, affine lenses, typed graphs and hypothesis records. These are research instruments, not emotion detectors or automatic semantic circuit discovery.

Use the same local API and explicitly launched single worker as v1. Fetch `/api/v1/health`, then provide `X-GhostTrace-Token` on mutations. Jobs use the existing `/api/v1/jobs/{id}` status/events/cancel/resume routes and return resource IDs only after computation completes. Nothing implicitly downloads models. A frozen validated local profile must match before any model operation; imported studies and dictionaries do not become trusted local preflights.

## Core workflow

1. `POST /studies` with a strict `StudyCreate` JSON object (see `/openapi.json` or `configs/research-copy.json`). Conditions have unique IDs, names and `messages`; optional `group_id` records experimental grouping. The frozen study includes its complete config, model profile, hashes, capture budget and prior hypothesis.
2. `POST /studies/{id}/captures` with `{}` or `{condition_ids:[...]}` queues component inspection. `GET /studies/{id}/bundle` links captures, a structural graph, measured overlays and all supporting records.
3. `POST /studies/{id}/interventions` queues a frozen `kind:exact|attribution|path` specification: donor/recipient condition IDs, site, zero-based block, explicit positions, operation, optional head/channel and alignment declaration. Paths require a strictly later receiver and may select receiver head/channel separately.
4. `GET /captures/{id}/tiles?array=L11.resid_post` returns bounded literal coordinates. Selected token/query rows and flattened head/channel/key axes are disclosed; omitted coordinates are not zero. Full selected arrays remain in bounded, checksummed FP32 NPZ artifacts.
5. `POST /studies/{id}/hypotheses` stores a versioned interpretation, expected/actual effects, alternatives, control outcomes and supported/refuted/unresolved status. An annotation is not scientific certification.

Targets are `next_token_logit_diff` (one positive/negative vocabulary token) or `continuation_loglikelihood` (1–64 token candidates, explicit sum/mean normalization). Text targets are resolved using the pinned tokenizer and the resolved IDs are saved. Inputs are bounded to 512 tokens; selected positions to 32, layers to 32, and conditions to 12. Native captures additionally enforce 64 MB per forward; study quotas and conservative planning estimates are separate from measured memory.

Capture sites: `resid_pre`, `attn_in`, `attn_out`, `resid_mid`, `mlp_in`, `mlp_out`, `resid_post`, `final_norm`, `head_pre`, `head_contribution`, `mlp_post_gate`, and opt-in `attention_pattern`. Normalized attention/MLP inputs remain distinct from residual vectors. Head identities and grouped-query KV relationships come from the loaded config. Projected head contributions and attention patterns are read-only; head patches use `head_pre` before output projection.

Exact operations are `patch`, `zero_ablate`, `mean_ablate` and `norm_matched_random`. Mean ablation uses the declared donor selected-position mean, not an undeclared corpus. Absolute token correspondence is verified; differing content requires an explicit semantic-alignment explanation. An optional distinct `same_condition_condition_id` supplies an independent matched control. If omitted, the exact recipient repeat is explicitly labeled and never described as an independent donor. Untouched/self/reverse/random controls, raw signed deltas and unavailable small-gap recovery states remain visible. Gradient attribution is first-order estimation; path patching tests a specified two-stage mediation and does not prove a unique direct edge.

## Feature and lens artifacts

- `POST /dictionaries/imports`: raw ZIP containing exactly `dictionary.json` and `weights.npz`. Metadata and array contracts are documented in [research artifacts](research-artifacts.md). Linear SAE/transcoder encoders support identity, ReLU, declared JumpReLU thresholds and deterministic TopK. Base revision, sites, separate input/output normalization, layer, widths, preprocessing, dictionary revision and license must match.
- `POST /studies/{id}/features`: `{capture_id,dictionary_id}` queues encoding and reconstruction/error analysis. Full coordinates remain stored; display ranking discloses omitted features. Coordinate-basis diagnostic dictionaries are explicitly nonlearned.
- `POST /studies/{id}/feature-interventions`: `{capture_id,dictionary_id,feature_indices:[...],operation:ablate|scale|set,value:0}` measures the decoded edit in the original model. The applied vector preserves the original unexplained reconstruction residual. Reconstruction-only, error-restoration, self and random controls distinguish behavioral fidelity from reconstruction MSE.
- `POST /studies/{id}/feature-attributions`: `{capture_id,dictionary_id,top_k:64}` computes original-model component gradients and first-order decoded-coordinate zero-ablation estimates. Decoder bias and unexplained reconstruction-error contributions remain separate. Matching exact feature interventions are compared explicitly; estimated graph edges are never promoted to tested paths.
- `POST /lenses/imports`: raw `lens.json` + `weights.npz` ZIP. Imported training provenance remains externally declared.
- `POST /lenses/train`: raw `training.json` + `weights.npz` ZIP. The document supplies lens metadata, model profile ID, NPZ SHA, explicit development mask and family/group IDs, optional ridge and bounded optimization settings. Arrays are `targets` and `inputs.L<block>` for each declared layer. Only development rows fit the translator; holdout groups are disjoint and evaluated separately. Predictive-KL training uses the compatible frozen model decoder. Ridge hidden-state alignment is labeled as a distinct objective and never called canonical predictive-KL training. Corpus scope and minimum sample/group requirements are enforced; the thirty persona fixtures are not a calibration corpus.
- `POST /studies/{id}/lenses/apply`: `{capture_id,lens_id,decode:true,layers?:[...]}` queues a compatible translated readout. Without a decoder, translated vectors are explicitly not token predictions.
- `POST /studies/{id}/circuits`: `{document,source_metadata}` imports bounded safe graph JSON, including Circuit Tracer frontend JSON. External causal/path claims are retained as unverified annotations or estimates. Imported IDs are namespaced so they cannot overwrite native nodes or acquire local evidence.

Graph nodes retain all condition-specific observation references and intervention references. Structural, observed activation, approximate attribution, exact intervention, tested path, external report and human interpretation are distinct. Filtering counts and reconstruction/coverage limitations are exposed; a sparse displayed graph does not establish causal completeness.

## Portable evidence and CLI

`POST /studies/{id}/exports` queues a raw or summary ZIP. `POST /imports` accepts raw ZIP bytes (500 MB limit) and queues safe read-only import. The bundle contains canonical study-definition/model-profile bindings, immutable records, literal-coordinate tiles and optional raw arrays. Paths, duplicates, symlinks, encrypted archives, nonfinite JSON, invalid NumPy headers and shape/size violations are rejected before registration. Imports verify raw checksums and recompute saved capture tiles within absolute 1e−6 / relative 1e−5; they do not silently rerun a model or certify causality. Global graph views must match their immutable evidence records. Imported array paths are relocated separately from original hashed records.

```bash
ghosttrace research create --config configs/research-copy.json
ghosttrace research capture --study STUDY_UUID
ghosttrace worker
ghosttrace research intervene --study STUDY_UUID --spec intervention.json
ghosttrace research bundle --study STUDY_UUID --output research.json
ghosttrace research export --study STUDY_UUID
ghosttrace research import evidence.zip
```

The CLI also exposes `dictionary-import`, `features`, `feature-intervene`, `feature-attribution`, `lens-import`, `lens-train`, `lens-apply`, `circuit-import` and `hypothesis`, using the same engine functions as the API. Safe artifact handling supports supplied compatible dictionaries; it does not assert that any published artifact is compatible without its required record and checks.
