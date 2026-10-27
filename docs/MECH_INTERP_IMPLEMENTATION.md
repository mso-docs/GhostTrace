# Mechanistic research implementation

The v2 research workspace supplements the original v1 controlled-persona protocol. It does not change the v1 preregistration, human-rating requirements, or existing replay format.

Implemented locally:

- Immutable arbitrary-condition studies with pinned model/backend provenance, explicit token targets or continuation likelihood, hypotheses, bounded captures and safe evidence exports.
- Literal residual, RMSNorm input, attention/MLP output, pre-output-projection head, projected head contribution, post-gate MLP and selected-query attention captures. Axes, token indices, omitted tiles and normalization are explicit. Residual bookkeeping and untouched-logit parity are checked against the original model.
- Model-derived transformer/GQA structure, raw signed activation microscope and actual final-normalization/unembedding logit-lens readouts.
- Exact patches, zero/selected-donor-mean ablation, seeded norm-matched random controls, self/reverse controls, raw effects and unclamped or undefined recovery. Gradient estimates and two-stage source/receiver path mediation have separate evidence types.
- Safe compatible ReLU, identity, JumpReLU and TopK linear SAE/transcoder packages, feature projection, reconstruction/error nodes and error-preserving feature edits in the original model. Learned semantic meanings require independent evidence.
- Affine lens imports, development-only ridge alignment and predictive-KL training with explicit held-out groups and frozen model parameters. The bundled small-corpus lens is a software diagnostic, not a generally validated language lens.
- Typed circuit imports, hypothesis/evidence records, and a static read-only research viewer for GitHub Pages.

## Measured case and limits

`configs/research-copy.json` defines a small blue/red copying diagnostic on the pinned Qwen2.5-0.5B-Instruct revision. `scripts/run_research_diagnostic.py` uses the same frozen studies, queue and worker services as the application. `scripts/extend_research_diagnostic.py` attaches an explicitly nonlearned identity-coordinate control and the independently fitted diagnostic lens. Negative, near-zero and undefined effects are retained. A last-layer full-residual replacement is a wiring positive control by construction, not discovery of a copying circuit.

The built-in model's real-model suite checks logits, components, head edits, path mediation and teardown. Other models, devices, reduced-precision profiles and externally published dictionaries need their own exact-revision compatibility/parity checks. The available published Qwen3 transcoder is not compatible with the bundled Qwen2.5 model and exceeds current safe per-package limits; it is not silently converted or downloaded.

A dictionary's reconstruction score is not behavioral fidelity. The demo uses an identity coordinate basis solely to exercise import, projection and original-model intervention controls; it contains no claimed trained SAE. See [artifact conventions](research-artifacts.md) for supplying a compatible learned dictionary.

## Reproduce and review

Run the ordinary Python suite, opt-in real-model tests, Playwright workflow tests and `scripts/evaluate_mechanistic.py`. The evaluation report records actual evidence hashes and checks; it does not substitute for human scientific review. Inspect `artifacts/mechanistic/service-validation-final/` and `artifacts/qa/` for local outputs. See the README and Pages guide for startup/build commands.

The Pages workflow prepares a downloadable production build for review. It does not publish the site. The static viewer executes no model inference and submits no ratings or live API calls; recorded effects remain labeled recorded.
