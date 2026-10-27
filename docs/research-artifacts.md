# Research artifact contracts

Dictionary and lens packages are ZIP files with exactly two root entries: `dictionary.json` or `lens.json`, and `weights.npz`. The JSON contains `metadata` and `weights_sha256`. NPZ members must be finite float32 NPY arrays; object arrays, pickle, executable loaders, encrypted/duplicate ZIP entries and path traversal are rejected. Current package limits are 200 MB expanded arrays, 100 MB per array, and 210 MB archive overhead allowance. Imported artifacts are immutable and bound to an exact model repository, 40-character revision, component, normalization state, width, and license. Integrity proves which bytes were supplied, not authorship or scientific validity.

## Dictionary convention

`W_enc[input_width,feature_count]`, `b_enc[feature_count]`, `W_dec[feature_count,output_width]`, `b_dec[output_width]` are required. The encoder is `z = activation(x @ W_enc + b_enc)`, or first subtract `b_dec` when `preprocessing=center_decoder_bias` and input/output widths match. Reconstruction is `z @ W_dec + b_dec`. SAE input/output site and width must match; a transcoder may use different sites with independently declared input/output normalization states.

Supported activations are identity, ReLU, JumpReLU (`pre * (pre > threshold)` with a finite nonnegative `threshold[feature_count]` array), and TopK (keep the largest `top_k` positive ReLU entries; feature-index order resolves ties). Log thresholds, gated encoders, skip connections and other architectures must be converted explicitly or rejected. Conversion must retain original checksum, revision, architecture and preprocessing provenance; renaming an unsupported architecture is insufficient.

The identity-coordinate diagnostic has exact identity weights, zero biases and identity activation. It is explicitly not a learned SAE and its coordinates have no inferred semantic labels. Reconstruction error, explained variance and activation sparsity do not establish behavioral fidelity. Feature interventions patch `y_original + decoded_modified - decoded_original`, preserving the unexplained error; reconstruction-only and self controls are separate original-model measurements.

## Affine lenses

Lens arrays are `W[layer,input,output]` and `b[layer,output]`. Metadata selects an affine output `h @ W + b` or residual affine delta `h + h @ W + b`, a component and exact base-model revision. Output token probabilities require the compatible model's final normalization and unembedding. Translated vectors alone are not token predictions or hidden thoughts.

Local predictive-KL training follows the residual affine readout objective: detached final-layer output distributions supervise translators at earlier layers. Only translator `W,b` gradients are requested; base-model parameters and their gradient buffers remain untouched. Ridge hidden-state alignment is available as an explicitly different objective, never labeled canonical predictive-KL tuned lens. Development groups alone fit weights. Held-out groups are disjoint and only score fit quality. The default supplied corpus minimum is 1,024 rows and 100 development groups. An explicitly labeled small-corpus diagnostic lowers this to 32 rows and 16 groups and supports no generalization claim. The 30 persona scenarios must not be used as the training corpus.

Method references: [Tuned Lens paper](https://arxiv.org/abs/2303.08112), [official translator implementation](https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py).

## Circuit evidence

Structural connections, observed activation, attribution estimates, original-model interventions and tested paths have distinct evidence types. Missing effects are `null`, never zero. Local intervention/path labels require a completed matching record. Display filters retain evidence type and report hidden counts; node counts are not model fidelity.

The Circuit Tracer importer accepts safe frontend `nodes`/`links` JSON with pinned source metadata. Reconstruction-error nodes and reported weights remain visible. External causal labels are retained as unverified source annotations and demoted to attribution estimates; importing a graph cannot establish a local causal test. Unknown fields are reported explicitly. Pickled `.pt` graph files are unsupported. Source conventions: [official frontend graph schema](https://github.com/decoderesearch/circuit-tracer/blob/main/circuit_tracer/frontend/graph_models.py), [replacement-model methods](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html).

## Published Qwen3 PLT resource audit

Metadata-only audit: `artifacts/research/qwen3-dictionary-audit.json`. The public `mwhanna/qwen3-0.6b-transcoders-lowl0` repository at `28aefe686c09e9bc1a862195b51e145f10868d87` declares MIT, Qwen/Qwen3-0.6B, MLP input/output hooks, ReLU, width 1024, 163840 features, no skip connection and unit input/output scales. Each layer safetensors file is 671,418,696 bytes. Its supplied configuration does not pin the base-model revision. This exceeds this application's bounded dictionary package and feature-count limits and is incompatible with the Qwen2.5 diagnostic model. No dictionary weights were downloaded. A separate resource profile, immutable base-revision attestation and explicit safe conversion are required before use; silently truncating features or mapping to another model is invalid.

## Executable validation

Run `uv run python -m pytest tests/research tests/mechanistic -q`. Predictive-KL/native gradient cases require the optional model runtime and explicitly skip when it is absent; real-model release evidence must supply the separately recorded five-test JUnit file. Validate the curated measured case with `.venv/bin/python scripts/evaluate_mechanistic.py --bundle web/public/research/demo.json --real-model-junit artifacts/mechanistic/real-model-tests.xml`. This writes a structured report with archive hashes and fifteen numerical/provenance checks. It does not replace human scientific review.

Run `npm run build:pages`, then `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 npx playwright test -c playwright.pages.config.ts`. The three static browser tests exercise the `/Ghost-Trace/` project subpath, offline navigation, absence of API requests, keyboard component selection, exact feature estimate/reference/error values and all four mobile tabs' WCAG A/AA checks. The ordinary production build uses `npm run build` and `npm run test:e2e`. Both build modes are tested independently because static Pages disables inference and local API controls.
