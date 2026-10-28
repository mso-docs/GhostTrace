# A first research session in GhostTrace

Open **Help & docs** in the sidebar, or use `#help` on your local or Pages URL. The in-app guide includes a two-minute recording of the running application, descriptive English captions, a full transcript, chapter navigation, a first-session self-check and searchable explanations.

The recording is silent by design; all instruction is available as captions and text. It shows the real bundled Qwen2.5-0.5B CPU/FP32 case. It executes no new model inference. [Download the recording](../web/public/help/researcher-walkthrough.webm), [read the transcript](../web/public/help/walkthrough-transcript.txt), or inspect its [capture and integrity manifest](../web/public/help/walkthrough-manifest.json).

## Start without a model

```sh
npm ci
npm run dev
```

Open the printed local URL, choose **Help & docs**, then **Start guided recorded demo**. This explicitly opens the bundled read-only case and restarts its six-step guide. Saved local studies remain on disk. No API key, account, model download or GPU is required for replay. Pages opens the same recorded research workspace automatically.

## Follow one question from baseline to limits

1. **Case study:** read the blue-minus-red next-token target, prior hypothesis and unresolved interpretation. The donor and recipient differ in the code color. Assess baseline task behavior before making claims about a mechanism.
2. **Linked workspace:** choose a captured block, token and component. The graph describes literal model structure; structural connections are not discovered causal edges. Inspect the component's normalization and token mask.
3. **Activation microscope:** select a signed coordinate. Check the tensor axes and coverage. Saved tiles are bounded, so omitted coordinates remain unavailable rather than zero. Activation magnitude does not identify a semantic concept.
4. **Target-output readout:** compare available blocks and instruments. The untrained lens projects through final normalization/unembedding. The fitted lens is a separate small-corpus predictive-KL diagnostic, fitted on 24 development prompts and evaluated on 8 held-out topic groups. Neither is a transcript of thoughts.
5. **Interventions:** inspect the exact settings and raw target change alongside controls. Gradient estimates and exact edits remain separate. A two-stage receiver-path experiment supports bounded mediation, not a unique direct route. Replacing the entire final residual is a wiring control by construction.
6. **Features & circuits:** inspect reconstruction and unexplained error; compare the saved first-order estimate with an original-model edit. The bundled identity dictionary is a coordinate diagnostic, not a learned SAE. Reconstruction alone does not establish behavioral fidelity or semantic meaning.
7. **Case study and export:** retain alternatives, null/negative results and unresolved conclusions. Case JSON preserves viewer records and provenance. Export a research ZIP with selected raw arrays from the local engine when a colleague needs independent numeric recomputation.

The in-app checkboxes track your own understanding for the current visit. They do not approve a scientific result, change a study, or certify human QA.

## Run your own study

Use Python 3.11 or 3.12 and Node 22:

```sh
uv sync --frozen --extra test
npm ci
npm run dev:full
```

The full stack starts the localhost API, explicit worker and UI. Model dependencies, weights and a pinned validated profile are separate requirements: follow [explicit live-model setup](../README.md#explicit-live-model-setup), including the CPU runtime lock, explicit weight download and `doctor` preflight. Startup and replay do not download model weights.

In **Research lab**, connect to the local API, write a bounded question and hypothesis, define aligned conditions and a fixed target, choose explicit sites/blocks/token positions/budgets, then freeze the study. Queue captures before interventions. Keep self/random/reverse/same-condition controls where appropriate, examine failures and raw effects, and record interpretation separately. Do not use the original persona protocol's holdout data to tune exploratory studies.

Pages can display saved evidence but cannot run the Python worker. Browser imports remain local. For a useful report, record the job error, model revision/backend, target, token alignment, component site and smallest reproducible configuration; review private prompts before sharing.

Further reading: [implementation and measured limits](MECH_INTERP_IMPLEMENTATION.md), [artifact compatibility](research-artifacts.md), [research API](research-api.md), [human release QA](QA.md).

## Maintain the video

The recording is produced from the actual running viewer, not generated frames. Re-record after changing relevant UI or the bundled case:

```sh
# Terminal 1 — a production snapshot avoids development hot reloads
npm run build:pages
npm run preview:pages -- --port 4180 --strictPort

# Terminal 2 — requires the installed Playwright Chromium runtime
WALKTHROUGH_URL=http://127.0.0.1:4180/GhostTrace/ npm run record:walkthrough
node scripts/validate_help_assets.mjs
# Rebuild when packaging the newly recorded media
npm run build:pages
```

The recorder uses the shared chapter script in `web/src/help-content.json`, saves a two-minute WebM, a poster, VTT captions, a transcript and hashes. Captions describe the action and its research meaning. Recording is a deliberate maintenance operation, not part of ordinary CI; CI validates the checked-in assets and browser playback instead. If a browser cannot play WebM, the transcript and interactive guide remain available.

Changes to source invalidate the earlier release's frozen independent-review packet. Re-run relevant checks and obtain a fresh release review before publication; the previous packet remains historical evidence.

## Video fails inside VS Code

The recording is VP8 WebM. Some VS Code shipped Electron builds have a [known WebM decoding issue](https://github.com/microsoft/vscode/issues/343345), even though a standalone browser can play the same file. Open your app URL (for example `http://127.0.0.1:5173/#help`) in Firefox, Chrome or Edge, or download the video and open it in a media player. The Vite Fast Refresh message after a source edit is separate from video decoding.

The help player reports the actual media error code: 2 means a network/download failure; 3 means a decoding failure; 4 means no supported source. Retry loading after fixing a connection problem. Captions failing to load do not by themselves mark the video as unplayable; the written transcript remains available.
