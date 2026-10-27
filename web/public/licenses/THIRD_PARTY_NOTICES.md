# Third-party notices

GhostTrace original application code is Apache-2.0 (see LICENSE). Original synthetic dataset and prompt drafts are separately CC0-1.0 (datasets/synthetic-v1/LICENSE). AI assistance was used to create this implementation and draft synthetic text; human scientific review is still required.

Target model families: Qwen/Qwen2.5-1.5B-Instruct and Qwen/Qwen2.5-0.5B-Instruct, published by Qwen under the licenses in their model repositories. No model weights are redistributed here. Downloaded weight licenses and notices must be retained by users.

Runtime dependencies retain their own licenses, including React, TypeScript, Vite, FastAPI, Pydantic, NumPy, PyTorch, Transformers and optional scikit-learn. The installed dependency/lock manifests identify actual versions; this notice does not relicense third-party work. Inspect package distributions for complete licenses and transitive notices before redistribution.

No external illustration, stock photograph, proprietary font or private research data is required by the application.

## Bundled interface fonts

DM Sans, Manrope, and IBM Plex Mono are bundled through their pinned Fontsource packages for offline use. Their SIL Open Font License notices are distributed in the respective npm packages. Font assets remain separately licensed from the application.

## Offline replay and browser QA

fflate 0.8.3 (MIT) performs bounded browser ZIP decompression. Playwright and axe-core are development-only automated browser/accessibility test dependencies. Bundled font OFL licenses are included under `web/public/licenses/` and copied into the production static assets.
