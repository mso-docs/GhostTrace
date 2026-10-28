# GitHub Pages deployment

The repository remote is `https://github.com/mso-docs/GhostTrace`, so the project-site base is `/GhostTrace/`. The workflow in `.github/workflows/pages.yml` is manually triggered. It builds and tests `dist/`, uploads a Pages artifact, then deploys from `main`. Set the repository’s Pages source to **GitHub Actions**.

## Build the read-only research viewer

First produce and review the real curated case at `web/public/research/demo.json`. It must be a version `2.0` research bundle with measured captures and their provenance, explicit controls and limitations. Do not replace it with generated mock activations or a test fixture.

```sh
npm ci
npm run build:pages
npm run preview:pages -- --port 4174
```

Open `http://127.0.0.1:4174/GhostTrace/`. This mode sets `VITE_STATIC_DEMO=true` and `VITE_BASE_PATH=/GhostTrace/`, automatically opens the recorded research case, and removes local API, inference, intervention submission and rating-save controls. Graph/token navigation, saved FP32 tiles, target readouts, recorded controls, compatible feature artifacts and case downloads remain interactive. Browser file imports are local; visitor research is not uploaded.

A different actual project slug requires its matching base:

```sh
VITE_STATIC_DEMO=true VITE_BASE_PATH=/ACTUAL-REPOSITORY/ npm run build
VITE_STATIC_DEMO=true VITE_BASE_PATH=/ACTUAL-REPOSITORY/ npm run preview -- --port 4174
```

The ordinary local application remains available through `npm run dev:full`; it preserves the version 1 persona protocol and the version 2 local research API. Pages cannot run the Python worker or model inference.

## Required review gates before publication

Run the local production browser suite separately from the project-site suite, because each serves a different build mode.

```sh
npm run build
npm run test:e2e
npm run build:pages
npm run help:validate:pages
npx playwright test -c playwright.pages.config.ts
```

Review direct entry and reload under `/GhostTrace/`, all fonts/assets/data downloads, keyboard graph and token navigation, the narrow-screen view, offline replay after loading, and **zero `/api` requests**. Inspect all four Research Lab panes. The endpoint whole-vector residual patch is a positive control by construction; it must not be called a discovered circuit. A coordinate-identity dictionary must be labeled a diagnostic basis, not a learned SAE. Small-corpus lens fitting must retain its limited scope and held-out metrics.

Inspect `dist/` before sharing. It must contain only the static app and deliberately curated public evidence; no model weights, metadata database, credentials, absolute host paths or private prompts. Raw local records and large arrays belong in explicitly downloaded research archives, not in the static app by accident. Browser tiles disclose omitted values rather than treating them as zero.

## Manual workflow handoff

After source is committed and pushed to `main`, run **Deploy GitHub Pages research demo** from the Actions tab. The deployment job runs only after the evidence, media, build and browser gates pass. Its `github-pages` environment links to the deployed site.

The deploy job has scoped Pages/OIDC permissions and depends on the verified build. Publication shares the curated recorded case; it does not establish completion of the separate human-review or confirmatory-study gates.


## Recorded video acceptance

The Pages artifact includes the actual WebM, English VTT captions, poster, transcript and integrity manifest under `help/`. The packaging gate verifies all five files match their source bytes and that app assets use the repository base. The Pages browser suite tests real playback and seeking, all 16 caption cues, project-relative media URLs, MIME types, the complete video checksum and HTTP byte-range responses. These checks run in both CI and the manual artifact workflow.

Before publication these checks use the production artifact served locally under `/GhostTrace/`; that is not a claim that the live GitHub CDN has been verified. After publication, run the same read-only suite against the actual public site:

```sh
GHOSTTRACE_PAGES_URL=https://mso-docs.github.io/GhostTrace/ \
  npx playwright test -c playwright.pages.config.ts
```

This skips the local preview server and verifies the published files against the checked-out source, so use the same revision that was deployed. It does not publish or modify the site. A changed repository slug or custom domain needs its actual public root URL with a trailing slash.

Hosting on Pages does not change a browser's codecs. The known VS Code embedded-browser decoding limitation can still occur when opening the public site inside that browser. Use a standalone browser for playback; the transcript and interactive guide remain available if decoding is unsupported.
