# Release diagnostic smoke: A/B wording controls

Recorded before execution, October 27, 2026. This is a tooling diagnostic, not a confirmatory pilot or amended holdout analysis. No holdout scenarios are included. Dataset and persona text are unchanged from synthetic-v1.

Three development scenarios × three persona conditions × templates A/B × two canonical modes = 36 main units, plus three exact neutral-repeat captures and three label-only captures = 42 units. Model: explicitly selected Qwen2.5-0.5B-Instruct CPU FP32, immutable revision 7ae557604adf67be50417f59c2c2f167def9a775. Canonical16-token aligned mask, greedy96-token generation, no active sampling settings.

Purpose: validate the release pipeline, both template families, wording-control plots, raw export/import metric recomputation, and offline replay. All errors remain in the denominator. The prior A-only smoke is retained separately; no prompts or scenarios are selected or revised based on separation. No inferential research conclusion is drawn from these three development scenarios. Bootstrap intervals, if present as a pipeline check, are descriptive diagnostics only. Human scenario/template review and a frozen confirmatory preregistration remain required before a balanced holdout pilot.
