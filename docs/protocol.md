# Measurement protocol

GhostTrace measures instruction-associated representation differences. It does not detect emotions, consciousness, guilt, deception or a moral center. Follow [the draft preregistration](../preregistrations/pilot-v1.md) before a confirmatory run.

Primary capture is one teacher-forced forward pass with the same continuation under three persona contexts. Block hooks capture residual outputs before final model normalization, not attention maps or a final-normalized hidden state. Save FP32 selected slices as [layer, selected_token, hidden], explicit layer/token maps, absolute positions, role spans, normalization/site metadata and SHA-256. Remove hooks in finally. Logits at position p−1 score continuation token p.

Verify continuation token IDs in the fully rendered inputs; matching strings alone are insufficient. Reject mismatches from matched primary analysis. Different prefix lengths remain a position confound. Never compare token 10 across freely generated responses as equivalent linguistic content.

Pool 16 continuation token vectors within each unit, compute paired metrics within template, average templates within scenario, then bootstrap scenarios. Always show numerical-repeat and wording controls, n, uncertainty, missingness, exclusions and family-level sensitivity. Zero is a valid metric; unavailable values are null with reasons.

Model download is explicit and separate from experiment execution. Canonical computation is unquantized, fixed dtype/device/backend, batch 1, local files only after setup, safetensors, trust_remote_code=False, eval/inference mode. Revisions, config, weights, tokenizer, template, packages and environment belong in exported provenance.
