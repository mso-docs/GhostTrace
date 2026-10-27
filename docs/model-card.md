# Model profiles

Primary target: Qwen/Qwen2.5-1.5B-Instruct. Explicit fallback: Qwen/Qwen2.5-0.5B-Instruct. Their model cards identify Apache-2.0 licensing. See [primary model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [fallback model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct); verify exact downloaded revision and its license before distribution.

No weights are bundled. Profiles must resolve immutable commits, tokenizer revision, weight/config hashes, chat template text/hash, capabilities and actual module dimensions. Never infer layer count from UI constants. No remote code, arbitrary model identifier, quantized canonical profile or silent model/dtype switch is allowed.

Planning memory estimates are not validated minimums. Run hardware doctor for three short prompts, hook shape checks, duplicate forwards, peak memory, 128-token generation timing and unload/reload. Record actual hardware, dtype, attention backend, OS, packages and threading. CPU/GPU/MPS results remain separate unless sensitivity is explicitly analyzed.

## Actual release diagnostic

The fallback Qwen2.5-0.5B-Instruct was explicitly downloaded and measured on CPU FP32 at immutable revision 7ae557604adf67be50417f59c2c2f167def9a775. Five real-model acceptance checks passed, and a3-development-scenario A/B tooling diagnostic completed42/42 units. This is not a confirmatory pilot; human review/ratings remain pending. Exact model/weight/tokenizer hashes, package versions, source hashes and measured device provenance accompany the recorded demo and local raw bundle. The exact measured LinuxCPU runtime is pinned separately in requirements-cpu.lock.txt.
