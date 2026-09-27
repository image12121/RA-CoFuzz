# Recorded environment and dependency lock

## Python environment

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.46.3 |
| Accelerate | 1.14.0 |
| FastChat | 0.2.36 |
| OpenAI Python SDK | 1.109.1 |
| NumPy | 2.3.2 |
| pandas | 2.3.3 |
| Hardware | NVIDIA RTX 4090 |

Install analysis-only dependencies with `requirements-analysis.txt`. Install
experiment dependencies with `requirements-experiment.txt`, then install a
CUDA-compatible PyTorch 2.8.0 build appropriate for the host.

## Recovered dependency provenance

The complete private archive recovered after v1.0.1 contains the following
audited upstream commits:

| Dependency | Commit |
|---|---|
| EasyJailbreak | `d5477e712e43b7a1a2110974b814ce578d609f5f` |
| PAIR (`JailbreakingLLMs`) | `6379ef705a0fc745530f7d895963510c021b496a` |

`extensions_q20/runtime_boundary_pass.json` additionally freezes byte-level
SHA-256 hashes for the EasyJailbreak files exercised by TAP, ReNeLLM, and
DeepInception. The commit and byte-level boundary are complementary checks.
Set `EASYJAILBREAK_ROOT` to the pinned checkout and run:

```bash
python extensions_q20/run.py audit --runtime
```

The runtime audit must pass before launching an experiment. A source mismatch
is a stop condition, not a warning.

## Endpoint configuration

The recorded mutation and StrongJudge provider is DeepSeek through its
OpenAI-compatible endpoint. The default model name is `deepseek-chat`. The
StrongJudge call uses temperature 0 and maximum output 512 tokens. Credentials
are intentionally absent and must be injected through `.env` or the process
environment.

## Target checkpoints

Target checkpoints are referenced by environment variables rather than private
absolute paths: `TARGET_LLAMA`, `TARGET_QWEN15`, `TARGET_QWEN3`, `TARGET_QWEN`,
and `TARGET_VICUNA`. Checkpoint redistribution is outside this release. The
recovered audit did not retain Hugging Face snapshot commit IDs, but it did
retain the SHA-256 digest of each local `config.json`; those hashes are
published in
[`provenance/q20_runtime_audit_public.json`](../provenance/q20_runtime_audit_public.json).
They identify the recorded local configurations without claiming a model
revision that was not preserved.

The RoBERTa evaluator is recorded as `hubert233/GPTFuzz`; its exact Hugging
Face snapshot revision was not preserved and is therefore explicitly marked
unresolved rather than guessed.

## Verification commands

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s extensions_q20 -p 'test_*.py'
python extensions_q20/run.py audit
python extensions_q20/run.py audit --runtime
```

The first three commands are safe offline checks. The runtime audit requires
the external dependency tree but does not launch the full 144-cell campaign.
