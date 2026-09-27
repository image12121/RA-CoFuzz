# Reproducibility guide

## Two reproducibility levels

The public package supports two distinct workflows:

1. **Aggregate consistency and figure verification:** uses the included safe aggregate, requires no model checkpoint or API key, and reproduces every public table and figure.
2. **Fresh experiment execution:** requires locally provisioned target checkpoints, an authorized mutation/judge endpoint, and the runtime dependencies used by the baseline implementations.

The original private per-response result tree used to build the released
aggregate is not available in the retained release materials. Consequently,
the first workflow verifies the internally consistent frozen aggregate; it does
not independently reconstruct the published numbers from historical raw
responses.

The manuscript has no separate appendix. This guide, together with
[`METHOD_AND_JUDGE_PROTOCOL.md`](METHOD_AND_JUDGE_PROTOCOL.md),
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md), and
[`PAPER_REPOSITORY_MAP.md`](PAPER_REPOSITORY_MAP.md), is the appendix-equivalent
reproducibility record referenced by the paper.

## Public verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
make verify-public
make figures
```

The figure command reads only `results/aggregate/summary.json`. Expected output is 18 standalone figures in PDF, SVG, and PNG.

## Aggregate-building interface

For a newly generated or independently retained `extension_outputs` directory,
build and validate an aggregate with:

```bash
python analysis/aggregate_results.py \
  --raw-root "$PRIVATE_RESULTS/extension_outputs" \
  --output results/aggregate/summary.json \
  --table-dir results/tables
```

The command validates exactly 144 cells and the phase breakdown before writing output. It rejects incomplete evaluation summaries, mismatched response counts, unknown labels, missing question coverage, and unexpected matrix cells. Without the original private tree, this command cannot serve as provenance reconstruction for the released numerical aggregate.

## Full runtime preparation

Copy `.env.example` to `.env`, fill only local paths and authorized credentials, and never commit that file. The recorded environment used Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.46.3, Accelerate 1.14.0, FastChat 0.2.36, OpenAI 1.109.1, NumPy 2.3.2, and pandas 2.3.3. See [`ENVIRONMENT_LOCK.md`](ENVIRONMENT_LOCK.md) before attempting full execution. PyTorch remains a platform-specific installation. The recovered audit pins EasyJailbreak and PAIR commits and records local model-config hashes.

Run the dependency-free tests before any expensive cell:

```bash
python -m unittest discover -s extensions_q20 -p 'test_*.py'
python extensions_q20/run.py audit
```

Before publishing or redistributing the release, also run:

```bash
make manifest
make verify-release
```

The included shell runners provide the matrix entry points. Review their local model paths and resource settings before execution. Do not launch multiple GPU cells on a single-device host unless memory isolation has been tested.

## Provenance and integrity

- `extensions_q20/original_hashes.json` records retained original-file hashes.
- `extensions_q20/reference_provenance.json` records reference-column provenance.
- `provenance/q20_runtime_audit_public.json` records recovered upstream commits,
  key package versions, hardware, and target-config hashes.
- `MANIFEST.sha256` covers every file in this public release except itself.
- Aggregate JSON includes protocol metadata and phase counts but excludes textual conversations.
- The retained release verifies aggregate structure and downstream artifacts;
  it does not contain the original per-response provenance needed to rebuild
  the published values.
