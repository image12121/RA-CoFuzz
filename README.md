# RA-CoFuzz

RA-CoFuzz is a refusal-aware, hybrid-selection fuzzing framework for evaluating the robustness of large language model safety alignment. This repository is the cleaned public release assembled from the completed Q20 experimental campaign. It also serves as the paper's appendix-equivalent technical record: implementation details, evaluator configuration, protocol decisions, extended tables, and reproducibility notes are maintained here instead of in a manuscript appendix.

The release contains the framework implementation, baseline adapters, frozen experiment protocol, a frozen content-free aggregate for 144 reported cells, tests, and publication-ready figures. Raw model conversations, API credentials, caches, checkpoints, and private machine-specific files are intentionally excluded. The original private per-response tree used to build the released aggregate is not available in the retained release materials, so the public package supports aggregate consistency checks and figure regeneration but not a byte-for-byte reconstruction of the published aggregate from historical raw records.

Release 1.0.4 aligns the appendix-equivalent repository record with the final
manuscript: it defines both ASR denominators, freezes the active selector
equations and exact model identifiers, and supplies every result panel as a
standalone figure. It does not alter the active algorithm, completed results,
or plotted values; see
[`CHANGELOG.md`](CHANGELOG.md).

> **Responsible-use scope.** This project is intended for authorized safety evaluation and defensive research. Run it only on models and services you are permitted to test. The repository reports aggregate measurements and does not publish generated model conversations.

## Experimental scope

| Dimension | Configuration |
|---|---|
| Methods | RA-CoFuzz, Strict GPTFuzzer, PAIR, TAP, ReNeLLM, DeepInception |
| Datasets | GPTFuzzer, AdvBench, JailbreakBench |
| Target models | Llama-3.2-3B-Instruct; Qwen2.5-1.5B-Instruct; Qwen2.5-3B-Instruct; Qwen2.5-7B-Instruct; Vicuna-7B-v1.5 |
| Published run labels | 100, 200, 300 (see randomness caveat below) |
| Per-cell protocol | 20 questions, target-query cap 200, greedy decoding, 128 new tokens |
| Evaluation | selective online and full offline StrongJudge (`deepseek-chat`), plus post-hoc RoBERTa |
| Completed cells | 144 |

The 144 cells comprise 18 main-comparison cells, 36 cross-dataset cells, 72 cross-model cells, 9 selection-ablation cells, and 9 feedback-ablation cells. The GPTFuzzer/Llama-3.2-3B-Instruct reference condition is counted once in the 18-cell main comparison and reused when reporting the complete three-dataset and five-model matrices. The labels `100/200/300` identify the three released repetitions, but they must not all be described as independent random seeds: the historical RA-CoFuzz/Strict-GPTFuzzer driver reset Python's RNG to 100 inside each process, while historical PAIR used 1235/1236/1237. See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

Question-level and response-level ASR intentionally use different denominators:

```text
ASR_question = successful questions / evaluated questions
ASR_response = Label-2 responses / recorded responses
```

A question is successful when at least one of its recorded responses receives
Label 2. The exact mathematical definitions and JSON field mapping are in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Results at a glance

On the main GPTFuzzer/Llama-3.2-3B-Instruct setting, RA-CoFuzz attained a mean question-level Label-2 ASR of **81.7%**, compared with **61.7%** for Strict GPTFuzzer and **60.0%** for PAIR. All reported values are means across the three released runs; uncertainty bars show sample standard deviation across those runs.

![Main comparison](figures/fig02a_main_question_asr.png)

The full tables are available in [`results/tables`](results/tables), and the machine-readable aggregate is [`results/aggregate/summary.json`](results/aggregate/summary.json).

## Rich figure suite

The release includes 18 standalone manuscript-oriented figures in PDF, SVG,
and PNG. Former multi-panel plots are separated so each image can carry its own
caption and explanatory paragraph:

1. RA-CoFuzz closed-loop framework overview
2. Main six-method question-level comparison
3. Main response-level comparison
4–5. Separate cross-dataset heatmaps
6–7. Separate cross-model heatmaps
8–9. Separate hybrid-selection ablations
10–12. Separate refusal-aware feedback ablations
13–14. Separate effectiveness–cost trade-offs
15. StrongJudge–RoBERTa alignment
16–17. Separate three-run stability plots
18. Response-label composition

See [`docs/FIGURE_GUIDE.md`](docs/FIGURE_GUIDE.md) for suggested manuscript placement and captions.

## Repository layout

```text
RA-CoFuzz-public-v1.0.4/
├── GPTFuzz-master/          # original framework source retained by hash
├── extensions_q20/          # Q20 experiment runners and compatibility layer
├── pair_official_autodl/    # PAIR adapter and scripts
├── JailbreakingLLMs-official/
├── analysis/                # safe aggregation and figure generation
├── results/                 # aggregate JSON and Markdown tables only
├── figures/                 # PDF, SVG, and PNG figures
├── docs/                    # appendix-equivalent protocol and method record
├── provenance/              # recovered commits and model-config hashes
├── archive/                 # inactive historical snapshots, never imported at runtime
└── tests/                   # public-result validation tests
```

## Paper and repository map

The manuscript intentionally has no separate appendix. Start with:

- [`docs/PAPER_REPOSITORY_MAP.md`](docs/PAPER_REPOSITORY_MAP.md) for the exact mapping from manuscript sections, figures, and tables to repository artifacts;
- [`docs/METHOD_AND_JUDGE_PROTOCOL.md`](docs/METHOD_AND_JUDGE_PROTOCOL.md) for refusal labels, weak screening, selective StrongJudge review, scoring, and selector details;
- [`docs/ENVIRONMENT_LOCK.md`](docs/ENVIRONMENT_LOCK.md) for the recorded software environment and external dependency checks;
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for the frozen Q20/B200 matrix and the non-duplicated 144-cell accounting.

Only `GPTFuzz-master/gptfuzzer/fuzzer/core.py`, `selection.py`, and
`GPTFuzz-master/gptfuzzer/utils/ra_judge.py` are authoritative for the RA-CoFuzz
runtime. Historical pre-fix snapshots are isolated under `archive/` and are not
on the Python import path.

## Quick verification

Create an analysis environment and regenerate the figures:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
make verify-public
make verify-extensions
make figures
make verify-release
```

The public aggregate can be audited without access to raw conversations:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s extensions_q20 -p 'test_*.py'
```

The aggregation interface can validate a newly generated or independently
retained private results directory:

```bash
python analysis/aggregate_results.py \
  --raw-root /path/to/extension_outputs \
  --output results/aggregate/summary.json \
  --table-dir results/tables
```

The aggregation script emits only numerical fields and hashed internal identifiers; it never writes prompts, questions, or model responses into the public result file. The command above documents the expected private result schema. It cannot reconstruct the released numerical aggregate unless the corresponding original `extension_outputs` tree is available.

## Documentation

- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Method and judge protocol](docs/METHOD_AND_JUDGE_PROTOCOL.md)
- [Paper–repository map](docs/PAPER_REPOSITORY_MAP.md)
- [Recorded environment](docs/ENVIRONMENT_LOCK.md)
- [Result interpretation](docs/RESULTS.md)
- [Baseline notes](docs/BASELINES.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data and release policy](docs/DATA_AND_RELEASE_POLICY.md)
- [Figure guide](docs/FIGURE_GUIDE.md)
- [GitHub release checklist](docs/GITHUB_RELEASE_CHECKLIST.md)

## License and third-party code

RA-CoFuzz release-specific code is distributed under the MIT License. Bundled upstream components retain their own license and attribution files. See [`NOTICE.md`](NOTICE.md) before redistribution.
