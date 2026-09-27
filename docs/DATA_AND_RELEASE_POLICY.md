# Data and release policy

## Included

- frozen 20-question benchmark subsets needed by the experiment interfaces;
- reference-column variants and provenance metadata;
- code, configuration, audit metadata, and unit tests;
- numerical aggregate results for all 144 completed cells;
- publication-ready figures and Markdown tables.

## Excluded

- generated prompts and target-model responses;
- per-response StrongJudge explanations;
- raw JSONL experiment output;
- API credentials and `.env` files;
- model checkpoints, tokenizer caches, and third-party package caches;
- process identifiers, logs, temporary backups, and machine-specific absolute paths.

This boundary makes the public repository useful for code inspection, aggregate consistency checks, and figure reproduction without redistributing generated conversations or private infrastructure state. The original private raw tree used for the published aggregate is not available in the retained release materials. `analysis/aggregate_results.py` remains available to validate newly generated or independently retained outputs that follow the documented schema, but the repository does not claim byte-for-byte reconstruction of the released aggregate from historical raw records.

Benchmark subsets may contain content intended solely for safety evaluation. Do not use them to target real systems without permission, and do not expose benchmark text in issue reports or screenshots when a hashed row identifier is sufficient.
