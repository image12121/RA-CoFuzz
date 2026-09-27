# Data notes

The frozen benchmark subsets live under `GPTFuzz-master/datasets/questions/`. Each dataset has 20 questions selected by the recorded subset procedure. The `_with_targets.csv` variants supply the reference column required by PAIR/TAP-compatible interfaces; their provenance and exact-match checks are recorded in `extensions_q20/reference_provenance.json` and tested by `extensions_q20/test_reference_data.py`.

Do not alter the subsets after running experiments. A changed row order changes the seed-to-question mapping and invalidates comparisons with the included aggregate results.

