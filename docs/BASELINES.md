# Baseline implementations

| Method | Integration | Stopping/query behavior |
|---|---|---|
| Strict GPTFuzzer | original GPTFuzzer path with the frozen strict configuration | runs to the 200-call cap |
| PAIR | official-style adapter in `pair_official_autodl` | native early stopping; call count may be below 200 |
| TAP | EasyJailbreak-compatible bridge with validated parser/candidate recovery | native tree search bounded by the shared cap |
| ReNeLLM | EasyJailbreak-compatible bridge with per-question iterative flow | native success or per-question cap |
| DeepInception | EasyJailbreak-compatible bridge | one transformed target query per question |

The repository keeps compatibility logic explicit in `extensions_q20` instead of editing upstream sources in place. Runtime-boundary and regression tests cover constructor compatibility, response accounting, seed mapping, candidate generation, completed-cell preservation, and evaluation completeness.

For the paper, baseline descriptions can be limited to one sentence each plus the original citation. Implementation-specific deviations that affect query counts or stopping should remain in the experimental setup or appendix-equivalent repository documentation.

