# Result interpretation

## Main comparison

RA-CoFuzz achieved the highest mean question-level Label-2 ASR in the main setting (81.7%). Strict GPTFuzzer reached 61.7%, PAIR 60.0%, TAP 46.7%, ReNeLLM 11.7%, and DeepInception 0.0%. RA-CoFuzz also had the highest response-level Label-2 ASR (18.5%) in this setting.

These means should be read with their standard deviations across three repeated runs. RA-CoFuzz's main-setting question-level standard deviation was 20.2 percentage points, so the claim supported by this experiment is stronger average coverage—not uniform superiority in every run.

## Cross-dataset behavior

RA-CoFuzz led the mean question-level Label-2 ASR on all three Llama-3.2-3B-Instruct dataset conditions: 81.7% on GPTFuzzer, 51.7% on AdvBench, and 55.0% on JailbreakBench. The smaller margins and larger variability on the latter two datasets should be reported alongside the means.

## Cross-model behavior

Performance changes substantially with the target model. RA-CoFuzz reached 98.3% mean question-level Label-2 ASR on both Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. Several baselines were also strong on the Qwen targets; PAIR reached 100% on the 3B and 7B conditions. The cross-model result therefore supports broad method effectiveness, while also showing that no single method dominates every target condition.

## Ablation evidence

Removing DP reduced mean question-level Label-2 ASR from 81.7% to 56.7%. Removing MCTS reduced it to 60.0%, and removing Elite reduced it to 75.0%. In the feedback family, removing the Label-1 reward reduced question-level ASR to 70.0%, while removing the online judge reduced it to 68.3%. Full online review consumed 200 judge calls in every run but did not improve question-level ASR over selective review.

## Baseline-specific caveat

DeepInception uses a single transformed query per question in this implementation and records 20 responses rather than approaching the 200-call ceiling. Its main-setting zero response-level Label-2 ASR is a result for this frozen target/configuration, not a general statement about the method. Use the effectiveness–cost plot and per-condition tables when discussing it.

## Recommended reporting language

Use “mean across three repeated runs” and report `mean ± sample standard deviation`.
Do not call all runs “independent seeds”; the method-specific historical seed
behavior is documented in `EXPERIMENT_PROTOCOL.md`. Avoid describing the
20-question subsets as population-level estimates. Treat the cross-dataset and
cross-model matrices as breadth and sensitivity evidence rather than as a
replacement for a larger benchmark.
