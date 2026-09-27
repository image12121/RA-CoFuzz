# Paper and repository map

The manuscript does not use a separate appendix. Extended implementation and
reproducibility material is maintained in this versioned repository release.

## Manuscript mapping

| Manuscript item | Repository record |
|---|---|
| Threat model and task definition | `docs/EXPERIMENT_PROTOCOL.md` |
| Three response labels and strict Label-2 endpoint | `docs/METHOD_AND_JUDGE_PROTOCOL.md` |
| Question-level and response-level ASR definitions | `docs/EXPERIMENT_PROTOCOL.md`, `results/aggregate/summary.json` |
| Weak screening and selective StrongJudge | `docs/METHOD_AND_JUDGE_PROTOCOL.md`, `GPTFuzz-master/gptfuzzer/utils/ra_judge.py` |
| MCTS, DP, Elite, and hybrid selection | `docs/METHOD_AND_JUDGE_PROTOCOL.md`, `provenance/reported_hybrid_config.env`, `GPTFuzz-master/gptfuzzer/fuzzer/selection.py` |
| MCTS reward, UCB, and path-update equations | `docs/METHOD_AND_JUDGE_PROTOCOL.md`, `GPTFuzz-master/gptfuzzer/fuzzer/selection.py` |
| Complete online search loop | `GPTFuzz-master/gptfuzzer/fuzzer/core.py` |
| Unified Q20/B200 protocol | `extensions_q20/protocol.json`, `docs/EXPERIMENT_PROTOCOL.md` |
| Baseline adaptations and stopping rules | `docs/BASELINES.md`, `extensions_q20/README.md` |
| 144-cell accounting | `docs/EXPERIMENT_PROTOCOL.md`, `results/aggregate/summary.json` |
| Main and extended result tables | `results/tables/`, `docs/RESULTS.md` |
| Figure sources and caption guidance | `figures/`, `docs/FIGURE_GUIDE.md`, `analysis/make_figures.py` |
| Environment and external dependency freeze | `docs/ENVIRONMENT_LOCK.md`, `provenance/q20_runtime_audit_public.json`, `extensions_q20/runtime_boundary_pass.json` |
| Data provenance and release boundary | `docs/DATA_AND_RELEASE_POLICY.md`, `extensions_q20/reference_provenance.json` |

## Figure filename mapping

The current release stores every visual panel as a standalone file so the
manuscript can place one figure, one caption, and one explanatory paragraph at
a time. Use this map when preparing the paper:

| Manuscript figure | Repository basename |
|---|---|
| Figure 1 | `fig01_framework_overview` |
| Figure 2(a) | `fig02a_main_question_asr` |
| Figure 2(b) | `fig02b_main_response_rates` |
| Figure 3 | `fig10_response_composition` |
| Figure 4(a) | `fig03a_cross_dataset_question_asr` |
| Figure 4(b) | `fig03b_cross_dataset_response_asr` |
| Figure 5(a) | `fig04a_cross_model_question_asr` |
| Figure 5(b) | `fig04b_cross_model_response_asr` |
| Figure 6(a) | `fig05a_selection_question_asr` |
| Figure 6(b) | `fig05b_selection_response_asr` |
| Figure 7(a) | `fig06a_feedback_question_asr` |
| Figure 7(b) | `fig06b_feedback_response_asr` |
| Figure 7(c) | `fig06c_feedback_judge_calls` |
| Figure 8(a) | `fig07a_efficiency_target_calls` |
| Figure 8(b) | `fig07b_efficiency_runtime` |
| Figure 9 | `fig08_evaluator_alignment` |
| Figure 10(a) | `fig09a_question_stability` |
| Figure 10(b) | `fig09b_response_stability` |

## Manuscript availability paragraph

The following text can be adapted after the final repository URL and archival
DOI are known:

> Code, frozen experiment protocols, content-free aggregate results, tests, and figure-generation scripts are available in the versioned RA-CoFuzz repository. The repository also contains the appendix-equivalent implementation and evaluator specifications referenced in the paper. Raw model conversations, API credentials, model checkpoints, and the historical private per-response result tree are not included. The public package supports aggregate consistency checks, table and figure regeneration, and fresh authorized execution through the documented interfaces; it does not claim byte-for-byte reconstruction of the released aggregate from historical raw records.

Before submission, replace the repository placeholder in the manuscript with a
permanent release URL or DOI and verify that its version matches `VERSION`.
