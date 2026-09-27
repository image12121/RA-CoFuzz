# Figure guide

All figures are provided as vector PDF/SVG and 300-dpi PNG. PDF/SVG are the
camera-ready masters; PNG is intended for GitHub preview and Word review.
Every former multi-panel result is exported as a standalone image so each
figure can be placed with its own caption and explanatory paragraph.

| Repository basename | Main purpose |
|---|---|
| `fig01_framework_overview` | RA-CoFuzz search, refusal-aware feedback, and state-update loop |
| `fig02a_main_question_asr` | Main six-method question-level Label-2 ASR |
| `fig02b_main_response_rates` | Main response-level Label-2, Label-(1+2), and RoBERTa rates |
| `fig03a_cross_dataset_question_asr` | Question-level ASR across three datasets |
| `fig03b_cross_dataset_response_asr` | Response-level ASR across three datasets |
| `fig04a_cross_model_question_asr` | Question-level ASR across five target models |
| `fig04b_cross_model_response_asr` | Response-level ASR across five target models |
| `fig05a_selection_question_asr` | Selection-component effects on question-level ASR |
| `fig05b_selection_response_asr` | Selection-component effects on response-level ASR |
| `fig06a_feedback_question_asr` | Feedback-component effects on question-level ASR |
| `fig06b_feedback_response_asr` | Feedback-component effects on response-level ASR |
| `fig06c_feedback_judge_calls` | Online StrongJudge cost of feedback variants |
| `fig07a_efficiency_target_calls` | Effectiveness versus target-query use |
| `fig07b_efficiency_runtime` | Effectiveness versus wall-clock runtime |
| `fig08_evaluator_alignment` | StrongJudge–RoBERTa alignment across 42 aggregates |
| `fig09a_question_stability` | Question-level variation across released runs |
| `fig09b_response_stability` | Response-level variation across released runs |
| `fig10_response_composition` | Label-0/1/2 response composition |

## Caption guidance

**Framework.** State that the online loop selects a parent from the dynamic
template pool, applies one of five mutation operators, assembles it with the
evaluation query, and queries the target model. Weak screening and selective
StrongJudge review provide three-level feedback to MCTS, DP, and Elite state
updates. Full offline StrongJudge and RoBERTa evaluation is post hoc.

**Comparisons and heatmaps.** State the metric level explicitly. Question-level
ASR counts a question once if any response is Label 2; response-level ASR uses
all recorded responses as its denominator. Values are means across three
repeated runs; where shown, error bars are sample standard deviations.

**Ablations.** Identify the GPTFuzzer/Llama-3.2-3B-Instruct condition and state
which component is removed. Discuss online judge calls separately from ASR so
effectiveness and evaluation cost are not conflated.

**Stability.** Call 100/200/300 released run labels rather than uniformly
independent random seeds; see `docs/EXPERIMENT_PROTOCOL.md`.

Do not merge these files into composite images in the manuscript. The
standalone organization implements the paper requirement that figures, tables,
and captions remain visually separate.
