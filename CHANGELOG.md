# Changelog

## 1.0.4 - 2026-09-27

- added explicit question-level and response-level Label-2 ASR definitions,
  including their different denominators and legacy JSON field mapping;
- documented the active MCTS shaped reward, UCB score, and path-update rule
  with the exact released coefficients;
- aligned model identifiers, StrongJudge configuration, table headings, and
  the non-duplicated 144-cell accounting with the final manuscript;
- exported every multi-panel result as a standalone figure so figures and
  captions can be placed independently in the manuscript;
- strengthened public-result and release-hygiene tests.
- clarified that the retained public materials support aggregate consistency
  checks and figure regeneration, but do not include the historical private
  per-response tree needed to rebuild the released aggregate;
- removed project-level author and citation placeholder metadata.

No active algorithm, experimental cell, reported result, or Figure 1 artwork
was changed.

## 1.0.3 - 2026-09-26

- aligned paper-facing figure organization with the manuscript requirement
  that panels be supplied as standalone images;
- retained the existing framework artwork and aggregate values.

## 1.0.2 - 2026-09-24

- incorporated the recovered Q20 runtime audit, including exact EasyJailbreak
  and PAIR commits, Python 3.12.3, RTX 4090, and target `config.json` hashes;
- corrected the historical randomness description: public `100/200/300`
  values are stable run labels, not uniformly independent seeds for every
  legacy method;
- clarified that the standalone six-term RAES scoring path was disabled in
  the final reported Hybrid configuration;
- removed generated Python caches from the distribution and strengthened
  release-hygiene tests.

No aggregate value, figure value, or active algorithm file was changed.

## 1.0.1 - 2026-09-19

- added appendix-equivalent method, evaluator, environment, and paper mapping documentation;
- froze the recorded DeepSeek `deepseek-chat` judge settings without exposing credentials;
- clarified non-duplicated 144-cell accounting and manuscript figure filename mapping;
- moved inactive pre-fix source snapshots out of the active import tree;
- fixed manifest generation so Python caches and notebook checkpoints are excluded;
- added release-hygiene tests and a single `make verify-release` gate;
- updated release metadata and the GitHub publication checklist.

No experiment result, aggregate metric, figure value, or active algorithm file
was changed. `results/aggregate/summary.json` therefore retains its original
v1.0.0 result-payload label.

## 1.0.0 - 2026-09-08

- initial public package assembled from the completed Q20 experimental campaign.
