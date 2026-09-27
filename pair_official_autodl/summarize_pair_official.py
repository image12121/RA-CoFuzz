#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_sd(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs": values,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()
    root = Path(args.results_dir)

    run_summaries = sorted(glob.glob(str(root / "summary_pair_official_*_run[123].json")))
    # Exclude evaluator summaries selected by the broad pattern.
    run_summaries = [p for p in run_summaries if not p.endswith(("_strongjudge.json", "_roberta.json"))]
    if len(run_summaries) != 3:
        raise RuntimeError(f"Expected 3 PAIR run summaries, found {len(run_summaries)}")

    pair_question_asr = []
    target_calls = []
    response_sj_asr = []
    question_sj_asr = []
    roberta_asr = []

    for run, summary_path in enumerate(run_summaries, start=1):
        s = read_json(summary_path)
        pair_question_asr.append(s["pair_question_asr"])
        target_calls.append(s["target_calls"])

        sj_path = root / f"eval_pair_official_llama32_q20_maxb200_run{run}_strongjudge.jsonl"
        sj_rows = read_jsonl(sj_path)
        response_sj_asr.append(sum(r.get("strongjudge_label") == 2 for r in sj_rows) / len(sj_rows))
        by_question = {}
        for row in sj_rows:
            by_question.setdefault(row["question"], []).append(row.get("strongjudge_label"))
        question_sj_asr.append(
            sum(any(label == 2 for label in labels) for labels in by_question.values()) / len(by_question)
        )

        rob_path = root / f"results_pair_official_llama32_q20_maxb200_run{run}_roberta.jsonl"
        rob_rows = read_jsonl(rob_path)
        roberta_asr.append(
            sum(r.get("original_evaluator_label") == 1 for r in rob_rows) / len(rob_rows)
        )

    result = {
        "pair_internal_question_asr": mean_sd(pair_question_asr),
        "strongjudge_response_asr": mean_sd(response_sj_asr),
        "strongjudge_question_success_at_max10": mean_sd(question_sj_asr),
        "roberta_response_asr": mean_sd(roberta_asr),
        "target_calls": mean_sd(target_calls),
    }
    out = root / "summary_pair_official_3runs_meanstd.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
