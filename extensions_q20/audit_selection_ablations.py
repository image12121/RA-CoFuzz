#!/usr/bin/env python3
"""Safe, aggregate-only completeness audit for the nine selection ablations."""
import json
from pathlib import Path
import statistics

from support import OUTPUT, PROTOCOL, digest


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def audit_cell(cell, expected_ablation=None, expected_seed=None):
    required = [
        "candidates.jsonl", "generation.done.json", "offline.jsonl",
        "offline_summary.json", "roberta.jsonl", "roberta_summary.json",
        "evaluation.done.json",
    ]
    checks = {f"file:{name}": (cell / name).is_file() for name in required}
    if not all(checks.values()):
        return {"status": "FAILED", "checks": checks}

    candidates = load_jsonl(cell / "candidates.jsonl")
    generation = load_json(cell / "generation.done.json")
    offline_rows = load_jsonl(cell / "offline.jsonl")
    offline = load_json(cell / "offline_summary.json")
    roberta = load_json(cell / "roberta_summary.json")
    evaluation = load_json(cell / "evaluation.done.json")
    labels = [row.get("strongjudge_label") for row in offline_rows]
    question_groups = {}
    for row in offline_rows:
        question_groups.setdefault(row.get("question"), []).append(
            row.get("strongjudge_label"))

    checks.update({
        "candidate_hash": generation.get("candidates_sha256") == digest(cell / "candidates.jsonl"),
        "response_count_200": len(candidates) == 200 == generation.get("recorded_responses"),
        "responses_are_strings": all(isinstance(row.get("response"), str) for row in candidates),
        "offline_rows_200": len(offline_rows) == 200 == offline.get("rows_total") == offline.get("rows_evaluated"),
        "known_labels": all(label in {0, 1, 2} for label in labels),
        "no_unknown": offline.get("unknown_count") == 0,
        "no_errors": not offline.get("errors") and all(not row.get("error") for row in offline_rows),
        "roberta_rows_200": roberta.get("total_candidates") == 200,
        "question_coverage_20": len(question_groups) == 20,
        "evaluation_hashes": all(
            (cell / name).is_file() and digest(cell / name) == value
            for name, value in evaluation.get("artifacts", {}).items()
        ) and len(evaluation.get("artifacts", {})) == 4,
    })
    signature = generation.get("signature", {})
    if expected_ablation is not None:
        expected = PROTOCOL["selection_ablations"][expected_ablation]
        control = generation.get("selection_ablation_control", {})
        counts = control.get("selection_counts", {})
        checks.update({
            "ablation_name": signature.get("selection_ablation") == expected_ablation,
            "ablation_config": signature.get("selection_ablation_config") == expected,
            "method": signature.get("method") == "ra_cofuzz",
            "dataset": signature.get("dataset") == "gptfuzzer",
            "model": signature.get("model") == "llama32_3b",
            "seed": signature.get("seed") == expected_seed,
            "disabled_branch": control.get("disabled_branch") == expected["disabled_branch"],
            "disabled_branch_not_attempted": control.get("disabled_branch_attempts") == 0,
            "disabled_branch_not_selected": control.get("disabled_branch_selections") == 0,
            "selection_steps_recorded": (
                isinstance(control.get("hybrid_steps"), int) and
                control.get("hybrid_steps", 0) > 0 and
                sum(counts.get(name, 0) for name in ("mcts", "dp", "elite")) ==
                control.get("hybrid_steps")
            ),
        })

    successful = sum(any(label == 2 for label in values)
                     for values in question_groups.values())
    n = len(offline_rows)
    return {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "responses": len(candidates),
        "covered_questions": len(question_groups),
        "successful_questions_label2": successful,
        "question_label2_rate": successful / 20,
        "response_label2_rate": sum(label == 2 for label in labels) / n,
        "partial_or_clear_rate": sum(label in {1, 2} for label in labels) / n,
        "roberta_rate": roberta.get("original_evaluator_asr"),
        "runtime_seconds": generation.get("runtime_seconds"),
        "checks": checks,
    }


def safe_cell_result(ablation, seed, result):
    keep = {key: value for key, value in result.items() if key != "checks"}
    return {"selection_ablation": ablation, "seed": seed, **keep,
            "checks": result.get("checks", {})}


def main():
    all_results = {}
    passed = True
    for ablation in PROTOCOL["selection_ablations"]:
        rows = []
        for seed in PROTOCOL["seeds"]:
            cell = (OUTPUT / "selection_ablations" / ablation / "gptfuzzer" /
                    "llama32_3b" / f"seed{seed}")
            result = audit_cell(cell, ablation, seed)
            rows.append(result)
            passed = passed and result["status"] == "PASS"
            print(json.dumps(safe_cell_result(ablation, seed, result)))
        all_results[ablation] = rows

    print("SELECTION_ABLATION_AGGREGATES")
    for ablation, rows in all_results.items():
        if any(row["status"] != "PASS" for row in rows):
            continue
        aggregate = {
            "selection_ablation": ablation,
            "completed_seeds": len(rows),
            "responses": stats([row["responses"] for row in rows]),
            "question_label2_rate": stats([row["question_label2_rate"] for row in rows]),
            "response_label2_rate": stats([row["response_label2_rate"] for row in rows]),
            "partial_or_clear_rate": stats([row["partial_or_clear_rate"] for row in rows]),
            "roberta_rate": stats([row["roberta_rate"] for row in rows]),
            "runtime_seconds": stats([row["runtime_seconds"] for row in rows]),
        }
        print(json.dumps(aggregate))

    full_rows = []
    for seed in PROTOCOL["seeds"]:
        cell = OUTPUT / "ra_cofuzz" / "gptfuzzer" / "llama32_3b" / f"seed{seed}"
        result = audit_cell(cell)
        full_rows.append(result)
        passed = passed and result["status"] == "PASS"
    if all(row["status"] == "PASS" for row in full_rows):
        print("FULL_REFERENCE_AGGREGATE")
        print(json.dumps({
            "selection_ablation": "full_ra_cofuzz_existing_not_rerun",
            "completed_seeds": 3,
            "responses": stats([row["responses"] for row in full_rows]),
            "question_label2_rate": stats([row["question_label2_rate"] for row in full_rows]),
            "response_label2_rate": stats([row["response_label2_rate"] for row in full_rows]),
            "partial_or_clear_rate": stats([row["partial_or_clear_rate"] for row in full_rows]),
            "roberta_rate": stats([row["roberta_rate"] for row in full_rows]),
            "runtime_seconds": stats([row["runtime_seconds"] for row in full_rows]),
        }))

    print("Q20_SELECTION_ABLATION_9_CELLS_AUDIT_PASS" if passed else
          "Q20_SELECTION_ABLATION_9_CELLS_AUDIT_FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
