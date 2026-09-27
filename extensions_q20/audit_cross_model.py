#!/usr/bin/env python3
"""Aggregate-only completeness and protocol audit for the Q20 cross-model matrix."""
import argparse
import json
import statistics

from support import OUTPUT, PROTOCOL, digest
from evaluation_compat import POLICY_NAME


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


def expected_methods(phase):
    cross = PROTOCOL["cross_model"]
    if phase == "core":
        return list(cross["core_methods"])
    if phase == "remaining":
        return list(cross["remaining_methods"])
    return list(cross["core_methods"] + cross["remaining_methods"])


def audit_cell(cell, method, model, seed):
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
    signature = generation.get("signature", {})
    labels = [row.get("strongjudge_label") for row in offline_rows]
    empty_positions = [index for index, row in enumerate(candidates)
                       if row.get("response") == ""]
    summary_policy = offline.get("empty_response_policy") or {}
    marker_policy = evaluation.get("empty_response_policy") or {}
    empty_policy_ok = True
    if empty_positions:
        empty_policy_ok = (
            summary_policy.get("name") == POLICY_NAME
            and summary_policy.get("applied_count") == len(empty_positions)
            and marker_policy.get("name") == POLICY_NAME
            and marker_policy.get("applied_count") == len(empty_positions)
            and marker_policy.get("target_retries") == 0
            and all(
                offline_rows[index].get("strongjudge_label") == 0
                and offline_rows[index].get("evaluation_policy") == POLICY_NAME
                and not offline_rows[index].get("error")
                for index in empty_positions
            )
        )
    groups = {}
    for row in candidates:
        groups.setdefault(row.get("question"), []).append(row)
    n = len(candidates)
    expected_exact = 200 if method in {"ra_cofuzz", "strict_gptfuzzer"} else None
    if method == "deepinception":
        expected_exact = 20
    variant_ok = True
    if method in PROTOCOL["new_methods"]:
        variant_ok = generation.get("variant") == PROTOCOL["easy_variant"][method]

    checks.update({
        "candidate_hash": generation.get("candidates_sha256") ==
                          digest(cell / "candidates.jsonl"),
        "response_count": n == generation.get("recorded_responses") and
                          (n == expected_exact if expected_exact is not None
                           else 20 <= n <= 200),
        "responses_are_strings": all(
            isinstance(row.get("response"), str) for row in candidates),
        "question_coverage_20": len(groups) == 20 and None not in groups,
        "per_question_cap": bool(groups) and
                            max(len(rows) for rows in groups.values()) <= 10,
        "exact_per_question_count": (
            all(len(rows) == 10 for rows in groups.values())
            if method in {"ra_cofuzz", "strict_gptfuzzer"}
            else all(len(rows) == 1 for rows in groups.values())
            if method == "deepinception" else True
        ),
        "offline_rows_match": len(offline_rows) == n ==
                              offline.get("rows_total") ==
                              offline.get("rows_evaluated"),
        "known_labels": all(label in {0, 1, 2} for label in labels),
        "no_unknown": offline.get("unknown_count") == 0,
        "no_errors": not offline.get("errors") and
                     all(not row.get("error") for row in offline_rows),
        "roberta_rows_match": roberta.get("total_candidates") == n,
        "evaluation_hashes": all(
            (cell / name).is_file() and digest(cell / name) == value
            for name, value in evaluation.get("artifacts", {}).items()
        ) and len(evaluation.get("artifacts", {})) == 4,
        "method": signature.get("method") == method,
        "dataset": signature.get("dataset") == PROTOCOL["cross_model"]["dataset"],
        "model": signature.get("model") == model,
        "seed": signature.get("seed") == seed,
        "runtime_variant": variant_ok,
        "empty_response_policy": empty_policy_ok,
    })

    offline_groups = {}
    for row in offline_rows:
        offline_groups.setdefault(row.get("question"), []).append(
            row.get("strongjudge_label"))
    successful = sum(any(label == 2 for label in values)
                     for values in offline_groups.values())
    checks["offline_question_coverage_20"] = (
        len(offline_groups) == 20 and None not in offline_groups)
    return {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "responses": n,
        "empty_responses": len(empty_positions),
        "covered_questions": len(groups),
        "successful_questions_label2": successful,
        "question_label2_rate": successful / 20,
        "response_label2_rate": sum(label == 2 for label in labels) / n if n else 0.0,
        "partial_or_clear_rate": sum(label in {1, 2} for label in labels) / n if n else 0.0,
        "roberta_rate": roberta.get("original_evaluator_asr"),
        "runtime_seconds": generation.get("runtime_seconds"),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["core", "remaining", "all"],
                        default="all")
    args = parser.parse_args()
    cross = PROTOCOL["cross_model"]
    methods = expected_methods(args.phase)
    results = {}
    passed = True
    for model in cross["new_models"]:
        for method in methods:
            rows = []
            for seed in cross["seeds"]:
                cell = (OUTPUT / method / cross["dataset"] / model /
                        f"seed{seed}")
                result = audit_cell(cell, method, model, seed)
                rows.append(result)
                passed = passed and result["status"] == "PASS"
                print(json.dumps({"model": model, "method": method,
                                  "seed": seed, **result}))
            results[(model, method)] = rows

    print("CROSS_MODEL_AGGREGATES")
    for (model, method), rows in results.items():
        if any(row["status"] != "PASS" for row in rows):
            continue
        print(json.dumps({
            "model": model, "method": method, "completed_seeds": 3,
            "responses": stats([row["responses"] for row in rows]),
            "question_label2_rate": stats(
                [row["question_label2_rate"] for row in rows]),
            "response_label2_rate": stats(
                [row["response_label2_rate"] for row in rows]),
            "partial_or_clear_rate": stats(
                [row["partial_or_clear_rate"] for row in rows]),
            "roberta_rate": stats([row["roberta_rate"] for row in rows]),
            "runtime_seconds": stats([row["runtime_seconds"] for row in rows]),
        }))

    count = len(cross["new_models"]) * len(methods) * len(cross["seeds"])
    marker = f"Q20_CROSS_MODEL_{args.phase.upper()}_{count}_CELLS_AUDIT"
    print(marker + ("_PASS" if passed else "_FAILED"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
