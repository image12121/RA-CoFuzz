#!/usr/bin/env python3
"""Aggregate-only completeness and treatment audit for feedback ablations."""
import json
from pathlib import Path
import statistics

from audit_selection_ablations import audit_cell as audit_common_cell
from support import OUTPUT, PROTOCOL, digest


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def audit_feedback_cell(cell, variant, seed):
    result = audit_common_cell(cell)
    checks = dict(result.get("checks", {}))
    generation_path = cell / "generation.done.json"
    native_path = cell / "native_summary.json"
    checks.update({
        "native_summary_file": native_path.is_file(),
        "generation_marker_file": generation_path.is_file(),
    })
    if not all(checks.values()):
        return {**result, "status": "FAILED", "checks": checks}

    generation = load_json(generation_path)
    native = load_json(native_path)
    signature = generation.get("signature", {})
    control = generation.get("online_feedback_control", {})
    expected = PROTOCOL["feedback_ablations"][variant]
    calls = control.get("strong_judge_calls")
    call_policy = expected["expected_online_calls"]
    calls_match = (
        (call_policy == "exactly_0" and calls == 0)
        or (call_policy == "exactly_200" and calls == 200)
        or (call_policy == "selective_0_to_200" and
            isinstance(calls, int) and 0 <= calls <= 200)
    )
    checks.update({
        "feedback_ablation_name": signature.get("feedback_ablation") == variant,
        "feedback_ablation_config": signature.get("feedback_ablation_config") == expected,
        "method": signature.get("method") == "ra_cofuzz",
        "dataset": signature.get("dataset") == "gptfuzzer",
        "model": signature.get("model") == "llama32_3b",
        "seed": signature.get("seed") == seed,
        "native_candidates_200": native.get("total_candidates") == 200 ==
                                 control.get("total_candidates"),
        "online_calls_match_policy": calls_match,
        "online_calls_match_native": calls == native.get("strong_judge_calls"),
        "online_mode": control.get("strong_judge_in_loop") ==
                       expected["strong_judge_in_loop"],
        "force_review_mode": control.get("force_review") == expected["force_review"],
        "native_summary_hash": control.get("native_summary_sha256") == digest(native_path),
    })

    label_control = generation.get("feedback_ablation_control")
    if variant == "wo_label1_reward":
        transforms = label_control.get("transforms", []) if isinstance(label_control, dict) else []
        checks.update({
            "label1_policy": isinstance(label_control, dict) and
                             label_control.get("policy") ==
                             "label1_to_unjudged_for_reward_and_selection_v1",
            "raw_labels_preserved": isinstance(label_control, dict) and
                                    label_control.get("raw_labels_preserved") is True,
            "seven_runtime_transforms": len(transforms) == 7,
            "transforms_changed": len(transforms) == 7 and all(
                row.get("source_sha256") != row.get("transformed_sha256")
                for row in transforms),
            "remap_counts_recorded": isinstance(
                label_control.get("core_label1_remaps"), int) and
                label_control.get("core_label1_remaps") >= 0 and
                isinstance(label_control.get("selection_label1_remaps"), int) and
                label_control.get("selection_label1_remaps") >= 0,
        })
    else:
        checks["no_label1_runtime_transform"] = label_control is None

    result["online_strong_judge_calls"] = calls
    result["online_strong_judge_call_rate"] = calls / 200
    result["status"] = "PASS" if all(checks.values()) else "FAILED"
    result["checks"] = checks
    return result


def safe_result(variant, seed, result):
    return {"feedback_ablation": variant, "seed": seed, **result}


def main():
    passed = True
    all_results = {}
    for variant in PROTOCOL["feedback_ablations"]:
        rows = []
        for seed in PROTOCOL["seeds"]:
            cell = (OUTPUT / "feedback_ablations" / variant / "gptfuzzer" /
                    "llama32_3b" / f"seed{seed}")
            result = audit_feedback_cell(cell, variant, seed)
            rows.append(result)
            passed = passed and result["status"] == "PASS"
            print(json.dumps(safe_result(variant, seed, result)))
        all_results[variant] = rows

    print("FEEDBACK_ABLATION_AGGREGATES")
    for variant, rows in all_results.items():
        if any(row["status"] != "PASS" for row in rows):
            continue
        print(json.dumps({
            "feedback_ablation": variant,
            "completed_seeds": 3,
            "responses": stats([row["responses"] for row in rows]),
            "question_label2_rate": stats([row["question_label2_rate"] for row in rows]),
            "response_label2_rate": stats([row["response_label2_rate"] for row in rows]),
            "partial_or_clear_rate": stats([row["partial_or_clear_rate"] for row in rows]),
            "roberta_rate": stats([row["roberta_rate"] for row in rows]),
            "online_strong_judge_calls": stats(
                [row["online_strong_judge_calls"] for row in rows]),
            "runtime_seconds": stats([row["runtime_seconds"] for row in rows]),
        }))

    full_rows = []
    for seed in PROTOCOL["seeds"]:
        cell = OUTPUT / "ra_cofuzz" / "gptfuzzer" / "llama32_3b" / f"seed{seed}"
        row = audit_common_cell(cell)
        full_rows.append(row)
        passed = passed and row["status"] == "PASS"
    if all(row["status"] == "PASS" for row in full_rows):
        print("FULL_REFERENCE_AGGREGATE")
        print(json.dumps({
            "feedback_ablation": "full_ra_cofuzz_existing_not_rerun",
            "completed_seeds": 3,
            "responses": stats([row["responses"] for row in full_rows]),
            "question_label2_rate": stats([row["question_label2_rate"] for row in full_rows]),
            "response_label2_rate": stats([row["response_label2_rate"] for row in full_rows]),
            "partial_or_clear_rate": stats([row["partial_or_clear_rate"] for row in full_rows]),
            "roberta_rate": stats([row["roberta_rate"] for row in full_rows]),
            "runtime_seconds": stats([row["runtime_seconds"] for row in full_rows]),
        }))

    print("Q20_FEEDBACK_ABLATION_9_CELLS_AUDIT_PASS" if passed else
          "Q20_FEEDBACK_ABLATION_9_CELLS_AUDIT_FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
