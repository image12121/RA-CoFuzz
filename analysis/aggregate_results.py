#!/usr/bin/env python3
"""Build public, content-free aggregate results from private experiment outputs.

The input tree may contain prompts and model responses.  This script reads them
only to group records by question and never writes any prompt or response text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHOD_ORDER = [
    "ra_cofuzz",
    "strict_gptfuzzer",
    "pair",
    "tap",
    "renellm",
    "deepinception",
]
DATASET_ORDER = ["gptfuzzer", "advbench", "jailbreakbench"]
MODEL_ORDER = [
    "llama32_3b",
    "qwen25_1_5b",
    "qwen25_3b",
    "qwen25_7b",
    "vicuna_7b",
]
SEED_ORDER = [100, 200, 300]
SELECTION_VARIANTS = ["full", "hybrid_wo_dp", "hybrid_wo_elite", "hybrid_wo_mcts"]
FEEDBACK_VARIANTS = [
    "full",
    "wo_label1_reward",
    "wo_online_strongjudge_feedback",
    "full_online_review",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(item)
    return rows


def question_key(row: dict[str, Any]) -> str:
    value = row.get("question", row.get("original_prompt"))
    if not isinstance(value, str) or not value:
        raise ValueError("record has no non-empty question field")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rate(value: int, total: int) -> float:
    return value / total if total else 0.0


def mean_sd(values: Iterable[float | int]) -> dict[str, float]:
    xs = [float(x) for x in values]
    if not xs:
        return {"mean": 0.0, "sample_sd": 0.0}
    return {
        "mean": statistics.mean(xs),
        "sample_sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def parse_cell(cell: Path, raw_root: Path) -> dict[str, Any]:
    rel = cell.relative_to(raw_root)
    parts = rel.parts
    if parts[0] in {"selection_ablations", "feedback_ablations"}:
        if len(parts) != 5:
            raise ValueError(f"unexpected ablation path: {rel}")
        family, variant, dataset, model, seed_name = parts
        method = "ra_cofuzz"
        phase = "ablation_selection" if family == "selection_ablations" else "ablation_feedback"
    else:
        if len(parts) != 4:
            raise ValueError(f"unexpected standard path: {rel}")
        method, dataset, model, seed_name = parts
        variant = "full" if method == "ra_cofuzz" else "native"
        if dataset == "gptfuzzer" and model == "llama32_3b":
            phase = "main"
        elif model == "llama32_3b":
            phase = "cross_dataset"
        elif dataset == "gptfuzzer":
            phase = "cross_model"
        else:
            raise ValueError(f"unsupported standard cell: {rel}")

    if not seed_name.startswith("seed") or not seed_name[4:].isdigit():
        raise ValueError(f"invalid seed path: {rel}")
    seed = int(seed_name[4:])

    required = [
        "candidates.jsonl",
        "generation.done.json",
        "offline.jsonl",
        "offline_summary.json",
        "roberta.jsonl",
        "roberta_summary.json",
        "evaluation.done.json",
    ]
    missing = [name for name in required if not (cell / name).is_file()]
    if missing:
        raise ValueError(f"incomplete cell {rel}: missing {missing}")

    generation = read_json(cell / "generation.done.json")
    offline_summary = read_json(cell / "offline_summary.json")
    roberta_summary = read_json(cell / "roberta_summary.json")
    evaluation = read_json(cell / "evaluation.done.json")
    candidates = read_jsonl(cell / "candidates.jsonl")
    offline = read_jsonl(cell / "offline.jsonl")
    roberta = read_jsonl(cell / "roberta.jsonl")

    responses = int(generation.get("recorded_responses", len(candidates)))
    target_calls = int(generation.get("actual_target_calls", responses))
    if not (responses == len(candidates) == len(offline) == len(roberta)):
        raise ValueError(
            f"row mismatch {rel}: marker={responses}, candidates={len(candidates)}, "
            f"offline={len(offline)}, roberta={len(roberta)}"
        )
    if int(offline_summary.get("rows_total", -1)) != responses:
        raise ValueError(f"offline total mismatch: {rel}")
    if int(offline_summary.get("rows_evaluated", -1)) != responses:
        raise ValueError(f"offline evaluation incomplete: {rel}")
    if int(offline_summary.get("unknown_count", -1)) != 0:
        raise ValueError(f"unknown labels present: {rel}")
    if offline_summary.get("errors"):
        raise ValueError(f"offline errors present: {rel}")
    if int(roberta_summary.get("total_candidates", -1)) != responses:
        raise ValueError(f"RoBERTa total mismatch: {rel}")
    if int(evaluation.get("recorded_responses", -1)) != responses:
        raise ValueError(f"evaluation marker mismatch: {rel}")

    label_counts = {0: 0, 1: 0, 2: 0}
    question_labels: dict[str, set[int]] = defaultdict(set)
    for row in offline:
        label = row.get("strongjudge_label")
        if label not in label_counts:
            raise ValueError(f"invalid StrongJudge label in {rel}: {label!r}")
        label_counts[int(label)] += 1
        question_labels[question_key(row)].add(int(label))

    roberta_successes = 0
    roberta_questions: dict[str, bool] = defaultdict(bool)
    for row in roberta:
        success = row.get("is_jailbreak_by_original_evaluator")
        if success is None:
            success = row.get("original_evaluator_label") == 1
        success = bool(success)
        roberta_successes += int(success)
        roberta_questions[question_key(row)] |= success

    if set(question_labels) != set(roberta_questions):
        raise ValueError(f"question identity mismatch between evaluators: {rel}")
    covered_questions = len(question_labels)
    if covered_questions != 20:
        raise ValueError(f"expected 20 covered questions in {rel}, found {covered_questions}")

    question_l2 = sum(2 in labels for labels in question_labels.values())
    question_l12 = sum(bool(labels & {1, 2}) for labels in question_labels.values())
    roberta_question_success = sum(roberta_questions.values())
    empty_responses = sum(
        isinstance(row.get("response"), str) and row.get("response") == ""
        for row in candidates
    )

    online_control = generation.get("online_feedback_control") or {}
    if "strong_judge_calls" in online_control:
        online_judge_calls = int(online_control["strong_judge_calls"])
    else:
        online_judge_calls = sum(
            row.get("strong_judge_label") is not None or row.get("strong_judge") is not None
            for row in candidates
        )

    if not math.isclose(
        rate(label_counts[2], responses),
        float(offline_summary.get("strong_judge_asr", -1)),
        abs_tol=1e-12,
    ):
        raise ValueError(f"StrongJudge response-rate mismatch: {rel}")
    if not math.isclose(
        rate(roberta_successes, responses),
        float(roberta_summary.get("original_evaluator_asr", -1)),
        abs_tol=1e-12,
    ):
        raise ValueError(f"RoBERTa response-rate mismatch: {rel}")

    return {
        "cell_id": str(rel),
        "phase": phase,
        "method": method,
        "variant": variant,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "target_calls": target_calls,
        "recorded_responses": responses,
        "covered_questions": covered_questions,
        "empty_responses": empty_responses,
        "label0_count": label_counts[0],
        "label1_count": label_counts[1],
        "label2_count": label_counts[2],
        "response_label2_rate": rate(label_counts[2], responses),
        "response_label1_or_2_rate": rate(label_counts[1] + label_counts[2], responses),
        "question_label2_count": question_l2,
        "question_label2_rate": rate(question_l2, covered_questions),
        "question_label1_or_2_rate": rate(question_l12, covered_questions),
        "roberta_response_rate": rate(roberta_successes, responses),
        "roberta_question_rate": rate(roberta_question_success, covered_questions),
        "online_judge_calls": online_judge_calls,
        "runtime_seconds": float(generation.get("runtime_seconds", 0.0)),
    }


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = [
        "target_calls",
        "recorded_responses",
        "empty_responses",
        "response_label2_rate",
        "response_label1_or_2_rate",
        "question_label2_rate",
        "question_label1_or_2_rate",
        "roberta_response_rate",
        "roberta_question_rate",
        "online_judge_calls",
        "runtime_seconds",
    ]
    output: list[dict[str, Any]] = []
    for group_key, items in groups.items():
        record = dict(zip(keys, group_key))
        record["seeds"] = sorted(item["seed"] for item in items)
        record["runs"] = len(items)
        for metric in metrics:
            record[metric] = mean_sd(item[metric] for item in items)
        output.append(record)
    return output


def add_full_reference(
    rows: list[dict[str, Any]], family: str, variants: list[str]
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["phase"] == family]
    full = [
        {**row, "variant": "full", "phase": family}
        for row in rows
        if row["phase"] == "main" and row["method"] == "ra_cofuzz"
    ]
    combined = full + selected
    found = {row["variant"] for row in combined}
    if found != set(variants):
        raise ValueError(f"unexpected {family} variants: {sorted(found)}")
    return combined


def percent(x: float) -> str:
    return f"{100*x:.1f}"


def pm(metric: dict[str, float], percentage: bool = False, digits: int = 1) -> str:
    scale = 100 if percentage else 1
    return f"{metric['mean']*scale:.{digits}f} ± {metric['sample_sd']*scale:.{digits}f}"


def write_tables(summary: dict[str, Any], table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    dataset_rank = {name: i for i, name in enumerate(DATASET_ORDER)}
    model_rank = {name: i for i, name in enumerate(MODEL_ORDER)}

    main = sorted(summary["main_comparison"], key=lambda x: method_rank[x["method"]])
    lines = [
        "# Main comparison",
        "",
        "| Method | Question L2 ASR (%) | Response L2 ASR (%) | Label-(1+2) rate (%) | RoBERTa rate (%) | Target calls | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main:
        lines.append(
            f"| {row['method']} | {pm(row['question_label2_rate'], True)} | "
            f"{pm(row['response_label2_rate'], True)} | {pm(row['response_label1_or_2_rate'], True)} | "
            f"{pm(row['roberta_response_rate'], True)} | {pm(row['target_calls'])} | "
            f"{pm(row['runtime_seconds'])} |"
        )
    (table_dir / "main_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, rows, column_key, order in [
        ("cross_dataset", summary["cross_dataset"], "dataset", DATASET_ORDER),
        ("cross_model", summary["cross_model"], "model", MODEL_ORDER),
    ]:
        lookup = {(r["method"], r[column_key]): r for r in rows}
        lines = [
            f"# {name.replace('_', ' ').title()} — question-level Label-2 ASR (%)",
            "",
            "| Method | " + " | ".join(order) + " |",
            "|---|" + "---:|" * len(order),
        ]
        for method in METHOD_ORDER:
            cells = [pm(lookup[(method, col)]["question_label2_rate"], True) for col in order]
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
        (table_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, rows, order in [
        ("selection_ablations", summary["selection_ablations"], SELECTION_VARIANTS),
        ("feedback_ablations", summary["feedback_ablations"], FEEDBACK_VARIANTS),
    ]:
        lookup = {r["variant"]: r for r in rows}
        lines = [
            f"# {name.replace('_', ' ').title()}",
            "",
            "| Variant | Question L2 ASR (%) | Response L2 ASR (%) | Online judge calls |",
            "|---|---:|---:|---:|",
        ]
        for variant in order:
            row = lookup[variant]
            lines.append(
                f"| {variant} | {pm(row['question_label2_rate'], True)} | "
                f"{pm(row['response_label2_rate'], True)} | {pm(row['online_judge_calls'])} |"
            )
        (table_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    marker_paths = sorted(raw_root.rglob("evaluation.done.json"))
    cells = [parse_cell(path.parent, raw_root) for path in marker_paths]
    cells.sort(key=lambda r: r["cell_id"])

    phase_counts = defaultdict(int)
    for row in cells:
        phase_counts[row["phase"]] += 1
    expected = {
        "main": 18,
        "cross_dataset": 36,
        "cross_model": 72,
        "ablation_selection": 9,
        "ablation_feedback": 9,
    }
    if len(cells) != 144 or dict(phase_counts) != expected:
        raise ValueError(
            f"expected 144 cells with phases {expected}; got {len(cells)} and {dict(phase_counts)}"
        )

    standard = [r for r in cells if r["phase"] in {"main", "cross_dataset", "cross_model"}]
    main_rows = [r for r in cells if r["phase"] == "main"]
    dataset_rows = [r for r in standard if r["model"] == "llama32_3b"]
    model_rows = [r for r in standard if r["dataset"] == "gptfuzzer"]
    selection_rows = add_full_reference(cells, "ablation_selection", SELECTION_VARIANTS)
    feedback_rows = add_full_reference(cells, "ablation_feedback", FEEDBACK_VARIANTS)

    summary = {
        "schema_version": 1,
        "release": "RA-CoFuzz public results v1.0.0",
        "privacy": "aggregate metrics only; no prompts or model responses",
        "protocol": {
            "questions_per_dataset": 20,
            "target_call_cap": 200,
            "seeds": SEED_ORDER,
            "seed_field_semantics": "stable release run labels; not uniformly independent seeds",
            "randomness_record": "docs/EXPERIMENT_PROTOCOL.md#randomness-record",
            "methods": METHOD_ORDER,
            "datasets": DATASET_ORDER,
            "models": MODEL_ORDER,
            "model_display_names": {
                "llama32_3b": "Llama-3.2-3B-Instruct",
                "qwen25_1_5b": "Qwen2.5-1.5B-Instruct",
                "qwen25_3b": "Qwen2.5-3B-Instruct",
                "qwen25_7b": "Qwen2.5-7B-Instruct",
                "vicuna_7b": "Vicuna-7B-v1.5",
            },
            "evaluation": ["full_offline_strongjudge", "roberta"],
            "strongjudge_model": "deepseek-chat",
            "metric_definitions": {
                "question_label2_rate": "questions with at least one Label-2 response / covered questions",
                "response_label2_rate": "Label-2 responses / recorded responses",
            },
        },
        "cell_count": len(cells),
        "phase_counts": dict(phase_counts),
        "cells": cells,
        "main_comparison": aggregate(main_rows, ["method"]),
        "cross_dataset": aggregate(dataset_rows, ["method", "dataset"]),
        "cross_model": aggregate(model_rows, ["method", "model"]),
        "selection_ablations": aggregate(selection_rows, ["variant"]),
        "feedback_ablations": aggregate(feedback_rows, ["variant"]),
        "evaluator_alignment": aggregate(standard, ["method", "dataset", "model"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    write_tables(summary, args.table_dir)
    print(json.dumps({
        "status": "PUBLIC_AGGREGATION_PASS",
        "cells": len(cells),
        "phase_counts": dict(phase_counts),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
