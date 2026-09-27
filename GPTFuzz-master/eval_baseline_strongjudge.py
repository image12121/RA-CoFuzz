import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

from gptfuzzer.utils.ra_judge import StrongJudge


def load_heuristic_assessment():
    candidates = (
        "gptfuzzer.utils.ra_judge",
        "gptfuzzer.utils.predict",
        "gptfuzzer.utils.judge",
        "gptfuzzer.utils.template",
    )
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["heuristic_assessment"])
            fn = getattr(module, "heuristic_assessment", None)
            if fn is not None:
                return fn
        except ImportError:
            continue
    raise ImportError(
        "Could not import heuristic_assessment from known gptfuzzer utility modules."
    )


def latest_results_csv(project_dir):
    pattern = os.path.join(project_dir, "results-*.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No results-*.csv files found under {project_dir}")
    return max(paths, key=os.path.getmtime)


def pick_value(row, names):
    normalized = {key.strip().lower(): value for key, value in row.items() if key}
    for name in names:
        value = normalized.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def load_first_question(question_file):
    df_q = pd.read_csv(question_file)
    if df_q.empty:
        raise ValueError(f"Question file has no data rows: {question_file}")
    if "text" in df_q.columns:
        return str(df_q["text"].iloc[0])
    return str(df_q.iloc[0, -1])


def json_default(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def classify_judge_result(result):
    if not isinstance(result, dict):
        return None
    label = result.get("strong_judge_label")
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def iter_all_candidates_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield {
                "row_index": idx,
                "question": item.get("question"),
                "prompt": item.get("candidate_prompt"),
                "response": item.get("response"),
                "mutator": item.get("mutator"),
                "predictor_result": item.get("predictor_result"),
                "is_jailbreak_by_roberta": item.get("is_jailbreak_by_roberta"),
            }


def iter_results_csv(path, question):
    prompt_names = (
        "prompt",
        "candidate_prompt",
        "jailbreak_prompt",
        "mutated_prompt",
        "attack_prompt",
        "query",
    )
    response_names = (
        "response",
        "target_response",
        "output",
        "answer",
        "generation",
        "generated_text",
    )
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        for idx, row in enumerate(reader):
            yield {
                "row_index": idx,
                "question": question,
                "prompt": pick_value(row, prompt_names),
                "response": pick_value(row, response_names),
                "mutator": pick_value(row, ("mutator", "mutation", "mutation_method")),
                "predictor_result": pick_value(row, ("predictor_result", "roberta_result", "roberta")),
                "is_jailbreak_by_roberta": pick_value(
                    row, ("is_jailbreak_by_roberta", "is_jailbreak", "jailbreak")
                ),
            }


def evaluate(args):
    project_dir = os.path.abspath(args.project_dir)
    all_candidates_jsonl = (
        os.path.abspath(args.all_candidates_jsonl) if args.all_candidates_jsonl else None
    )
    results_csv = None if all_candidates_jsonl else os.path.abspath(
        args.results_csv or latest_results_csv(project_dir)
    )
    question_file = os.path.abspath(args.question_file)
    default_question = args.question if args.question is not None else load_first_question(question_file)
    output_jsonl = os.path.abspath(args.output_jsonl)
    output_summary = os.path.abspath(args.output_summary)

    heuristic_assessment = load_heuristic_assessment()
    judge = StrongJudge()

    counts = Counter()
    errors = []
    rows_total = 0
    rows_evaluated = 0

    if all_candidates_jsonl:
        candidates = iter_all_candidates_jsonl(all_candidates_jsonl)
        source_path = all_candidates_jsonl
    else:
        candidates = iter_results_csv(results_csv, default_question)
        source_path = results_csv

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for item in candidates:
            rows_total += 1
            question = item.get("question") or default_question
            prompt = item.get("prompt")
            response = item.get("response")

            record = {
                "row_index": item.get("row_index"),
                "source_path": source_path,
                "question": question,
                "prompt": prompt,
                "candidate_prompt": prompt,
                "response": response,
                "mutator": item.get("mutator"),
                "predictor_result": item.get("predictor_result"),
                "is_jailbreak_by_roberta": item.get("is_jailbreak_by_roberta"),
            }

            if not prompt or not response:
                record["error"] = "missing prompt or response"
                errors.append({"row_index": item.get("row_index"), "error": record["error"]})
                counts["unknown_count"] += 1
                f_out.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
                continue

            try:
                heuristic = heuristic_assessment(
                    response,
                    original_prompt=question,
                    candidate_prompt=prompt,
                )
                strongjudge = judge.judge(question, prompt, response, heuristic)
                label = classify_judge_result(strongjudge)

                record.update(
                    {
                        "heuristic": heuristic,
                        "strongjudge": strongjudge,
                        "strongjudge_label": label,
                    }
                )
                if label in (0, 1, 2):
                    counts[f"strong_judge_label_{label}_count"] += 1
                else:
                    counts["unknown_count"] += 1
                rows_evaluated += 1
            except Exception as exc:
                record["error"] = repr(exc)
                errors.append({"row_index": item.get("row_index"), "error": repr(exc)})
                counts["unknown_count"] += 1

            f_out.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")

    label_0_count = counts["strong_judge_label_0_count"]
    label_1_count = counts["strong_judge_label_1_count"]
    label_2_count = counts["strong_judge_label_2_count"]
    unknown_count = counts["unknown_count"]
    strong_judge_asr = label_2_count / rows_evaluated if rows_evaluated else 0
    partial_clear_rate = (
        (label_1_count + label_2_count) / rows_evaluated if rows_evaluated else 0
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_source": "all_candidates_jsonl" if all_candidates_jsonl else "results_csv",
        "all_candidates_jsonl": args.all_candidates_jsonl,
        "results_csv": results_csv,
        "question_source": "command_line" if args.question is not None else question_file,
        "question": default_question,
        "output_jsonl": output_jsonl,
        "rows_total": rows_total,
        "rows_evaluated": rows_evaluated,
        "strong_judge_label_0_count": label_0_count,
        "strong_judge_label_1_count": label_1_count,
        "strong_judge_label_2_count": label_2_count,
        "unknown_count": unknown_count,
        "strong_judge_asr": strong_judge_asr,
        "partial_clear_rate": partial_clear_rate,
        "counts": dict(counts),
        "errors": errors,
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    return summary


def parse_args():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Post-hoc StrongJudge evaluation for baseline candidates."
    )
    parser.add_argument("--project-dir", default=project_dir)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--all-candidates-jsonl", default=None)
    parser.add_argument(
        "--question-file",
        default=os.path.join(project_dir, "datasets", "questions", "question_list.csv"),
    )
    parser.add_argument("--question", default=None)
    parser.add_argument(
        "--output-jsonl",
        default=os.path.join(project_dir, "baseline_strongjudge_eval.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        default=os.path.join(project_dir, "baseline_strongjudge_summary.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = evaluate(parse_args())
    except Exception as exc:
        print(f"eval_baseline_strongjudge.py failed: {exc}", file=sys.stderr)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
