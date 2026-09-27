import argparse
import json
from collections import Counter, defaultdict

from gptfuzzer.utils.predict import RoBERTaPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    responses = [r.get("response", "") or "" for r in rows]

    predictor = RoBERTaPredictor("hubert233/GPTFuzz", device=args.device)
    preds = predictor.predict(responses)

    total = len(rows)
    roberta_success = 0
    strong_counts = Counter()
    roberta_counts = Counter()
    cross = defaultdict(Counter)

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for row, pred in zip(rows, preds):
            try:
                pred_int = int(pred)
            except Exception:
                pred_int = 0

            row["original_evaluator_label"] = pred_int
            row["is_jailbreak_by_original_evaluator"] = bool(pred_int)

            strong_label = row.get("strong_judge_label")
            strong_counts[str(strong_label)] += 1
            roberta_counts[str(pred_int)] += 1
            cross[str(strong_label)][str(pred_int)] += 1

            if pred_int == 1:
                roberta_success += 1

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "total_candidates": total,
        "original_evaluator_success_count": roberta_success,
        "original_evaluator_asr": roberta_success / total if total else 0.0,
        "roberta_label_counts": dict(roberta_counts),
        "strong_judge_label_counts": dict(strong_counts),
        "cross_table_strong_vs_original": {
            k: dict(v) for k, v in cross.items()
        },
    }

    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
