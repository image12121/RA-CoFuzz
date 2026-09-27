import json
from pathlib import Path
import tempfile
import unittest

from audit_cross_model import audit_cell
from evaluation_compat import (ORIGINAL_MISSING_ERROR, POLICY_NAME,
                               apply_empty_response_policy)
from support import digest
import test_cross_model


class EmptyResponseEvaluation(unittest.TestCase):
    @staticmethod
    def prepare_empty_records(cell, count=15):
        candidates = [json.loads(line) for line in
                      (cell / "candidates.jsonl").read_text().splitlines()]
        offline = [json.loads(line) for line in
                   (cell / "offline.jsonl").read_text().splitlines()]
        for index in range(count):
            candidates[index]["response"] = ""
            offline[index]["response"] = ""
            offline[index].pop("strongjudge_label", None)
            offline[index]["error"] = ORIGINAL_MISSING_ERROR
        (cell / "candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in candidates))
        (cell / "offline.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in offline))
        generation = json.loads((cell / "generation.done.json").read_text())
        generation["candidates_sha256"] = digest(cell / "candidates.jsonl")
        (cell / "generation.done.json").write_text(json.dumps(generation))
        labels = [row.get("strongjudge_label") for row in offline[count:]]
        summary = json.loads((cell / "offline_summary.json").read_text())
        summary.update({
            "rows_evaluated": len(offline) - count,
            "unknown_count": count,
            "strong_judge_label_0_count": sum(value == 0 for value in labels),
            "strong_judge_label_1_count": sum(value == 1 for value in labels),
            "strong_judge_label_2_count": sum(value == 2 for value in labels),
            "errors": [{"row_index": index,
                        "error": ORIGINAL_MISSING_ERROR}
                       for index in range(count)],
        })
        (cell / "offline_summary.json").write_text(json.dumps(summary))
        return candidates, offline

    def test_repair_15_empty_records_and_cross_model_audit(self):
        with tempfile.TemporaryDirectory() as root:
            cell = test_cross_model.CrossModel.make_cell(
                root, "ra_cofuzz", "vicuna_7b", 100)
            self.prepare_empty_records(cell)
            result = apply_empty_response_policy(
                cell / "candidates.jsonl", cell / "offline.jsonl",
                cell / "offline_summary.json")
            self.assertEqual(result["applied_count"], 15)
            summary = json.loads((cell / "offline_summary.json").read_text())
            self.assertEqual(summary["rows_evaluated"], 200)
            self.assertEqual(summary["unknown_count"], 0)
            self.assertEqual(summary["strong_judge_label_0_count"] +
                             summary["strong_judge_label_1_count"] +
                             summary["strong_judge_label_2_count"], 200)
            evaluation = {
                "empty_response_policy": result,
                "artifacts": {name: digest(cell / name) for name in
                              ("offline.jsonl", "offline_summary.json",
                               "roberta.jsonl", "roberta_summary.json")},
            }
            (cell / "evaluation.done.json").write_text(json.dumps(evaluation))
            audited = audit_cell(cell, "ra_cofuzz", "vicuna_7b", 100)
            self.assertEqual(audited["status"], "PASS")
            self.assertEqual(audited["empty_responses"], 15)

    def test_no_empty_response_is_noop(self):
        with tempfile.TemporaryDirectory() as root:
            cell = test_cross_model.CrossModel.make_cell(root, "ra_cofuzz")
            before = (cell / "offline.jsonl").read_bytes()
            result = apply_empty_response_policy(
                cell / "candidates.jsonl", cell / "offline.jsonl",
                cell / "offline_summary.json")
            self.assertEqual(result["applied_count"], 0)
            self.assertEqual((cell / "offline.jsonl").read_bytes(), before)

    def test_alignment_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = test_cross_model.CrossModel.make_cell(root, "ra_cofuzz")
            self.prepare_empty_records(cell, count=1)
            rows = [json.loads(line) for line in
                    (cell / "offline.jsonl").read_text().splitlines()]
            rows[0]["question"] = "different-neutral-question"
            (cell / "offline.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError,
                                        "empty_policy_source_alignment_mismatch"):
                apply_empty_response_policy(
                    cell / "candidates.jsonl", cell / "offline.jsonl",
                    cell / "offline_summary.json")

    def test_unexpected_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = test_cross_model.CrossModel.make_cell(root, "ra_cofuzz")
            self.prepare_empty_records(cell, count=1)
            rows = [json.loads(line) for line in
                    (cell / "offline.jsonl").read_text().splitlines()]
            rows[0]["error"] = "different-error"
            (cell / "offline.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError,
                                        "empty_policy_unexpected_empty_record"):
                apply_empty_response_policy(
                    cell / "candidates.jsonl", cell / "offline.jsonl",
                    cell / "offline_summary.json")

    def test_summary_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = test_cross_model.CrossModel.make_cell(root, "ra_cofuzz")
            self.prepare_empty_records(cell, count=1)
            summary = json.loads((cell / "offline_summary.json").read_text())
            summary["unknown_count"] = 2
            (cell / "offline_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(
                    ValueError, "empty_policy_summary_not_exactly_repairable"):
                apply_empty_response_policy(
                    cell / "candidates.jsonl", cell / "offline.jsonl",
                    cell / "offline_summary.json")


if __name__ == "__main__":
    unittest.main()
