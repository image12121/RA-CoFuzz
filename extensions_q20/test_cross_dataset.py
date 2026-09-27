import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audit_cross_dataset import audit_cell, expected_methods
from support import PROTOCOL, digest


class CrossDataset(unittest.TestCase):
    def test_frozen_matrix(self):
        cross = PROTOCOL["cross_dataset"]
        self.assertEqual(cross["datasets"], ["advbench", "jailbreakbench"])
        self.assertEqual(cross["model"], "llama32_3b")
        self.assertEqual(cross["core_methods"], [
            "ra_cofuzz", "strict_gptfuzzer", "pair"])
        self.assertEqual(cross["remaining_methods"], [
            "tap", "renellm", "deepinception"])
        self.assertEqual(cross["seeds"], [100, 200, 300])
        self.assertEqual(set(expected_methods("all")),
                         set(PROTOCOL["existing_methods"] +
                             PROTOCOL["new_methods"]))

    def test_plans_are_exact_and_disjoint(self):
        sets = {}
        for suite, expected in (("cross_dataset_core", 18),
                                ("cross_dataset_remaining", 18),
                                ("cross_dataset_all", 36)):
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("run.py")),
                 "plan", "--suite", suite],
                capture_output=True, text=True, check=True,
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            cells = {(row["dataset"], row["method"], row["seed"])
                     for row in rows}
            self.assertEqual(len(rows), expected)
            self.assertEqual(len(cells), expected)
            self.assertEqual({row["model"] for row in rows}, {"llama32_3b"})
            self.assertEqual({row["target_budget_cap"] for row in rows}, {200})
            self.assertEqual({row["per_question_cap"] for row in rows}, {10})
            self.assertNotIn("gptfuzzer", {row["dataset"] for row in rows})
            sets[suite] = cells
        self.assertFalse(sets["cross_dataset_core"] &
                         sets["cross_dataset_remaining"])
        self.assertEqual(sets["cross_dataset_all"],
                         sets["cross_dataset_core"] |
                         sets["cross_dataset_remaining"])

    @staticmethod
    def make_cell(root, method, dataset="advbench", seed=100):
        cell = Path(root)
        per_question = 10 if method in {
            "ra_cofuzz", "strict_gptfuzzer"} else 1 if method == \
            "deepinception" else 5
        candidates = []
        offline = []
        roberta = []
        for q_index in range(20):
            for attempt in range(per_question):
                question = f"neutral-{q_index}"
                candidates.append({"question": question,
                                   "response": "neutral-response"})
                offline.append({"question": question,
                                "strongjudge_label": (q_index + attempt) % 3})
                roberta.append({"label": attempt % 2})
        for name, rows in (("candidates.jsonl", candidates),
                           ("offline.jsonl", offline),
                           ("roberta.jsonl", roberta)):
            (cell / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
        generation = {
            "recorded_responses": len(candidates),
            "runtime_seconds": 1.0,
            "candidates_sha256": digest(cell / "candidates.jsonl"),
            "signature": {"method": method, "dataset": dataset,
                          "model": "llama32_3b", "seed": seed},
            "variant": (PROTOCOL["easy_variant"][method]
                        if method in PROTOCOL["new_methods"]
                        else "original_entry_with_explicit_seed"),
        }
        (cell / "generation.done.json").write_text(json.dumps(generation))
        (cell / "offline_summary.json").write_text(json.dumps({
            "rows_total": len(candidates), "rows_evaluated": len(candidates),
            "unknown_count": 0, "errors": [],
        }))
        (cell / "roberta_summary.json").write_text(json.dumps({
            "total_candidates": len(candidates), "original_evaluator_asr": 0.5,
        }))
        artifacts = {name: digest(cell / name) for name in
                     ("offline.jsonl", "offline_summary.json",
                      "roberta.jsonl", "roberta_summary.json")}
        (cell / "evaluation.done.json").write_text(
            json.dumps({"artifacts": artifacts}))
        return cell

    def test_synthetic_cells_for_all_six_methods(self):
        for method in expected_methods("all"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                result = audit_cell(self.make_cell(root, method), method,
                                    "advbench", 100)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["covered_questions"], 20)

    def test_wrong_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = self.make_cell(root, "ra_cofuzz")
            lines = (cell / "candidates.jsonl").read_text().splitlines()
            (cell / "candidates.jsonl").write_text("\n".join(lines[:-1]) + "\n")
            result = audit_cell(cell, "ra_cofuzz", "advbench", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["response_count"])

    def test_tampered_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = self.make_cell(root, "pair")
            with (cell / "offline.jsonl").open("a") as stream:
                stream.write(json.dumps({"question": "neutral",
                                         "strongjudge_label": 0}) + "\n")
            result = audit_cell(cell, "pair", "advbench", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["offline_rows_match"])


if __name__ == "__main__":
    unittest.main()
