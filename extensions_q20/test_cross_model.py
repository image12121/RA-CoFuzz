import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from audit_cross_model import audit_cell, expected_methods
from support import PROTOCOL, digest


class CrossModel(unittest.TestCase):
    def test_frozen_matrix(self):
        cross = PROTOCOL["cross_model"]
        self.assertEqual(cross["dataset"], "gptfuzzer")
        self.assertEqual(cross["reference_model"], "llama32_3b")
        self.assertEqual(cross["new_models"], [
            "qwen25_1_5b", "qwen25_3b", "qwen25_7b", "vicuna_7b"])
        self.assertNotIn(cross["reference_model"], cross["new_models"])
        self.assertEqual(cross["core_methods"], [
            "ra_cofuzz", "strict_gptfuzzer", "pair"])
        self.assertEqual(cross["remaining_methods"], [
            "tap", "renellm", "deepinception"])
        self.assertEqual(cross["seeds"], [100, 200, 300])
        self.assertEqual(set(expected_methods("all")),
                         set(PROTOCOL["existing_methods"] +
                             PROTOCOL["new_methods"]))

    def test_plans_are_exact_disjoint_and_exclude_reference_model(self):
        sets = {}
        for suite, expected in (("cross_model_core", 36),
                                ("cross_model_remaining", 36),
                                ("cross_model_all", 72)):
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("run.py")),
                 "plan", "--suite", suite],
                capture_output=True, text=True, check=True,
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            cells = {(row["model"], row["method"], row["seed"])
                     for row in rows}
            self.assertEqual(len(rows), expected)
            self.assertEqual(len(cells), expected)
            self.assertEqual({row["dataset"] for row in rows}, {"gptfuzzer"})
            self.assertEqual({row["target_budget_cap"] for row in rows}, {200})
            self.assertEqual({row["per_question_cap"] for row in rows}, {10})
            self.assertNotIn("llama32_3b", {row["model"] for row in rows})
            sets[suite] = cells
        self.assertFalse(sets["cross_model_core"] &
                         sets["cross_model_remaining"])
        self.assertEqual(sets["cross_model_all"],
                         sets["cross_model_core"] |
                         sets["cross_model_remaining"])

    @staticmethod
    def make_cell(root, method, model="qwen25_1_5b", seed=100):
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
                                   "candidate_prompt": "neutral-prompt",
                                   "response": "neutral-response"})
                offline.append({"row_index": len(offline),
                                "question": question,
                                "candidate_prompt": "neutral-prompt",
                                "response": "neutral-response",
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
            "signature": {"method": method, "dataset": "gptfuzzer",
                          "model": model, "seed": seed},
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

    def test_synthetic_cells_for_all_methods_and_models(self):
        for model in PROTOCOL["cross_model"]["new_models"]:
            for method in expected_methods("all"):
                with self.subTest(model=model, method=method), \
                        tempfile.TemporaryDirectory() as root:
                    result = audit_cell(
                        self.make_cell(root, method, model), method, model, 100)
                    self.assertEqual(result["status"], "PASS")
                    self.assertEqual(result["covered_questions"], 20)

    def test_wrong_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = self.make_cell(root, "ra_cofuzz")
            lines = (cell / "candidates.jsonl").read_text().splitlines()
            (cell / "candidates.jsonl").write_text("\n".join(lines[:-1]) + "\n")
            result = audit_cell(cell, "ra_cofuzz", "qwen25_1_5b", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["response_count"])

    def test_wrong_model_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = self.make_cell(root, "pair")
            marker = json.loads((cell / "generation.done.json").read_text())
            marker["signature"]["model"] = "llama32_3b"
            (cell / "generation.done.json").write_text(json.dumps(marker))
            result = audit_cell(cell, "pair", "qwen25_1_5b", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["model"])

    def test_tampered_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cell = self.make_cell(root, "pair")
            with (cell / "offline.jsonl").open("a") as stream:
                stream.write(json.dumps({"question": "neutral",
                                         "strongjudge_label": 0}) + "\n")
            result = audit_cell(cell, "pair", "qwen25_1_5b", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["offline_rows_match"])


if __name__ == "__main__":
    unittest.main()
