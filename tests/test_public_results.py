import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/aggregate/summary.json"


class PublicResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_cell_count(self):
        self.assertEqual(self.data["cell_count"], 144)
        self.assertEqual(
            self.data["phase_counts"],
            {
                "main": 18,
                "cross_dataset": 36,
                "cross_model": 72,
                "ablation_selection": 9,
                "ablation_feedback": 9,
            },
        )

    def test_complete_seed_sets(self):
        self.assertEqual(
            self.data["protocol"]["seed_field_semantics"],
            "stable release run labels; not uniformly independent seeds",
        )
        for section in [
            "main_comparison",
            "cross_dataset",
            "cross_model",
            "selection_ablations",
            "feedback_ablations",
        ]:
            for row in self.data[section]:
                self.assertEqual(row["seeds"], [100, 200, 300])
                self.assertEqual(row["runs"], 3)

    def test_no_content_fields(self):
        forbidden = {"prompt", "candidate_prompt", "response", "question", "strongjudge"}
        for row in self.data["cells"]:
            self.assertTrue(forbidden.isdisjoint(row))

    def test_rates_are_bounded(self):
        suffixes = ("_rate",)
        for row in self.data["cells"]:
            for key, value in row.items():
                if key.endswith(suffixes):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_label2_denominators(self):
        for row in self.data["cells"]:
            self.assertTrue(
                math.isclose(
                    row["question_label2_rate"],
                    row["question_label2_count"] / row["covered_questions"],
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    row["response_label2_rate"],
                    row["label2_count"] / row["recorded_responses"],
                    abs_tol=1e-12,
                )
            )

    def test_strongjudge_model(self):
        self.assertEqual(self.data["protocol"]["strongjudge_model"], "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
