import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from audit_feedback_ablations import audit_feedback_cell
from feedback_ablation import (CORE_FITNESS, CORE_SEED, RAW_LABEL,
                               SELECTION_METHODS, disable_label1_reward)
from run import (feedback_ablation_config, output_cell, status_fields)
from support import LEGACY, OUTPUT, PROTOCOL, digest


def args(feedback=None, method="ra_cofuzz", dataset="gptfuzzer",
         model="llama32_3b", seed=100, selection=None):
    return argparse.Namespace(feedback_ablation=feedback, ablation=selection,
                              method=method, dataset=dataset, model=model,
                              seed=seed)


def fake_fitness(weak_info, label):
    return 0 if label is None else label


class FakeFuzzer:
    def evaluate_ra(self, labels):
        results = []
        for strong_label in labels:
            weak_info = {}
            fitness = calculate_fitness_score(weak_info, strong_label)
            seed_candidate = False
            if False:
                seed_candidate = False
            elif strong_label in (1, 2) or (strong_label is None and fitness > 2):
                seed_candidate = True
            results.append((fitness, seed_candidate))
        return results


class FakeMCTS:
    def _boundary_score(self, pn):
        label = getattr(pn, "strong_judge_label", None)
        return label

    def _boundary_nodes(self, pn):
        label = getattr(pn, "strong_judge_label", None)
        return label

    def _ra_bonus(self, pn):
        label = getattr(pn, "strong_judge_label", None)
        return label


class FakeHybrid(FakeMCTS):
    def _semantic_score(self, pn):
        label = getattr(pn, "strong_judge_label", None)
        return label

    def _elite_score(self, pn):
        label = getattr(pn, "strong_judge_label", None)
        return label

    def _elite_nodes(self, pn):
        label = getattr(pn, "strong_judge_label", None)
        return label


class FeedbackAblations(unittest.TestCase):
    def test_exact_protocol_matrix(self):
        self.assertEqual(list(PROTOCOL["feedback_ablations"]), [
            "wo_label1_reward", "wo_online_strongjudge_feedback",
            "full_online_review",
        ])
        self.assertEqual(PROTOCOL["seeds"], [100, 200, 300])

    def test_scope_and_mutual_exclusion(self):
        for changed in [
            args("wo_label1_reward", method="strict_gptfuzzer"),
            args("wo_label1_reward", dataset="advbench"),
            args("wo_label1_reward", model="qwen25_3b"),
        ]:
            with self.assertRaisesRegex(ValueError, "invalid_feedback_ablation_scope"):
                feedback_ablation_config(changed)
        with self.assertRaisesRegex(ValueError, "multiple_ablation_families_selected"):
            feedback_ablation_config(args("wo_label1_reward", selection="hybrid_wo_dp"))

    def test_protocol_treatment_assignments(self):
        no_online = feedback_ablation_config(
            args("wo_online_strongjudge_feedback"))
        full = feedback_ablation_config(args("full_online_review"))
        label1 = feedback_ablation_config(args("wo_label1_reward"))
        self.assertEqual((no_online["strong_judge_in_loop"],
                          no_online["force_review"],
                          no_online["expected_online_calls"]),
                         (False, False, "exactly_0"))
        self.assertEqual((full["strong_judge_in_loop"], full["force_review"],
                          full["expected_online_calls"]),
                         (True, True, "exactly_200"))
        self.assertEqual(label1["label1_reward"], "disabled")
        self.assertEqual(label1["heuristic_partial_reward"], "unchanged")

    def test_output_and_status_are_isolated(self):
        parsed = args("full_online_review", seed=300)
        self.assertEqual(output_cell(parsed), OUTPUT / "feedback_ablations" /
                         "full_online_review" / "gptfuzzer" /
                         "llama32_3b" / "seed300")
        self.assertEqual(status_fields(parsed), {
            "method": "ra_cofuzz", "seed": 300,
            "feedback_ablation": "full_online_review",
        })

    def test_plan_has_nine_unique_cells(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("run.py")),
             "plan", "--suite", "feedback_ablations"],
            capture_output=True, text=True, check=True,
        )
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        cells = {(row["feedback_ablation"], row["seed"]) for row in rows}
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(cells), 9)
        self.assertEqual({row["target_budget"] for row in rows}, {200})

    def test_uploaded_source_has_all_pinned_transform_contexts(self):
        core = (LEGACY / "gptfuzzer/fuzzer/core.py").read_text(encoding="utf-8")
        self.assertEqual(core.count(CORE_FITNESS.strip()), 1)
        self.assertEqual(core.count(CORE_SEED.strip()), 1)
        selection_path = LEGACY / "gptfuzzer/fuzzer/selection.py"
        tree = ast.parse(selection_path.read_text(encoding="utf-8"))
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name in SELECTION_METHODS:
                source = ast.get_source_segment(
                    selection_path.read_text(encoding="utf-8"), node)
                found[node.name] = source.count(RAW_LABEL)
        self.assertEqual(found, {name: 1 for name in SELECTION_METHODS})

    def test_label1_runtime_mapping_and_restoration(self):
        core = SimpleNamespace(GPTFuzzer=FakeFuzzer,
                               calculate_fitness_score=fake_fitness)
        selection = SimpleNamespace(MCTSRAESSelectPolicy=FakeMCTS,
                                    HybridRAESSelectPolicy=FakeHybrid)
        original = FakeFuzzer.evaluate_ra
        with disable_label1_reward(core, selection) as observed:
            self.assertEqual(FakeFuzzer().evaluate_ra([0, 1, 2]), [
                (0, False), (0, False), (2, True),
            ])
            node = SimpleNamespace(strong_judge_label=1)
            self.assertIsNone(FakeMCTS()._boundary_score(node))
            self.assertIsNone(FakeHybrid()._elite_score(node))
            self.assertEqual(len(observed["transforms"]), 7)
            self.assertEqual(observed["core_label1_remaps"], 1)
            self.assertEqual(observed["selection_label1_remaps"], 2)
        self.assertIs(FakeFuzzer.evaluate_ra, original)
        self.assertEqual(FakeMCTS()._boundary_score(
            SimpleNamespace(strong_judge_label=1)), 1)

    @staticmethod
    def make_complete_cell(root, variant, calls):
        cell = Path(root)
        candidates = []
        offline = []
        roberta = []
        for index in range(200):
            q = f"neutral-{index % 20}"
            candidates.append({"question": q, "response": "neutral-response"})
            offline.append({"question": q, "strongjudge_label": index % 3})
            roberta.append({"label": index % 2})
        for name, rows in (("candidates.jsonl", candidates),
                           ("offline.jsonl", offline),
                           ("roberta.jsonl", roberta)):
            (cell / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
        config = PROTOCOL["feedback_ablations"][variant]
        native = {"total_candidates": 200, "strong_judge_calls": calls}
        (cell / "native_summary.json").write_text(json.dumps(native))
        control = {
            "strong_judge_in_loop": config["strong_judge_in_loop"],
            "force_review": config["force_review"],
            "strong_judge_calls": calls,
            "expected_online_calls": config["expected_online_calls"],
            "total_candidates": 200,
            "native_summary_sha256": digest(cell / "native_summary.json"),
        }
        generation = {
            "recorded_responses": 200, "runtime_seconds": 1.0,
            "candidates_sha256": digest(cell / "candidates.jsonl"),
            "signature": {
                "feedback_ablation": variant,
                "feedback_ablation_config": config,
                "method": "ra_cofuzz", "dataset": "gptfuzzer",
                "model": "llama32_3b", "seed": 100,
            },
            "online_feedback_control": control,
        }
        if variant == "wo_label1_reward":
            generation["feedback_ablation_control"] = {
                "policy": "label1_to_unjudged_for_reward_and_selection_v1",
                "raw_labels_preserved": True,
                "core_label1_remaps": 1,
                "selection_label1_remaps": 1,
                "transforms": [{"source_sha256": str(i),
                                "transformed_sha256": f"changed-{i}"}
                               for i in range(7)],
            }
        (cell / "generation.done.json").write_text(json.dumps(generation))
        (cell / "offline_summary.json").write_text(json.dumps({
            "rows_total": 200, "rows_evaluated": 200,
            "unknown_count": 0, "errors": [],
        }))
        (cell / "roberta_summary.json").write_text(json.dumps({
            "total_candidates": 200, "original_evaluator_asr": 0.5,
        }))
        artifacts = {name: digest(cell / name) for name in
                     ("offline.jsonl", "offline_summary.json",
                      "roberta.jsonl", "roberta_summary.json")}
        (cell / "evaluation.done.json").write_text(
            json.dumps({"artifacts": artifacts}))
        return cell

    def test_synthetic_audit_accepts_each_call_policy(self):
        cases = [("wo_label1_reward", 37),
                 ("wo_online_strongjudge_feedback", 0),
                 ("full_online_review", 200)]
        for variant, calls in cases:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                result = audit_feedback_cell(
                    self.make_complete_cell(directory, variant, calls),
                    variant, 100)
                self.assertEqual(result["status"], "PASS")

    def test_synthetic_audit_rejects_wrong_online_call_count(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_feedback_cell(
                self.make_complete_cell(directory,
                                        "wo_online_strongjudge_feedback", 1),
                "wo_online_strongjudge_feedback", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["online_calls_match_policy"])


if __name__ == "__main__":
    unittest.main()
