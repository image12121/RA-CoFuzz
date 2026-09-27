import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run import (apply_selection_ablation, clear_method_env, output_cell,
                 selection_ablation_config, status_fields)
from audit_selection_ablations import audit_cell
from selection_ablation import strict_branch_ablation
from support import OUTPUT, PROTOCOL, digest


def args(ablation=None, method="ra_cofuzz", dataset="gptfuzzer",
         model="llama32_3b", seed=100):
    return argparse.Namespace(ablation=ablation, method=method, dataset=dataset,
                              model=model, seed=seed)


class SelectionAblations(unittest.TestCase):
    @staticmethod
    def fake_selection_module():
        class Hybrid:
            def select(self):
                return "original"

        class MCTS:
            def select(self):
                return "mcts-result"

        return SimpleNamespace(
            HybridRAESSelectPolicy=Hybrid,
            MCTSExploreSelectPolicy=MCTS,
            random=SimpleNamespace(random=lambda: 0.5),
        )

    def test_exact_protocol_matrix(self):
        self.assertEqual(list(PROTOCOL["selection_ablations"]), [
            "hybrid_wo_elite", "hybrid_wo_dp", "hybrid_wo_mcts"
        ])
        self.assertEqual(PROTOCOL["seeds"], [100, 200, 300])

    def test_each_variant_removes_exactly_one_branch(self):
        removed = set()
        for name in PROTOCOL["selection_ablations"]:
            config = selection_ablation_config(args(name))
            ratios = [config[k] for k in
                      ("mcts_ratio", "dp_ratio", "elite_ratio")]
            self.assertAlmostEqual(sum(ratios), 1.0)
            self.assertEqual(ratios.count(0.0), 1)
            self.assertEqual(ratios[("mcts", "dp", "elite").index(
                config["disabled_branch"])], 0.0)
            removed.add(ratios.index(0.0))
        self.assertEqual(removed, {0, 1, 2})

    def test_mcts_removal_disables_mcts_warmup(self):
        config = selection_ablation_config(args("hybrid_wo_mcts"))
        self.assertEqual(config["warmup_steps"], 0)
        self.assertEqual(config["mcts_ratio"], 0.0)

    def test_scope_is_frozen(self):
        for changed in [
            args("hybrid_wo_dp", method="strict_gptfuzzer"),
            args("hybrid_wo_dp", dataset="advbench"),
            args("hybrid_wo_dp", model="qwen25_3b"),
        ]:
            with self.assertRaisesRegex(ValueError, "invalid_selection_ablation_scope"):
                selection_ablation_config(changed)

    def test_environment_is_exact(self):
        keys = [
            "GPTFUZZ_HYBRID_WARMUP_STEPS", "GPTFUZZ_HYBRID_MCTS_RATIO",
            "GPTFUZZ_HYBRID_DP_RATIO", "GPTFUZZ_HYBRID_ELITE_RATIO",
        ]
        with patch.dict(os.environ, {}, clear=False):
            clear_method_env()
            apply_selection_ablation(selection_ablation_config(
                args("hybrid_wo_elite")))
            self.assertEqual([os.environ[k] for k in keys],
                             ["2", "0.50", "0.50", "0.00"])

    def test_disabled_dp_is_not_used_by_fallback(self):
        selection = self.fake_selection_module()
        selector = selection.HybridRAESSelectPolicy()
        selector.hybrid_step = 0
        selector.warmup_steps = 0
        selector.mcts_ratio = 0.8
        selector.dp_ratio = 0.0
        selector.elite_ratio = 0.2
        selector.hybrid_branch_counts = {"mcts": 0, "dp": 0, "elite": 0,
                                         "fallback": 0}
        selector._select_elite = lambda: None
        selector._select_dp = lambda: self.fail("disabled DP was attempted")
        with patch.object(selection.random, "random", return_value=0.9), \
             strict_branch_ablation("dp", selection) as observed:
            self.assertEqual(selector.select(), "mcts-result")
        self.assertEqual(observed["disabled_branch_attempts"], 0)
        self.assertEqual(observed["disabled_branch_selections"], 0)
        self.assertEqual(observed["selection_counts"]["mcts"], 1)

    def test_disabled_elite_is_not_attempted(self):
        selection = self.fake_selection_module()
        selector = selection.HybridRAESSelectPolicy()
        selector.hybrid_step = 0
        selector.warmup_steps = 0
        selector.mcts_ratio = 0.5
        selector.dp_ratio = 0.5
        selector.elite_ratio = 0.0
        selector.hybrid_branch_counts = {"mcts": 0, "dp": 0, "elite": 0,
                                         "fallback": 0}
        selector._select_dp = lambda: "dp-seed"
        selector._select_elite = lambda: self.fail("disabled Elite was attempted")
        selector._manual_select = lambda seed, branch: (seed, branch)
        with patch.object(selection.random, "random", return_value=0.9), \
             strict_branch_ablation("elite", selection) as observed:
            self.assertEqual(selector.select(), ("dp-seed", "dp"))
        self.assertEqual(observed["disabled_branch_attempts"], 0)
        self.assertEqual(observed["selection_counts"]["dp"], 1)

    def test_ablation_output_is_separate(self):
        cell = output_cell(args("hybrid_wo_dp", seed=200))
        self.assertEqual(cell, OUTPUT / "selection_ablations" /
                         "hybrid_wo_dp" / "gptfuzzer" /
                         "llama32_3b" / "seed200")
        self.assertNotIn("ra_cofuzz/gptfuzzer", str(cell))

    def test_existing_output_path_is_unchanged(self):
        cell = output_cell(args(seed=300))
        self.assertEqual(cell, OUTPUT / "ra_cofuzz" / "gptfuzzer" /
                         "llama32_3b" / "seed300")

    def test_status_adds_ablation_without_text_fields(self):
        self.assertEqual(status_fields(args("hybrid_wo_mcts", seed=300)), {
            "method": "ra_cofuzz", "seed": 300,
            "selection_ablation": "hybrid_wo_mcts",
        })

    def test_plan_has_nine_unique_cells(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("run.py")),
             "plan", "--suite", "selection_ablations"],
            capture_output=True, text=True, check=True,
        )
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        cells = {(row["selection_ablation"], row["seed"]) for row in rows}
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(cells), 9)
        self.assertEqual({row["method"] for row in rows}, {"ra_cofuzz"})
        self.assertEqual({row["target_budget"] for row in rows}, {200})

    def make_complete_cell(self, root):
        cell = Path(root)
        candidates = []
        offline = []
        roberta = []
        for index in range(200):
            question_index = index % 20
            candidates.append({"question_index": question_index,
                               "question": f"neutral-{question_index}",
                               "response": "neutral-response"})
            offline.append({"question": f"neutral-{question_index}",
                            "strongjudge_label": index % 3})
            roberta.append({"label": index % 2})
        for name, rows in (("candidates.jsonl", candidates),
                           ("offline.jsonl", offline),
                           ("roberta.jsonl", roberta)):
            (cell / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        config = PROTOCOL["selection_ablations"]["hybrid_wo_dp"]
        generation = {
            "recorded_responses": 200,
            "runtime_seconds": 1.0,
            "candidates_sha256": digest(cell / "candidates.jsonl"),
            "signature": {"selection_ablation": "hybrid_wo_dp",
                          "selection_ablation_config": config,
                          "method": "ra_cofuzz", "dataset": "gptfuzzer",
                          "model": "llama32_3b", "seed": 100},
            "selection_ablation_control": {
                "disabled_branch": "dp",
                "selection_attempts": {"mcts": 1, "dp": 0, "elite": 0},
                "selection_counts": {"mcts": 1, "dp": 0, "elite": 0},
                "fallback_events": 0,
                "native_branch_counts": {"mcts": 1, "dp": 0,
                                         "elite": 0, "fallback": 0},
                "hybrid_steps": 1,
                "disabled_branch_attempts": 0,
                "disabled_branch_selections": 0,
            },
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
        (cell / "evaluation.done.json").write_text(json.dumps({"artifacts": artifacts}))
        return cell

    def test_completed_cell_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_cell(self.make_complete_cell(directory),
                                "hybrid_wo_dp", 100)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["covered_questions"], 20)

    def test_completed_cell_audit_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = self.make_complete_cell(directory)
            with (cell / "candidates.jsonl").open("a") as stream:
                stream.write(json.dumps({"question": "neutral", "response": "neutral"}) + "\n")
            result = audit_cell(cell, "hybrid_wo_dp", 100)
            self.assertEqual(result["status"], "FAILED")
            self.assertFalse(result["checks"]["candidate_hash"])


if __name__ == "__main__":
    unittest.main()
