"""Neutral tests; reference loop is the user's redacted installed source."""
import json
from pathlib import Path
import tempfile
import unittest

from renellm_flow import _Dataset, native_outer_audit, neutral_probe, run_question
from support import Recorder


def server_reference(self):
    # Same loop, result retention and stopping condition as supplied by user.
    self.attack_results = _Dataset([])
    for instance in self.jailbreak_datasets:
        for time in range(self.evo_max):
            new_Instance = self.single_attack(instance)[0]
            self.evaluator(_Dataset([new_Instance]))
            if new_Instance.eval_results[0] == True:
                break
        self.attack_results.add(new_Instance)
    self.update(self.attack_results)
    self.jailbreak_datasets = self.attack_results
    self.log()


class ReNeLLMFlow(unittest.TestCase):
    def execute(self, labels, limit=20):
        recipe, instances, recorder, events = neutral_probe(labels, limit)
        result = run_question(recipe, instances[0], recorder, _Dataset)
        return result, recorder, events

    def test_native_reference_equivalence(self):
        self.assertEqual(native_outer_audit(server_reference)["native_outer_mock_cases"], 6)

    def test_audit_detects_previous_missing_loop(self):
        def old_entry(recipe):
            for instance in recipe.jailbreak_datasets:
                recipe.single_attack(instance)
        with self.assertRaisesRegex(RuntimeError, "renellm_native_outer_flow_mismatch"):
            native_outer_audit(old_entry)

    def test_success_first(self):
        result, recorder, _ = self.execute([True])
        self.assertEqual(result, dict(rounds=1, evaluations=1, stop="native_success"))
        self.assertEqual(recorder.calls, 1)

    def test_fail_fail_success(self):
        result, _, events = self.execute([False, False, True])
        self.assertEqual(result["rounds"], 3)
        self.assertEqual([x[0] for x in events], ["generate", "evaluate"] * 3)

    def test_all_false_evaluate_tenth_no_eleventh(self):
        result, recorder, _ = self.execute([False])
        self.assertEqual((recorder.calls, result["evaluations"]), (10, 10))
        self.assertEqual(result["stop"], "target_cap")

    def test_native_evo_max(self):
        result, recorder, _ = self.execute([False], 3)
        self.assertEqual(recorder.calls, 3)
        self.assertEqual(result["stop"], "native_evo_max")

    def test_original_instance_retained_including_mutations(self):
        recipe, instances, recorder, _ = neutral_probe([False, False, True], 20)
        run_question(recipe, instances[0], recorder, _Dataset)
        self.assertEqual(instances[0].visits, 3)

    def test_global_budget(self):
        recipe, instances, recorder, _ = neutral_probe([False], 20)
        recorder.calls = 199
        result = run_question(recipe, instances[0], recorder, _Dataset)
        self.assertEqual((recorder.calls, result["evaluations"]), (200, 1))

    def test_eval_exception_propagates(self):
        recipe, instances, recorder, _ = neutral_probe([False], 20)
        failure = RuntimeError("neutral")
        def evaluate(dataset):
            raise failure
        recipe.evaluator = evaluate
        with self.assertRaises(RuntimeError) as caught:
            run_question(recipe, instances[0], recorder, _Dataset)
        self.assertIs(caught.exception, failure)
        self.assertEqual(recorder.calls, 1)

    def test_no_feedback_rejected(self):
        recipe, instances, recorder, _ = neutral_probe([False], 20)
        recipe.evaluator = lambda dataset: None
        with self.assertRaisesRegex(ValueError, "missing_renellm_feedback"):
            run_question(recipe, instances[0], recorder, _Dataset)

    def test_invalid_native_limit_rejected(self):
        for value in [0, -1, True, 1.5, "20"]:
            recipe, instances, recorder, _ = neutral_probe([False], value)
            with self.assertRaisesRegex(ValueError, "invalid_renellm_evo_max"):
                run_question(recipe, instances[0], recorder, _Dataset)

    def test_real_recorder_twenty_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neutral.jsonl"
            recorder = Recorder(path)
            try:
                for index in range(20):
                    recipe, instances, fake, _ = neutral_probe([False], 20)
                    original = recipe.single_attack
                    def single(instance):
                        recorder.generate(lambda _: "neutral response", "neutral input")
                        return original(instance)
                    recipe.single_attack = single
                    recorder.begin(index, "neutral question")
                    result = run_question(recipe, instances[0], recorder, _Dataset)
                    self.assertEqual(result["evaluations"], 10)
                self.assertEqual((recorder.calls, recorder.completed), (200, 200))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len({row["question_index"] for row in rows}), 20)
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
