"""Neutral, API-free regressions for the TAP .generate.calls failure."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from easy_bridge import construct, ensure_generation_counter, select_tap_system_prompt


class GenerationCounter(unittest.TestCase):
    def test_structured_template_selected_independent_of_order(self):
        short = "Return a neutral reformulation for {query}."
        structured = ('For {query}, return a mapping with '
                      '\"improvement\": \"neutral\" and \"prompt\": \"neutral\".')
        first, profile = select_tap_system_prompt([short, structured])
        second, _ = select_tap_system_prompt([structured, short])
        self.assertEqual(first, second)
        self.assertEqual(profile["length"], len(structured))
        self.assertEqual(len(profile["sha256"]), 64)

    def test_missing_structured_template_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "tap_structured_template_contract_mismatch"):
            select_tap_system_prompt(["Return a neutral reformulation for {query}."])

    def test_ambiguous_structured_templates_rejected(self):
        templates = [
            '{query} "improvement": "one", "prompt": "one"',
            '{query} "improvement": "two", "prompt": "two"',
        ]
        with self.assertRaisesRegex(RuntimeError, "tap_structured_template_contract_mismatch"):
            select_tap_system_prompt(templates)

    def test_reported_statistics_expression(self):
        recipe = SimpleNamespace(evaluator=SimpleNamespace(
            eval_model=SimpleNamespace(generate=lambda text: "neutral")))
        with self.assertRaises(AttributeError):
            _ = recipe.evaluator.eval_model.generate.calls
        ensure_generation_counter(recipe.evaluator.eval_model)
        for _ in range(3):
            recipe.evaluator.eval_model.generate("neutral")
        num_responses = 3
        self.assertEqual(recipe.evaluator.eval_model.generate.calls - num_responses, 0)

    def test_arguments_and_return_unchanged(self):
        received, result = [], object()
        def original(*args, **kwargs):
            received.append((args, kwargs))
            return result
        model = SimpleNamespace(generate=original)
        ensure_generation_counter(model)
        self.assertIs(model.generate("neutral", temperature=0), result)
        self.assertEqual(received, [(("neutral",), {"temperature": 0})])
        self.assertEqual(model.generate.calls, 1)

    def test_exception_unchanged(self):
        failure = RuntimeError("neutral")
        def original(*args):
            raise failure
        model = SimpleNamespace(generate=original)
        ensure_generation_counter(model)
        with self.assertRaises(RuntimeError) as caught:
            model.generate("neutral")
        self.assertIs(caught.exception, failure)
        self.assertEqual(model.generate.calls, 1)

    def test_install_idempotent(self):
        model = SimpleNamespace(generate=lambda text: text)
        ensure_generation_counter(model)
        original = model.generate
        model.generate("neutral")
        ensure_generation_counter(model)
        self.assertIs(model.generate, original)
        self.assertEqual(model.generate.calls, 1)

    def test_instances_independent(self):
        class Model:
            def generate(self, text):
                return text
        first, second = Model(), Model()
        ensure_generation_counter(first)
        ensure_generation_counter(second)
        first.generate("neutral")
        self.assertEqual((first.generate.calls, second.generate.calls), (1, 0))

    def test_existing_counter_preserved(self):
        def original(text):
            return text
        original.calls = 7
        model = SimpleNamespace(generate=original)
        ensure_generation_counter(model)
        self.assertIs(model.generate, original)
        self.assertEqual(model.generate.calls, 7)

    def test_constructor_actual_evaluator_counter(self):
        separate_model = SimpleNamespace(generate=lambda text: text)
        def constructor(attack_model, target_model, eval_model, jailbreak_datasets):
            return SimpleNamespace(mutator=SimpleNamespace(system_prompt="{query}", model=attack_model),
                                   evaluator=SimpleNamespace(eval_model=separate_model),
                                   single_attack=lambda instance: instance)
        attack = SimpleNamespace(generate=lambda text: text)
        evaluator = SimpleNamespace(generate=lambda text: text)
        template = '{query} "improvement": "neutral", "prompt": "neutral"'
        with patch("easy_bridge.recipe_class", return_value=constructor), \
                patch("easy_bridge.tap_templates", return_value=[template]), \
                patch("tap_compat.install"):
            recipe = construct("tap", attack, object(), evaluator, [])
        self.assertEqual(recipe.evaluator.eval_model.generate.calls, 0)
        recipe.evaluator.eval_model.generate("neutral")
        self.assertEqual(recipe.evaluator.eval_model.generate.calls, 1)
        self.assertEqual(attack.generate.calls, 0)
        self.assertEqual(evaluator.generate.calls, 0)


if __name__ == "__main__":
    unittest.main()
