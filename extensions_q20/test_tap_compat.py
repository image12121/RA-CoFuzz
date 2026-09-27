"""Neutral fixtures reproduce the supplied source's empty/singleton failures."""
import ast
from types import SimpleNamespace
import unittest

from tap_compat import (ConstraintGuard, EmptyCandidateGeneration, EmptyCandidateInput,
                        MutationGuard, balanced_extract_json, parser_compatibility,
                        singleton_callable)


class JailbreakDataset(list):
    pass


np = SimpleNamespace(random=SimpleNamespace(shuffle=lambda values: None))


class NativeConstraint:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.score_calls = 0
        self.eval_model = SimpleNamespace(conversation=SimpleNamespace(messages=[]),
                                          set_system_message=lambda text: None,
                                          generate=lambda text: "neutral")
        self.system_prompt = "{query}"
        self.tree_width = 2
        self.get_evaluator_prompt_on_topic = lambda text: text

    def process_output_on_topic_score(self, output):
        self.score_calls += 1
        return next(self.scores)

    def __call__(self, jailbreak_dataset, *args, **kwargs):
        dataset = jailbreak_dataset
        tuples_list = []
        self.eval_model.conversation.messages = []
        self.eval_model.set_system_message(self.system_prompt.format(query=dataset[0].query))
        for instance in dataset:
            raw_output = self.eval_model.generate(self.get_evaluator_prompt_on_topic(instance.jailbreak_prompt))
            score = self.process_output_on_topic_score(raw_output)
            tuples_list.append((score, instance))
        np.random.shuffle(tuples_list)
        tuples_list.sort(key=lambda x: x[0], reverse=True)
        width = min(self.tree_width, len(tuples_list))
        truncated_list = [tuples_list[i][1] for i in range(width) if tuples_list[i][0] > 0]
        if len(truncated_list) == 0:
            truncated_list = [tuples_list[0][1], tuples_list[1][1]]
        return JailbreakDataset(truncated_list)


class NativeMutation:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0
        self.histories = []
        self.system_prompt = "neutral"
        def generate(text):
            return text
        generate.calls = 0
        self.model = SimpleNamespace(generate=generate)

    def __call__(self, dataset, *args, **kwargs):
        self.calls += 1
        self.model.generate.calls += 1
        self.histories.append(list(dataset[0].history))
        dataset[0].history.append("neutral attempt")
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


def batch(n=1):
    return JailbreakDataset([SimpleNamespace(query="neutral", jailbreak_prompt="neutral", history=[])
                             for _ in range(n)])


def native_parser_fixture(text):
    start = text.find("{")
    end = text.find("}") + 1
    block = text[start:end]
    try:
        parsed = ast.literal_eval(block)
    except (SyntaxError, ValueError):
        return None, None
    if not all(key in parsed for key in ["improvement", "prompt"]):
        return None, None
    return parsed, block


class TapCompatibility(unittest.TestCase):
    def test_parser_source_shape_guard_accepts_pinned_pattern(self):
        parser = parser_compatibility(native_parser_fixture)
        parsed, _ = parser('{"improvement":"neutral","prompt":"Hello."}')
        self.assertEqual(parsed["prompt"], "Hello.")

    def test_balanced_parser_standard_json(self):
        parsed, block = balanced_extract_json(
            'prefix {"improvement":"neutral","prompt":"Return a neutral greeting."} suffix')
        self.assertEqual(parsed["prompt"], "Return a neutral greeting.")
        self.assertTrue(block.startswith("{"))

    def test_balanced_parser_python_literal(self):
        parsed, _ = balanced_extract_json(
            "{'improvement':'neutral','prompt':'Return a neutral greeting.'}")
        self.assertEqual(parsed["improvement"], "neutral")

    def test_balanced_parser_ignores_quoted_braces(self):
        parsed, _ = balanced_extract_json(
            '{"improvement":"neutral","prompt":"Return literal {token} safely."}')
        self.assertIn("{token}", parsed["prompt"])

    def test_balanced_parser_supports_json_only_values(self):
        parsed, _ = balanced_extract_json(
            '{"improvement":"neutral","prompt":"Return a neutral greeting.","note":null}')
        self.assertIsNone(parsed["note"])

    def test_balanced_parser_skips_invalid_mapping_before_valid(self):
        parsed, _ = balanced_extract_json(
            '{"note":"neutral"} then {"improvement":"neutral","prompt":"Hello."}')
        self.assertEqual(parsed["prompt"], "Hello.")

    def test_balanced_parser_rejects_missing_or_empty_prompt(self):
        for value in ('{"improvement":"neutral"}',
                      '{"improvement":"neutral","prompt":""}'):
            self.assertEqual(balanced_extract_json(value), (None, None))

    def test_balanced_parser_rejects_non_string_prompt(self):
        self.assertEqual(balanced_extract_json(
            '{"improvement":"neutral","prompt":3}'), (None, None))

    def test_balanced_parser_rejects_unclosed_mapping(self):
        self.assertEqual(balanced_extract_json(
            '{"improvement":"neutral","prompt":"Hello."'), (None, None))

    def test_parser_wrapper_preserves_native_success(self):
        expected = ({"improvement": "native", "prompt": "Hello."}, "native-block")
        parser = parser_compatibility(lambda text: expected, validate_source=False)
        self.assertIs(parser("neutral input"), expected)
        self.assertEqual(parser.native_accepted, 1)
        self.assertEqual(parser.compat_accepted, 0)

    def test_parser_wrapper_fallback_and_rejection_counters(self):
        parser = parser_compatibility(lambda text: (None, None), validate_source=False)
        parsed, _ = parser('{"improvement":"neutral","prompt":"Hello {name}."}')
        self.assertEqual(parsed["prompt"], "Hello {name}.")
        self.assertEqual(parser("neutral invalid"), (None, None))
        self.assertEqual((parser.compat_accepted, parser.rejected), (1, 1))

    def test_original_empty_failure_reproduced(self):
        with self.assertRaises(IndexError):
            NativeConstraint([])(batch(0))

    def test_original_singleton_failure_reproduced(self):
        with self.assertRaises(IndexError):
            NativeConstraint([0])(batch())

    def test_empty_constraint_rejected_before_eval(self):
        native = NativeConstraint([])
        with self.assertRaises(EmptyCandidateInput):
            ConstraintGuard(native, [])(batch(0))
        self.assertEqual(native.score_calls, 0)

    def test_singleton_negative_fallback(self):
        source, native = batch(), NativeConstraint([0])
        result = ConstraintGuard(native, [])(source)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], source[0])
        self.assertEqual(native.score_calls, 1)

    def test_singleton_positive_unchanged(self):
        source, native = batch(), NativeConstraint([1])
        self.assertIs(ConstraintGuard(native, [])(source)[0], source[0])
        self.assertEqual(native.score_calls, 1)

    def test_two_negative_native_fallback_unchanged(self):
        source, native = batch(2), NativeConstraint([0, 0])
        result = ConstraintGuard(native, [])(source)
        self.assertEqual(result, source)
        self.assertEqual(native.score_calls, 2)

    def test_multiple_ranked_candidates_unchanged(self):
        source = batch(3)
        expected = NativeConstraint([0, 2, 1])(source)
        result = ConstraintGuard(NativeConstraint([0, 2, 1]), [])(source)
        self.assertEqual(result, expected)

    def test_native_file_function_untouched(self):
        before = NativeConstraint.__call__.__code__
        ConstraintGuard(NativeConstraint([0]), [])(batch())
        self.assertIs(NativeConstraint.__call__.__code__, before)

    def test_unrecognized_source_refused(self):
        def different(self, dataset):
            return dataset
        with self.assertRaisesRegex(RuntimeError, "tap_constraint_source_shape_mismatch"):
            singleton_callable(different)

    def test_mutation_nonempty_exact_return(self):
        expected = batch()
        native = NativeMutation([expected])
        events = []
        self.assertIs(MutationGuard(native, events)(batch()), expected)
        self.assertEqual(native.calls, 1)
        self.assertEqual(events, [])

    def test_empty_then_valid_restores_input_state(self):
        expected, native = batch(), NativeMutation([batch(0), batch()])
        events = []
        result = MutationGuard(native, events)(expected)
        self.assertEqual(len(result), 1)
        self.assertEqual(native.histories, [[], ["neutral attempt"]])
        self.assertEqual(native.calls, 2)
        self.assertEqual(events[-1]["status"], "recovered")
        self.assertEqual(events[-1]["mutation_calls_after"], 2)

    def test_empty_exhaustion_not_label_or_empty_return(self):
        native = NativeMutation([batch(0), batch(0)])
        events = []
        with self.assertRaises(EmptyCandidateGeneration):
            MutationGuard(native, events)(batch())
        self.assertEqual(native.calls, 2)
        self.assertEqual(events[-1]["status"], "empty_exhausted")

    def test_one_extra_batch_per_question_not_per_call(self):
        native = NativeMutation([batch(0), batch(), batch(0)])
        guard = MutationGuard(native, [])
        guard.begin_question(0)
        guard(batch())
        with self.assertRaises(EmptyCandidateGeneration):
            guard(batch())
        self.assertEqual(native.calls, 3)

    def test_new_question_resets_extra_batch_cap(self):
        native = NativeMutation([batch(0), batch(), batch(0), batch()])
        events = []
        guard = MutationGuard(native, events)
        for index in [0, 1]:
            guard.begin_question(index)
            guard(batch())
        self.assertEqual([e['question_index'] for e in events if e['event']=='extra_mutation_batch'],[0,1])

    def test_input_empty_not_retried(self):
        native = NativeMutation([])
        with self.assertRaises(EmptyCandidateInput):
            MutationGuard(native, [])(batch(0))
        self.assertEqual(native.calls, 0)

    def test_unrelated_exception_propagates_without_retry(self):
        failure = IndexError("neutral unrelated error")
        native = NativeMutation([failure])
        with self.assertRaises(IndexError) as caught:
            MutationGuard(native, [])(batch())
        self.assertIs(caught.exception, failure)
        self.assertEqual(native.calls, 1)

    def test_exception_during_recovery_preserved_and_recorded(self):
        failure = RuntimeError("neutral")
        native = NativeMutation([batch(0), failure])
        events = []
        with self.assertRaises(RuntimeError) as caught:
            MutationGuard(native, events)(batch())
        self.assertIs(caught.exception, failure)
        self.assertEqual(events[-1]['status'], 'exception')

    def test_template_setter_delegates_to_native(self):
        native = NativeMutation([])
        guard = MutationGuard(native, [])
        guard.system_prompt = "neutral replacement"
        self.assertEqual(native.system_prompt, "neutral replacement")

    def test_recovery_does_not_require_generic_deepcopy(self):
        class NoDeepcopy:
            query = "neutral"
            jailbreak_prompt = "neutral"
            history = []
            def __deepcopy__(self, memo):
                raise RecursionError("generic deepcopy must not be used")
        first, second = JailbreakDataset([]), JailbreakDataset([SimpleNamespace()])
        native = NativeMutation([first, second])
        result = MutationGuard(native, [])(JailbreakDataset([NoDeepcopy()]))
        self.assertIs(result, second)
        self.assertEqual(native.calls, 2)


if __name__ == '__main__':
    unittest.main()
