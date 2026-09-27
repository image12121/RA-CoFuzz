"""ReNeLLM per-question driver, transcribed from the server's outer loop.

Preserve native single_attack(instance)[0], evaluator(dataset), and label == True.
The original instance is reused (not replaced by the returned child), exactly as
in the inspected source. Only the target-call cap bounds the native evo_max.
Final aggregate reporting remains the extension's recorded-response pipeline.
"""
from types import SimpleNamespace
from unittest.mock import patch

from support import BudgetReached

VARIANT = "native_renellm_feedback_loop_per_question_cap10_v1"


def run_question(recipe, instance, recorder, dataset_class):
    native_limit = recipe.evo_max
    if isinstance(native_limit, bool) or not isinstance(native_limit, int) or native_limit < 1:
        raise ValueError("invalid_renellm_evo_max")
    rounds = evaluations = 0
    for _ in range(native_limit):
        # Avoid starting another mutation/API stage once target budget is spent.
        if recorder.q_calls >= recorder.per_question or recorder.calls >= recorder.budget:
            return dict(rounds=rounds, evaluations=evaluations, stop="target_cap")
        before = recorder.calls
        try:
            new_instance = recipe.single_attack(instance)[0]
        except BudgetReached:
            return dict(rounds=rounds, evaluations=evaluations, stop="target_cap")
        if recorder.calls == before:
            raise RuntimeError("renellm_round_without_target_call")
        rounds += 1
        recipe.evaluator(dataset_class([new_instance]))
        evaluations += 1
        if not new_instance.eval_results:
            raise ValueError("missing_renellm_feedback")
        if new_instance.eval_results[0] == True:
            return dict(rounds=rounds, evaluations=evaluations, stop="native_success")
    return dict(rounds=rounds, evaluations=evaluations, stop="native_evo_max")


class _Dataset(list):
    def add(self, item):
        self.append(item)


def neutral_probe(labels, evo_max, question_count=1):
    """All stages are neutral fakes; no model objects, clients or credentials."""
    events = []
    originals = [SimpleNamespace(index=i, query="neutral", visits=0)
                 for i in range(question_count)]
    recorder = SimpleNamespace(calls=0, q_calls=0, per_question=10, budget=200)
    def single(instance):
        if instance is not originals[instance.index]:
            raise RuntimeError("original_instance_not_preserved")
        instance.visits += 1
        recorder.calls += 1
        recorder.q_calls += 1
        events.append(("generate", instance.index, instance.visits))
        return _Dataset([SimpleNamespace(index=instance.index, visit=instance.visits,
                                         eval_results=[])])
    def evaluate(dataset):
        item = dataset[0]
        value = labels[min(item.visit - 1, len(labels) - 1)]
        item.eval_results.append(value)
        events.append(("evaluate", item.index, item.visit, value))
    recipe = SimpleNamespace(evo_max=evo_max, jailbreak_datasets=_Dataset(originals),
                             single_attack=single, evaluator=evaluate,
                             update=lambda data: None, log=lambda: None)
    return recipe, originals, recorder, events


def native_outer_audit(native_attack):
    """Compare real installed outer control flow against this bounded driver.

Mocks replace generation, evaluation, datasets and logging only for this probe.
This is NOT a complete runtime/mutation/API test. The native loop is bounded at
10 rounds for comparison to the target cap under one target call per round.
"""
    cases = [([True], 20, 1), ([False, False, True], 20, 1),
             ([False], 20, 1), ([False], 3, 1), ([False, True], 20, 2),
             ([False, 1], 20, 1)]
    for labels, limit, count in cases:
        native, _, _, expected = neutral_probe(labels, min(limit, 10), count)
        replacements = {"JailbreakDataset": _Dataset,
                        "tqdm": lambda values, **kwargs: values,
                        "logging": SimpleNamespace(info=lambda *args, **kwargs: None)}
        with patch.dict(native_attack.__globals__, replacements):
            native_attack(native)
        adapted, instances, recorder, actual = neutral_probe(labels, limit, count)
        for instance in instances:
            recorder.q_calls = 0
            run_question(adapted, instance, recorder, _Dataset)
        if expected != actual:
            raise RuntimeError("renellm_native_outer_flow_mismatch")
    return dict(method="renellm", native_outer_mock_cases=len(cases),
                real_api_calls=0, gpu_calls=0,
                scope="outer_control_flow_only_not_end_to_end")
