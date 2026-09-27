"""Runtime-only feedback ablations without editing the uploaded legacy source."""
from contextlib import contextmanager
import hashlib
import inspect
import textwrap


CORE_FITNESS = "        fitness = calculate_fitness_score(weak_info, strong_label)\n"
CORE_FITNESS_REPLACEMENT = (
    "        fitness_label = _q20_effective_label(strong_label)\n"
    "        fitness = calculate_fitness_score(weak_info, fitness_label)\n"
)
CORE_SEED = (
    "        elif strong_label in (1, 2) or "
    "(strong_label is None and fitness > 2):\n"
)
CORE_SEED_REPLACEMENT = (
    "        elif fitness_label in (1, 2) or "
    "(fitness_label is None and fitness > 2):\n"
)
RAW_LABEL = 'label = getattr(pn, "strong_judge_label", None)'
EFFECTIVE_LABEL = "label = self._effective_label(pn)"
SELECTION_METHODS = (
    "_boundary_score", "_boundary_nodes", "_ra_bonus",
    "_semantic_score", "_elite_score", "_elite_nodes",
)


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_exact(source, old, new, expected=1):
    if source.count(old) != expected:
        raise RuntimeError("feedback_ablation_source_context_mismatch")
    return source.replace(old, new)


def _compile_method(method, module, replacements):
    source = textwrap.dedent(inspect.getsource(method))
    transformed = source
    for old, new, expected in replacements:
        transformed = _replace_exact(transformed, old, new, expected)
    namespace = {}
    exec(compile(transformed, str(inspect.getsourcefile(method)), "exec"),
         module.__dict__, namespace)
    return namespace[method.__name__], {
        "method": method.__name__,
        "source_sha256": _sha256(source),
        "transformed_sha256": _sha256(transformed),
    }


@contextmanager
def disable_label1_reward(core_module=None, selection_module=None):
    """Map online Label-1 to no label only in reward/selection decisions.

    Raw labels remain recorded for audit and offline evaluation. This reproduces
    the historical ``GPTFUZZ_DISABLE_LABEL1_REWARD`` semantics while keeping the
    uploaded original source files byte-for-byte unchanged.
    """
    if core_module is None:
        from gptfuzzer.fuzzer import core as core_module
    if selection_module is None:
        from gptfuzzer.fuzzer import selection as selection_module

    fuzzer_cls = core_module.GPTFuzzer
    select_cls = selection_module.MCTSRAESSelectPolicy
    original_core = fuzzer_cls.evaluate_ra
    original_effective = select_cls.__dict__.get("_effective_label")
    original_selection = {
        name: getattr(select_cls if name in {
            "_boundary_score", "_boundary_nodes", "_ra_bonus"
        } else selection_module.HybridRAESSelectPolicy, name)
        for name in SELECTION_METHODS
    }
    observation = {
        "policy": "label1_to_unjudged_for_reward_and_selection_v1",
        "raw_labels_preserved": True,
        "core_label1_remaps": 0,
        "selection_label1_remaps": 0,
        "transforms": [],
    }

    def effective_raw_label(label):
        if label == 1:
            observation["core_label1_remaps"] += 1
            return None
        return label

    def effective_node_label(pn):
        label = getattr(pn, "strong_judge_label", None)
        if label == 1:
            observation["selection_label1_remaps"] += 1
            return None
        return label

    try:
        core_method, info = _compile_method(original_core, core_module, (
            (CORE_FITNESS, CORE_FITNESS_REPLACEMENT, 1),
            (CORE_SEED, CORE_SEED_REPLACEMENT, 1),
        ))
        observation["transforms"].append(info)
        core_module._q20_effective_label = effective_raw_label
        fuzzer_cls.evaluate_ra = core_method
        select_cls._effective_label = staticmethod(effective_node_label)

        for name in SELECTION_METHODS:
            owner = (select_cls if name in {
                "_boundary_score", "_boundary_nodes", "_ra_bonus"
            } else selection_module.HybridRAESSelectPolicy)
            method, info = _compile_method(
                original_selection[name], selection_module,
                ((RAW_LABEL, EFFECTIVE_LABEL, 1),),
            )
            setattr(owner, name, method)
            observation["transforms"].append(info)
        yield observation
    finally:
        fuzzer_cls.evaluate_ra = original_core
        if hasattr(core_module, "_q20_effective_label"):
            delattr(core_module, "_q20_effective_label")
        if original_effective is None:
            if "_effective_label" in select_cls.__dict__:
                delattr(select_cls, "_effective_label")
        else:
            setattr(select_cls, "_effective_label", original_effective)
        for name, method in original_selection.items():
            owner = (select_cls if name in {
                "_boundary_score", "_boundary_nodes", "_ra_bonus"
            } else selection_module.HybridRAESSelectPolicy)
            setattr(owner, name, method)
