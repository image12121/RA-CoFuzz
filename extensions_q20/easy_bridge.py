"""External recipes use the original project's LocalLLM for local inference.

The three recipes are budget-bounded adaptations, not unmodified upstream runs.
No external text or source templates are bundled in this adapter.
"""
import importlib
import hashlib
import inspect
import json
import os
from functools import wraps
from types import SimpleNamespace

from support import BudgetReached, Recorder, normalize_tap

RECIPES = {
    "tap": ("TAP_Mehrotra_2023", "TAP"),
    "renellm": ("ReNeLLM_ding_2023", "ReNeLLM"),
    "deepinception": ("DeepInception_Li_2023", "DeepInception"),
}


def select_tap_system_prompt(templates):
    """Select the one TAP template that satisfies the installed parser contract."""
    accepted = {}
    for template in templates:
        if not isinstance(template, str):
            continue
        normalized = normalize_tap(template)
        required = all(
            ('"' + key + '"') in normalized or ("'" + key + "'") in normalized
            for key in ("improvement", "prompt")
        )
        if required:
            accepted[normalized] = None
    if len(accepted) != 1:
        raise RuntimeError("tap_structured_template_contract_mismatch")
    selected = next(iter(accepted))
    return selected, {
        "policy": "unique_template_with_improvement_and_prompt_keys_v1",
        "sha256": hashlib.sha256(selected.encode()).hexdigest(),
        "length": len(selected),
    }


def tap_templates():
    from easyjailbreak.seed.seed_template import SeedTemplate
    return SeedTemplate().new_seeds(seeds_num=None, method_list=["TAP"])


def ensure_generation_counter(model):
    """Supply the .calls interface used by TAP's post-generation statistics.

    Counts Python generate invocations, not provider retries, successful answers,
    or target budget use. Existing counters and exceptions are left intact.
    """
    original = model.generate
    if hasattr(original, "calls"):
        return

    @wraps(original)
    def counted(*args, **kwargs):
        counted.calls += 1
        return original(*args, **kwargs)

    counted.calls = 0
    model.generate = counted


def recipe_class(method):
    module, name = RECIPES[method]
    return getattr(importlib.import_module("easyjailbreak.attacker." + module), name)


def model_classes():
    from easyjailbreak.models.huggingface_model import HuggingfaceModel
    from easyjailbreak.models.openai_model import OpenaiModel
    from fastchat.conversation import get_conv_template

    class OriginalTarget(HuggingfaceModel):
        def __init__(self, local, recorder):
            # Keep isinstance compatibility, but use only original LocalLLM.generate.
            self.local, self.recorder = local, recorder
            self.model = local.model
            self.tokenizer = local.tokenizer
            self.model_name = "original_local_target"
            self.generation_config = {}
            self.conversation = get_conv_template("one_shot")

        def generate(self, messages, **kwargs):
            if isinstance(messages, list) and len(messages) == 1:
                messages = messages[0]
            return self.recorder.generate(self.local.generate, messages)

        def batch_generate(self, conversations, **kwargs):
            return [self.generate(item, **kwargs) for item in conversations]

    return OriginalTarget, OpenaiModel, get_conv_template


def construct(method, attack, target, evaluator, dataset):
    cls = recipe_class(method)
    supplied = dict(attack_model=attack, target_model=target,
                    eval_model=evaluator, jailbreak_datasets=dataset)
    signature = inspect.signature(cls)
    signature.bind(**supplied)  # No silently dropped constructor arguments.
    result = cls(**supplied)
    if not callable(getattr(result, "single_attack", None)):
        raise TypeError("single_question_entry_unavailable")
    if method == "tap":
        selected, profile = select_tap_system_prompt(tap_templates())
        result.mutator.system_prompt = selected
        result.q20_tap_template_profile = profile
        for model in (attack, evaluator, result.evaluator.eval_model):
            ensure_generation_counter(model)
        ensure_generation_counter(result.mutator.model)
        from tap_compat import install
        install(result)
    return result


def prepare_api(OpenaiModel):
    from openai import OpenAI
    api_key = os.environ["MUTATION_API_KEY"]
    endpoint = os.environ["MUTATION_API_BASE_URL"]
    name = os.environ["MUTATION_API_MODEL_NAME"]
    # Pinned OpenaiModel accepts api_keys, not the SDK's api_key spelling.
    result = OpenaiModel(model_name=name, api_keys=api_key)
    result.client = OpenAI(api_key=api_key, base_url=endpoint, timeout=120, max_retries=2)
    return result


def run(method, rows, target_path, output, progress):
    from easyjailbreak.datasets import Instance, JailbreakDataset
    from gptfuzzer.llm import LocalLLM
    Target, Api, _ = model_classes()
    local = LocalLLM(target_path, max_gpu_memory="22GiB", local_files_only=True,
                     default_max_new_tokens=128, default_batch_size=1)
    recorder = Recorder(output, budget=200, per_question=10)
    progress["recorder"] = recorder
    target = Target(local, recorder)
    attack, evaluator = prepare_api(Api), prepare_api(Api)
    instances = [Instance(query=r["text"],
                          reference_responses=[r["target"]] if r.get("target") else [])
                 for r in rows]
    dataset = JailbreakDataset(instances)
    recipe = construct(method, attack, target, evaluator, dataset)
    question_flow = []
    try:
        for index, instance in enumerate(instances):
            recorder.begin(index, instance.query)
            if method == "tap":
                from tap_compat import begin_question
                begin_question(recipe, index)
            try:
                if method == "renellm":
                    from renellm_flow import run_question
                    question_flow.append(dict(question_index=index, **run_question(
                        recipe, instance, recorder, JailbreakDataset)))
                else:
                    recipe.single_attack(instance)
            except BudgetReached:
                pass  # Explicit per-question cap; never count it as a success.
    finally:
        recorder.close()
        if method == "tap":
            from tap_compat import report
            with output.with_name("tap_compat_events.json").open("x", encoding="utf-8") as stream:
                json.dump(report(recipe), stream, indent=2)
    if not recorder.completed or recorder.calls != recorder.completed:
        raise RuntimeError("incomplete_target_recording")
    metadata = dict(actual_target_calls=recorder.calls, recorded_responses=recorder.completed,
                    per_question_target_cap=10,
                    variant="native_single_attack_per_question_cap10")
    if method == "renellm":
        from renellm_flow import VARIANT
        metadata.update(variant=VARIANT, question_flow=question_flow,
                        native_evo_max=recipe.evo_max,
                        feedback_evaluations=sum(q["evaluations"] for q in question_flow))
    elif method == "tap":
        from tap_compat import report
        metadata.update(report(recipe))
    return metadata


def boundary_audit():
    """No API, no GPU. Check real constructors until their first generate call.

    This intentionally does not claim end-to-end algorithm validation.
    """
    from easyjailbreak.datasets import Instance, JailbreakDataset
    Target, Api, conversation = model_classes()

    class BoundaryReached(BaseException):
        pass

    def blocked(*args, **kwargs):
        raise BoundaryReached()

    def fake_api():
        api = Api.__new__(Api)
        api.model_name = "gpt-3.5-turbo"
        api.generation_config = {}
        api.conversation = conversation("chatgpt")
        api.generate = blocked
        api.batch_generate = blocked
        return api

    fake_local = SimpleNamespace(model=SimpleNamespace(),
                                 tokenizer=SimpleNamespace(eos_token_id=0), generate=blocked)
    def boundary_target_generate(generator, prompt):
        if not isinstance(prompt, str):
            raise TypeError("unsupported_target_message_type")
        return generator(prompt)

    target = Target(fake_local, SimpleNamespace(generate=boundary_target_generate))
    selected_template, selected_profile = select_tap_system_prompt(tap_templates())
    checked = []
    for method in RECIPES:
        variants = [selected_template] if method == "tap" else [None]
        for index, template in enumerate(variants):
            instance = Instance(query="Return a neutral greeting.", reference_responses=["Hello."])
            recipe = construct(method, fake_api(), target, fake_api(), JailbreakDataset([instance]))
            if method == "tap":
                # Exercise the exact attribute read missed by the earlier audit.
                counted = recipe.evaluator.eval_model.generate
                before = counted.calls
                try:
                    counted("Return a neutral greeting.")
                except BoundaryReached:
                    pass
                if counted.calls != before + 1:
                    raise RuntimeError("generation_counter_check_failed")
            if template is not None:
                if recipe.mutator.system_prompt != template:
                    raise RuntimeError("tap_selected_template_mismatch")
            try:
                recipe.single_attack(instance)
            except BoundaryReached:
                checked.append(dict(method=method, template_index=index,
                                    statistics_counter_checked=(method == "tap"),
                                    structured_template_checked=(
                                        selected_profile if method == "tap" else None)))
            else:
                raise RuntimeError("recipe_did_not_reach_generation_boundary")
    from renellm_flow import native_outer_audit
    checked.append(native_outer_audit(recipe_class("renellm").attack))
    from tap_compat import runtime_audit
    def tap_factory():
        neutral = Instance(query="Return a neutral greeting.", reference_responses=["Hello."])
        return construct("tap", fake_api(), target, fake_api(), JailbreakDataset([neutral]))
    checked.append(runtime_audit(tap_factory, JailbreakDataset, Instance))
    return checked
