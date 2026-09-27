"""TAP parser/empty/singleton compatibility; no upstream source or prompt edits.

At most one extra mutation batch per question, only after an empty result.
Native per-branch retries remain unchanged. Exhaustion is an error, not a label.
Normal candidate batches use the original mutator and constraint unchanged.
"""
import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

VARIANT = "native_tap_cap10_fixed_structured_template_balanced_parser_v2"


class EmptyCandidateGeneration(RuntimeError):
    pass


class EmptyCandidateInput(RuntimeError):
    pass


class EmptyCandidateFilter(RuntimeError):
    pass


def counter(model):
    value = getattr(getattr(model, "generate", None), "calls", None)
    return value if isinstance(value, int) else None


def _balanced_mappings(text):
    """Yield balanced brace substrings while ignoring braces inside strings."""
    if not isinstance(text, str):
        return
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        quote = None
        escaped = False
        for end in range(start, len(text)):
            character = text[end]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'"}:
                quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:end + 1]
                    break
                if depth < 0:
                    break


def balanced_extract_json(text):
    """Return the first valid TAP mapping without changing its field values."""
    for block in _balanced_mappings(text):
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(block)
            except (TypeError, ValueError, SyntaxError):
                continue
            break
        if (isinstance(parsed, dict)
                and "improvement" in parsed
                and isinstance(parsed.get("prompt"), str)
                and parsed["prompt"].strip()):
            return parsed, block
    return None, None


def _validate_native_parser(native):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(native)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError("tap_parser_source_shape_mismatch") from exc
    constants = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    literal_eval_calls = [node for node in ast.walk(tree)
                          if (isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Attribute)
                              and isinstance(node.func.value, ast.Name)
                              and node.func.value.id == "ast"
                              and node.func.attr == "literal_eval")]
    find_calls = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "find"]
    if not {"improvement", "prompt"}.issubset(constants) or not literal_eval_calls or len(find_calls) < 2:
        raise RuntimeError("tap_parser_source_shape_mismatch")


def parser_compatibility(native, validate_source=True):
    """Prefer the pinned native parser, then try a balanced strict fallback."""
    if validate_source:
        _validate_native_parser(native)

    def compatible(text):
        native_result = native(text)
        if (isinstance(native_result, tuple) and len(native_result) == 2
                and native_result[0] is not None):
            compatible.native_accepted += 1
            return native_result
        result = balanced_extract_json(text)
        if result[0] is not None:
            compatible.compat_accepted += 1
        else:
            compatible.rejected += 1
        return result

    compatible.native_accepted = 0
    compatible.compat_accepted = 0
    compatible.rejected = 0
    compatible._q20_balanced_parser = True
    compatible._q20_native_parser = native
    return compatible


def enable_parser_compat(mutator):
    globals_dict = mutator.get_attack.__globals__
    native = globals_dict.get("extract_json")
    if not callable(native):
        raise RuntimeError("tap_parser_not_found")
    if getattr(native, "_q20_balanced_parser", False):
        parser = native
    else:
        parser = parser_compatibility(native)
        globals_dict["extract_json"] = parser
    return parser


def parser_counts(parser):
    return {name: int(getattr(parser, name)) for name in
            ("native_accepted", "compat_accepted", "rejected")}


class MutationGuard:
    def __init__(self, native, events):
        self.native, self.events = native, events
        self.begin_question(None)

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def system_prompt(self):
        return self.native.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.native.system_prompt = value

    def begin_question(self, index):
        self.question_index = index
        self.recoveries_used = 0

    def __call__(self, dataset, *args, **kwargs):
        if len(dataset) == 0:
            raise EmptyCandidateInput("tap_empty_mutation_input")
        before = counter(self.native.model)
        result = self.native(dataset, *args, **kwargs)
        if len(result):
            return result
        after = counter(self.native.model)
        self.events.append(dict(event="empty_mutation_output", question_index=self.question_index,
                                input_candidates=len(dataset), mutation_calls_before=before,
                                mutation_calls_after=after))
        if self.recoveries_used >= 1:
            raise EmptyCandidateGeneration("tap_empty_recovery_exhausted")
        self.recoveries_used += 1
        event = dict(event="extra_mutation_batch", question_index=self.question_index,
                     recovery_number=self.recoveries_used, mutation_calls_before=after,
                     mutation_calls_after=None, output_candidates=None, status="started")
        self.events.append(event)
        try:
            # The native empty path has no accepted child to feed forward. Reuse
            # the same batch, as the next native attempt would, without copying
            # EasyJailbreak Instance objects (their dynamic attributes are not
            # compatible with generic deepcopy in the pinned dependency).
            result = self.native(dataset, *args, **kwargs)
        except BaseException:
            event.update(status="exception", mutation_calls_after=counter(self.native.model))
            raise
        event.update(output_candidates=len(result), mutation_calls_after=counter(self.native.model),
                     status="recovered" if len(result) else "empty_exhausted")
        if not len(result):
            raise EmptyCandidateGeneration("tap_empty_recovery_exhausted")
        return result


def singleton_callable(native_function):
    """Change only the inspected two-element fallback, in memory for len==1.

Fail closed on a different source shape. The local dependency file is untouched.
All scoring, shuffle, sorting, and positive-score selection remain native.
"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(native_function)))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].decorator_list:
        raise RuntimeError("tap_constraint_source_shape_mismatch")
    old = ast.parse("[tuples_list[0][1], tuples_list[1][1]]", mode="eval").body
    replacements = 0
    for node in ast.walk(functions[0]):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "truncated_list"
                and ast.dump(node.value) == ast.dump(old)):
            node.value = ast.copy_location(ast.parse(
                "[item[1] for item in tuples_list[:2]]", mode="eval").body, node.value)
            replacements += 1
    if replacements != 1:
        raise RuntimeError("tap_constraint_source_shape_mismatch")
    functions[0].name = "_q20_singleton_call"
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, "<q20_singleton_constraint>", "exec"), native_function.__globals__, namespace)
    return namespace["_q20_singleton_call"]


class ConstraintGuard:
    def __init__(self, native, events):
        self.native, self.events = native, events
        self.singleton = singleton_callable(type(native).__call__)
        self.question_index = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, dataset, *args, **kwargs):
        if not len(dataset):
            raise EmptyCandidateInput("tap_empty_constraint_input")
        if len(dataset) == 1:
            self.events.append(dict(event="singleton_constraint_input", question_index=self.question_index))
            result = self.singleton(self.native, dataset, *args, **kwargs)
        else:
            result = self.native(dataset, *args, **kwargs)
        if not len(result):
            raise EmptyCandidateFilter("tap_empty_constraint_output")
        return result


def install(recipe):
    events = []
    recipe.q20_tap_parser = enable_parser_compat(recipe.mutator)
    recipe.q20_tap_parser_start = parser_counts(recipe.q20_tap_parser)
    recipe.constraint = ConstraintGuard(recipe.constraint, events)
    recipe.mutator = MutationGuard(recipe.mutator, events)
    recipe.q20_tap_events = events


def begin_question(recipe, index):
    recipe.mutator.begin_question(index)
    recipe.constraint.question_index = index


def report(recipe):
    current = parser_counts(recipe.q20_tap_parser)
    start = recipe.q20_tap_parser_start
    return dict(
        variant=VARIANT,
        parser_policy="native_first_then_balanced_json_or_literal_mapping_v1",
        parser_counts={name: current[name] - start[name] for name in current},
        template_profile=getattr(recipe, "q20_tap_template_profile", None),
        extra_mutation_batches_per_question_cap=1,
        mutation_generate_calls=counter(recipe.mutator.native.model),
        events=recipe.q20_tap_events,
    )


def runtime_audit(recipe_factory, dataset_class, instance_class):
    """Real installed mutator/parser/constraint, but neutral fake API replies.

No target model call, real API call, or complete TAP search. Audit objects only.
"""
    parser_recipe = recipe_factory()
    parser = parser_recipe.q20_tap_parser
    parser_cases = {
        "native_mapping": '{"improvement":"neutral","prompt":"Return a neutral greeting."}',
        "quoted_brace": '{"improvement":"neutral","prompt":"Return literal {token} safely."}',
        "json_null": '{"improvement":"neutral","prompt":"Return a neutral greeting.","note":null}',
        "python_literal": "{'improvement':'neutral','prompt':'Return a neutral greeting.'}",
    }
    for name, value in parser_cases.items():
        parsed, _ = parser(value)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("prompt"), str):
            raise RuntimeError("tap_parser_runtime_mismatch")
    for value in (
        "neutral text without a mapping",
        '{"improvement":"neutral"}',
        '{"improvement":"neutral","prompt":"unfinished"',
    ):
        if parser(value) != (None, None):
            raise RuntimeError("tap_parser_runtime_mismatch")

    completed = []
    for case in ("empty_then_valid", "empty_exhausted", "two_candidate_fallback"):
        recipe = recipe_factory()
        mutator = recipe.mutator.native
        mutator.branching_factor = 1  # This test object only, never formal settings.
        attempts = mutator.max_n_attack_attempts
        field = mutator.trans_dict2[mutator.attr_name]
        good = json.dumps({"improvement": "neutral", field: "Return a neutral greeting."})
        def fake_generate(*args, **kwargs):
            fake_generate.calls += 1
            if case == "empty_exhausted" or fake_generate.calls <= attempts:
                return "neutral invalid JSON"
            return good
        fake_generate.calls = 0
        mutator.model.generate = fake_generate
        constraint = recipe.constraint.native
        constraint.system_prompt = "{query}"
        constraint.eval_model = SimpleNamespace(
            conversation=SimpleNamespace(messages=[]),
            set_system_message=lambda text: None, generate=lambda text: "neutral score")
        constraint.get_evaluator_prompt_on_topic = lambda text: "neutral evaluator input"
        constraint.process_output_on_topic_score = lambda text: 0
        constraint.tree_width = 2
        instance = instance_class(query="Return a neutral greeting.",
                                  reference_responses=["Hello."],
                                  jailbreak_prompt="Return a neutral greeting.")
        dataset = dataset_class([instance])
        begin_question(recipe, 0)
        if case == "two_candidate_fallback":
            result = recipe.constraint(dataset_class([instance, instance.copy()]))
            if len(result) != 2 or fake_generate.calls:
                raise RuntimeError("tap_guard_runtime_mismatch")
        elif case == "empty_exhausted":
            try:
                recipe.mutator(dataset)
            except EmptyCandidateGeneration:
                if fake_generate.calls != 2 * attempts:
                    raise RuntimeError("tap_guard_runtime_mismatch")
            else:
                raise RuntimeError("tap_guard_runtime_mismatch")
        else:
            generated = recipe.mutator(dataset)
            result = recipe.constraint(generated)
            if len(generated) != 1 or len(result) != 1 or fake_generate.calls != attempts + 1:
                raise RuntimeError("tap_guard_runtime_mismatch")
        completed.append(case)
    return dict(method="tap", candidate_guard_mock_cases=completed,
                parser_compat_mock_cases=list(parser_cases) + [
                    "no_mapping", "missing_prompt", "unclosed_mapping"],
                real_api_calls=0, target_calls=0,
                scope="installed_mutator_parser_constraint_with_neutral_replies_not_end_to_end")
