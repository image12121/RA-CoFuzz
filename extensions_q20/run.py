#!/usr/bin/env python3
"""Incremental entry points on top of the uploaded original source tree."""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import traceback

from support import (ROOT, LEGACY, EXT, OUTPUT, PROTOCOL, digest,
                     generation_signature, read_questions, verify_original)

CONSOLE = None
SAFE_CODES = {
    "target_environment_variable_missing", "api_configuration_missing",
    "run_boundary_audit_first", "boundary_audit_is_stale",
    "completed_cell_mismatch_preserved_for_review",
    "incomplete_cell_preserved_choose_archive_manually", "generation_not_complete",
    "generation_artifact_changed", "evaluation_artifact_changed",
    "partial_evaluation_preserved_choose_archive_manually", "legacy_evaluation_incomplete",
    "legacy_judge_error_record", "target_column_file_required", "empty_reference",
    "question_count_mismatch", "paired_question_order_mismatch",
    "reference_provenance_mismatch",
    "original_source_or_dataset_mismatch", "invalid_external_dependency_path",
    "unsupported_target_message_type", "unsupported_template_fields",
    "single_question_entry_unavailable", "recipe_did_not_reach_generation_boundary",
    "local_model_config_missing", "incomplete_target_recording", "native_full_budget_not_recorded",
    "invalid_selection_ablation_scope", "invalid_selection_ablation_config",
    "selection_ablation_no_candidate", "selection_ablation_disabled_branch_used",
    "invalid_feedback_ablation_scope", "invalid_feedback_ablation_config",
    "multiple_ablation_families_selected", "feedback_ablation_source_context_mismatch",
    "feedback_ablation_online_call_mismatch", "feedback_ablation_transform_incomplete",
    "empty_policy_row_count_mismatch", "empty_policy_row_index_mismatch",
    "empty_policy_source_alignment_mismatch", "empty_policy_prompt_also_missing",
    "empty_policy_unexpected_empty_record", "empty_policy_nonempty_record_invalid",
    "empty_policy_summary_not_exactly_repairable",
    "empty_policy_postcondition_unknown_label",
}


def emit(**payload):
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    if CONSOLE is None:
        print(data.decode().rstrip(), flush=True)
    else:
        os.write(CONSOLE, data)


@contextlib.contextmanager
def suppress_original_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [sink]:
            os.close(fd)


def dependencies():
    sys.path.insert(0, str(LEGACY))
    easy = os.getenv("EASYJAILBREAK_ROOT")
    if easy:
        if not (Path(easy) / "easyjailbreak").is_dir():
            raise ValueError("invalid_external_dependency_path")
        sys.path.insert(1, str(Path(easy).resolve()))
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)


def clear_method_env():
    for key in list(os.environ):
        if key.startswith("GPTFUZZ_"):
            os.environ.pop(key)
    os.environ.update({
        "GPTFUZZ_HYBRID_WARMUP_STEPS": "2", "GPTFUZZ_HYBRID_MCTS_RATIO": "0.50",
        "GPTFUZZ_HYBRID_DP_RATIO": "0.30", "GPTFUZZ_HYBRID_ELITE_RATIO": "0.20",
        "GPTFUZZ_HYBRID_DP_ALPHA": "0.5", "GPTFUZZ_HYBRID_DP_EPSILON": "0.10",
        "GPTFUZZ_HYBRID_DP_TOPK": "5", "GPTFUZZ_HYBRID_ELITE_TOPK": "5",
    })


def selection_ablation_config(args):
    """Return a validated, protocol-pinned selection ablation configuration."""
    if args.ablation is None:
        return None
    if (args.method != "ra_cofuzz" or args.dataset != "gptfuzzer" or
            args.model != "llama32_3b"):
        raise ValueError("invalid_selection_ablation_scope")
    raw = PROTOCOL["selection_ablations"][args.ablation]
    config = {
        "warmup_steps": int(raw["warmup_steps"]),
        "mcts_ratio": float(raw["mcts_ratio"]),
        "dp_ratio": float(raw["dp_ratio"]),
        "elite_ratio": float(raw["elite_ratio"]),
        "disabled_branch": str(raw["disabled_branch"]),
        "purpose": str(raw["purpose"]),
    }
    ratios = [config[name] for name in ("mcts_ratio", "dp_ratio", "elite_ratio")]
    branches = ("mcts", "dp", "elite")
    if (config["warmup_steps"] < 0 or any(value < 0.0 for value in ratios) or
            abs(sum(ratios) - 1.0) > 1e-9 or ratios.count(0.0) != 1 or
            config["disabled_branch"] not in branches or
            ratios[branches.index(config["disabled_branch"])] != 0.0):
        raise ValueError("invalid_selection_ablation_config")
    return config


def apply_selection_ablation(config):
    if config is None:
        return
    os.environ.update({
        "GPTFUZZ_HYBRID_WARMUP_STEPS": str(config["warmup_steps"]),
        "GPTFUZZ_HYBRID_MCTS_RATIO": format(config["mcts_ratio"], ".2f"),
        "GPTFUZZ_HYBRID_DP_RATIO": format(config["dp_ratio"], ".2f"),
        "GPTFUZZ_HYBRID_ELITE_RATIO": format(config["elite_ratio"], ".2f"),
    })


def feedback_ablation_config(args):
    """Return a validated, protocol-pinned feedback ablation configuration."""
    name = getattr(args, "feedback_ablation", None)
    if name is None:
        return None
    if getattr(args, "ablation", None) is not None:
        raise ValueError("multiple_ablation_families_selected")
    if (args.method != "ra_cofuzz" or args.dataset != "gptfuzzer" or
            args.model != "llama32_3b"):
        raise ValueError("invalid_feedback_ablation_scope")
    config = dict(PROTOCOL["feedback_ablations"][name])
    valid = (
        isinstance(config.get("strong_judge_in_loop"), bool)
        and isinstance(config.get("force_review"), bool)
        and config.get("label1_reward") in {"unchanged", "disabled"}
        and config.get("heuristic_partial_reward") == "unchanged"
        and config.get("expected_online_calls") in {
            "selective_0_to_200", "exactly_0", "exactly_200"
        }
        and bool(config.get("purpose"))
    )
    if (config.get("force_review") and not config.get("strong_judge_in_loop")):
        valid = False
    if not valid:
        raise ValueError("invalid_feedback_ablation_config")
    return config


def output_cell(args):
    if getattr(args, "feedback_ablation", None) is not None:
        return (OUTPUT / "feedback_ablations" / args.feedback_ablation /
                args.dataset / args.model / f"seed{args.seed}")
    if getattr(args, "ablation", None) is not None:
        return (OUTPUT / "selection_ablations" / args.ablation / args.dataset /
                args.model / f"seed{args.seed}")
    return OUTPUT / args.method / args.dataset / args.model / f"seed{args.seed}"


def status_fields(args):
    fields = {"method": args.method, "seed": args.seed}
    if getattr(args, "ablation", None) is not None:
        fields["selection_ablation"] = args.ablation
    if getattr(args, "feedback_ablation", None) is not None:
        fields["feedback_ablation"] = args.feedback_ablation
    return fields


def legacy_run(args, question_path, cell, target, ablation=None, feedback=None):
    if args.method == "pair":
        for role, prefix in (("ATTACK", "MUTATION"), ("JUDGE", "RACOFUZZ")):
            os.environ[f"PAIR_{role}_API_KEY"] = os.environ[prefix + "_API_KEY"]
            os.environ[f"PAIR_{role}_BASE_URL"] = os.environ[prefix + "_API_BASE_URL"]
            os.environ[f"PAIR_{role}_MODEL"] = os.environ[prefix + "_API_MODEL_NAME"]
        command = [sys.executable, str(ROOT / "pair_official_autodl/pair_official_autodl.py"),
                   "--official-pair-root", str(ROOT / "JailbreakingLLMs-official"),
                   "--gptfuzz-root", str(LEGACY), "--question-path", str(question_path),
                   "--require-targets", "--target-model", target,
                   "--output-jsonl", str(cell / "candidates.jsonl"),
                   "--summary-json", str(cell / "native_summary.json"),
                   "--run-id", str(PROTOCOL["seeds"].index(args.seed) + 1),
                   "--random-seed", str(args.seed), "--n-streams", "2",
                   "--n-iterations", "5", "--keep-last-n", "4",
                   "--target-max-new-tokens", "128", "--target-max-gpu-memory", "22GiB"]
        subprocess.run(command, cwd=cell, check=True)
        return {}
    # Import the original entry function. Restore its legacy import-time GPU/RNG overrides.
    spec = importlib.util.spec_from_file_location("original_entry", LEGACY / "gptfuzz.py")
    module = importlib.util.module_from_spec(spec)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    spec.loader.exec_module(module)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ra = args.method == "ra_cofuzz"
    judge_in_loop = (feedback["strong_judge_in_loop"]
                     if feedback is not None else ra)
    force_review = feedback["force_review"] if feedback is not None else False
    native_args = argparse.Namespace(
        seed_path=str(LEGACY / "datasets/prompts/GPTFuzzer.csv"),
        question_path=str(question_path), question_limit=-1,
        raes_mode=False, ra_mode=ra, strong_judge_in_loop=judge_in_loop,
        ra_force_review=force_review,
        selection_policy="hybrid_raes" if ra else "mcts", openai_key="",
        mutation_backend="openai_compatible", model_path="unused",
        target_backend="local_hf", target_model=target, target_max_gpu_memory="22GiB",
        target_cpu_offloading=False, target_local_files_only=True,
        target_max_new_tokens=128, target_batch_size=1, predictor_device="cpu",
        strong_judge_exploration_ratio=0.25, energy=1, max_jailbreak=-1, max_query=200,
        ra_result_file=str(cell / "candidates.jsonl"),
        ra_summary_file=str(cell / "native_summary.json"),
        log_all_candidates=True, all_candidates_file=str(cell / "candidates.jsonl"),
    )
    if feedback is not None and feedback["label1_reward"] == "disabled":
        from feedback_ablation import disable_label1_reward
        with disable_label1_reward() as observation:
            module.main(native_args)
        if len(observation.get("transforms", [])) != 7:
            raise ValueError("feedback_ablation_transform_incomplete")
        metadata = {"feedback_ablation_control": observation}
    elif ablation is None:
        module.main(native_args)
        metadata = {}
    else:
        from selection_ablation import strict_branch_ablation
        with strict_branch_ablation(ablation["disabled_branch"]) as observation:
            module.main(native_args)
        if (observation["disabled_branch_attempts"] != 0 or
                observation["disabled_branch_selections"] != 0):
            raise ValueError("selection_ablation_disabled_branch_used")
        metadata = {"selection_ablation_control": observation}

    if feedback is not None:
        summary = json.loads((cell / "native_summary.json").read_text(encoding="utf-8"))
        calls = int(summary.get("strong_judge_calls", -1))
        expectation = feedback["expected_online_calls"]
        valid_calls = (
            (expectation == "exactly_0" and calls == 0)
            or (expectation == "exactly_200" and calls == 200)
            or (expectation == "selective_0_to_200" and 0 <= calls <= 200)
        )
        if not valid_calls:
            raise ValueError("feedback_ablation_online_call_mismatch")
        metadata["online_feedback_control"] = {
            "strong_judge_in_loop": judge_in_loop,
            "force_review": force_review,
            "strong_judge_calls": calls,
            "expected_online_calls": expectation,
            "total_candidates": int(summary.get("total_candidates", -1)),
            "native_summary_sha256": digest(cell / "native_summary.json"),
        }
    return metadata


def count_rows(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or len(rows) > 200:
        raise ValueError("record_count_out_of_range")
    if any(not isinstance(r.get("response"), str) for r in rows):
        raise ValueError("missing_recorded_response")
    return len(rows)


def run_cell(args):
    verify_original()
    ablation = selection_ablation_config(args)
    feedback = feedback_ablation_config(args)
    env_name = PROTOCOL["models"][args.model]
    target = os.environ.get(env_name, "")
    if not target:
        raise ValueError("target_environment_variable_missing")
    signature = generation_signature(args.method, args.dataset, args.model, args.seed, target)
    if ablation is not None:
        signature["selection_ablation"] = args.ablation
        signature["selection_ablation_config"] = ablation
    if feedback is not None:
        signature["feedback_ablation"] = args.feedback_ablation
        signature["feedback_ablation_config"] = feedback
    question_path, rows = read_questions(args.dataset, args.method in {"pair", "tap"})
    for prefix in ("MUTATION", "RACOFUZZ"):
        for suffix in ("API_KEY", "API_BASE_URL", "API_MODEL_NAME"):
            if not os.environ.get(prefix + "_" + suffix):
                raise ValueError("api_configuration_missing")
    signature["api_models"] = {p: os.environ[p + "_API_MODEL_NAME"]
                               for p in ("MUTATION", "RACOFUZZ")}
    # Endpoints influence results; fingerprint without recording credentials/URLs.
    import hashlib
    signature["api_endpoints"] = {p: hashlib.sha256(os.environ[p + "_API_BASE_URL"].encode()).hexdigest()
                                  for p in ("MUTATION", "RACOFUZZ")}
    if args.method in PROTOCOL["new_methods"]:
        import easyjailbreak
        dep = Path(easyjailbreak.__file__).resolve().parent
        signature["external_source"] = {str(p.relative_to(dep)): digest(p)
                                         for p in sorted(dep.rglob("*"))
                                         if p.is_file() and p.suffix in {".py", ".json"}}
        stamp = EXT / "runtime_boundary_pass.json"
        if not stamp.is_file():
            raise ValueError("run_boundary_audit_first")
        saved = json.loads(stamp.read_text())
        if saved != {"external_source": signature["external_source"],
                     "bridge_sha256": digest(EXT / "easy_bridge.py"),
                     "bridge_helpers": {name: digest(EXT / name) for name in
                                        ("renellm_flow.py", "tap_compat.py")}}:
            raise ValueError("boundary_audit_is_stale")
    cell = output_cell(args)
    marker = cell / "generation.done.json"
    if marker.is_file():
        done = json.loads(marker.read_text())
        if done["signature"] != signature or done["candidates_sha256"] != digest(cell / "candidates.jsonl"):
            raise ValueError("completed_cell_mismatch_preserved_for_review")
        emit(status="SKIP_GENERATION", **status_fields(args))
        return
    if cell.exists() and any(cell.iterdir()):
        raise ValueError("incomplete_cell_preserved_choose_archive_manually")
    cell.mkdir(parents=True, exist_ok=True)
    progress, stop = {}, threading.Event()

    def heartbeat():
        while not stop.wait(30):
            recorder = progress.get("recorder")
            emit(status="RUNNING", target_calls=recorder.calls if recorder else None,
                 **status_fields(args))

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    start = time.monotonic()
    previous = Path.cwd()
    try:
        clear_method_env()
        apply_selection_ablation(ablation)
        os.chdir(cell)
        with suppress_original_output():
            random.seed(args.seed)
            import numpy as np
            import torch
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.method in PROTOCOL["new_methods"]:
                from easy_bridge import run
                metadata = run(args.method, rows, target, cell / "candidates.jsonl", progress)
            else:
                legacy_metadata = legacy_run(
                    args, question_path, cell, target, ablation, feedback)
                metadata = {"variant": "original_entry_with_explicit_seed",
                            **legacy_metadata}
        metadata["recorded_responses"] = count_rows(cell / "candidates.jsonl")
        if args.method in {"ra_cofuzz", "strict_gptfuzzer"} and metadata["recorded_responses"] != 200:
            raise ValueError("native_full_budget_not_recorded")
        metadata["runtime_seconds"] = round(time.monotonic() - start, 2)
        metadata["signature"] = signature
        metadata["candidates_sha256"] = digest(cell / "candidates.jsonl")
        marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        emit(status="GENERATION_PASS_EVALUATION_PENDING", **status_fields(args),
             recorded_responses=metadata["recorded_responses"], runtime_seconds=metadata["runtime_seconds"])
    finally:
        os.chdir(previous)
        stop.set()
        thread.join(timeout=1)


def evaluate_cell(args):
    verify_original()
    selection_ablation_config(args)
    feedback_ablation_config(args)
    cell = output_cell(args)
    marker = cell / "generation.done.json"
    if not marker.is_file():
        raise ValueError("generation_not_complete")
    saved = json.loads(marker.read_text())
    if digest(cell / "candidates.jsonl") != saved["candidates_sha256"]:
        raise ValueError("generation_artifact_changed")
    if (cell / "evaluation.done.json").exists():
        done = json.loads((cell / "evaluation.done.json").read_text())
        if any(digest(cell / name) != value for name, value in done["artifacts"].items()):
            raise ValueError("evaluation_artifact_changed")
        emit(status="SKIP_EVALUATION", **status_fields(args))
        return
    names = ["offline.jsonl", "offline_summary.json", "roberta.jsonl", "roberta_summary.json"]
    if any((cell / name).exists() for name in names):
        raise ValueError("partial_evaluation_preserved_choose_archive_manually")
    question_path, _ = read_questions(args.dataset)
    jobs = [
        ["eval_baseline_strongjudge.py", "--all-candidates-jsonl", str(cell / "candidates.jsonl"),
         "--question-file", str(question_path), "--output-jsonl", str(cell / "offline.jsonl"),
         "--output-summary", str(cell / "offline_summary.json")],
        ["posthoc_roberta_asr.py", "--input", str(cell / "candidates.jsonl"),
         "--output-jsonl", str(cell / "roberta.jsonl"),
         "--output-summary", str(cell / "roberta_summary.json"), "--device", "cpu"],
    ]
    for job in jobs:
        subprocess.run([sys.executable, str(LEGACY / job[0]), *job[1:]],
                       cwd=cell, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from evaluation_compat import apply_empty_response_policy
    empty_policy = apply_empty_response_policy(
        cell / "candidates.jsonl", cell / "offline.jsonl",
        cell / "offline_summary.json")
    offline = json.loads((cell / "offline_summary.json").read_text())
    roberta = json.loads((cell / "roberta_summary.json").read_text())
    n = count_rows(cell / "candidates.jsonl")
    if (offline.get("errors") or offline.get("unknown_count", 0) or
        offline.get("rows_evaluated") != n or offline.get("rows_total") != n or
        roberta.get("total_candidates") != n):
        raise ValueError("legacy_evaluation_incomplete")
    for line in (cell / "offline.jsonl").read_text().splitlines():
        if json.loads(line).get("error"):
            raise ValueError("legacy_judge_error_record")
    result = {"evaluator_semantics": "original_plus_documented_empty_response_policy",
              "recorded_responses": n,
              "empty_response_policy": empty_policy,
              "artifacts": {name: digest(cell / name) for name in names}}
    (cell / "evaluation.done.json").write_text(json.dumps(result, indent=2))
    emit(status="LEGACY_EVALUATION_COMPLETENESS_PASS", **status_fields(args),
         empty_responses_label0=empty_policy["applied_count"],
         response_label2_rate=offline.get("strong_judge_asr"),
         roberta_rate=roberta.get("original_evaluator_asr"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "audit", "run", "evaluate"])
    parser.add_argument("--method", choices=PROTOCOL["new_methods"] + PROTOCOL["existing_methods"])
    parser.add_argument("--ablation", choices=PROTOCOL["selection_ablations"])
    parser.add_argument("--feedback-ablation", choices=PROTOCOL["feedback_ablations"])
    parser.add_argument("--suite", choices=["new_baselines", "selection_ablations",
                                             "feedback_ablations",
                                             "cross_dataset_core",
                                             "cross_dataset_remaining",
                                             "cross_dataset_all",
                                             "cross_model_core",
                                             "cross_model_remaining",
                                             "cross_model_all"],
                        default="new_baselines")
    parser.add_argument("--dataset", choices=PROTOCOL["datasets"], default="gptfuzzer")
    parser.add_argument("--model", choices=PROTOCOL["models"], default="llama32_3b")
    parser.add_argument("--seed", type=int, choices=PROTOCOL["seeds"], default=100)
    parser.add_argument("--runtime", action="store_true", help="audit installed dependency boundaries without API/GPU")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    dependencies()
    if args.command == "plan":
        if args.suite.startswith("cross_model_"):
            cross = PROTOCOL["cross_model"]
            if args.suite == "cross_model_core":
                methods = cross["core_methods"]
            elif args.suite == "cross_model_remaining":
                methods = cross["remaining_methods"]
            else:
                methods = cross["core_methods"] + cross["remaining_methods"]
            for model in cross["new_models"]:
                for seed in cross["seeds"]:
                    for method in methods:
                        emit(method=method, dataset=cross["dataset"],
                             model=model, seed=seed,
                             target_budget_cap=cross["target_budget_cap"],
                             per_question_cap=cross["per_question_cap"],
                             action="NEW_NOT_STARTED")
        elif args.suite.startswith("cross_dataset_"):
            cross = PROTOCOL["cross_dataset"]
            if args.suite == "cross_dataset_core":
                methods = cross["core_methods"]
            elif args.suite == "cross_dataset_remaining":
                methods = cross["remaining_methods"]
            else:
                methods = cross["core_methods"] + cross["remaining_methods"]
            for dataset in cross["datasets"]:
                for seed in cross["seeds"]:
                    for method in methods:
                        emit(method=method, dataset=dataset,
                             model=cross["model"], seed=seed,
                             target_budget_cap=cross["target_budget_cap"],
                             per_question_cap=cross["per_question_cap"],
                             action="NEW_NOT_STARTED")
        elif args.suite == "feedback_ablations":
            for ablation, config in PROTOCOL["feedback_ablations"].items():
                for seed in PROTOCOL["seeds"]:
                    emit(method="ra_cofuzz", feedback_ablation=ablation,
                         dataset="gptfuzzer", model="llama32_3b", seed=seed,
                         target_budget=200, configuration=config,
                         action="NEW_NOT_STARTED")
        elif args.suite == "selection_ablations":
            for ablation, config in PROTOCOL["selection_ablations"].items():
                for seed in PROTOCOL["seeds"]:
                    emit(method="ra_cofuzz", selection_ablation=ablation,
                         dataset="gptfuzzer", model="llama32_3b", seed=seed,
                         target_budget=200, configuration=config, action="NEW_NOT_STARTED")
        else:
            for method in PROTOCOL["new_methods"]:
                for seed in PROTOCOL["seeds"]:
                    emit(method=method, dataset=args.dataset, model=args.model, seed=seed,
                         target_budget=200, per_question_cap=10, action="NEW_NOT_STARTED")
        return
    if args.command == "audit":
        emit(status="ORIGINAL_FILES_UNCHANGED", verified_files=verify_original())
        for dataset in PROTOCOL["datasets"]:
            _, rows = read_questions(dataset)
            emit(status="DATASET_CHECK", dataset=dataset, questions=len(rows))
            _, paired = read_questions(dataset, needs_targets=True)
            emit(status="REFERENCE_COLUMN_CHECK", dataset=dataset, references=len(paired))
        if args.runtime:
            with suppress_original_output():
                from easy_bridge import boundary_audit
                checked = boundary_audit()
                import easyjailbreak
                dep = Path(easyjailbreak.__file__).resolve().parent
                source = {str(p.relative_to(dep)): digest(p) for p in sorted(dep.rglob("*"))
                          if p.is_file() and p.suffix in {".py", ".json"}}
            (EXT / "runtime_boundary_pass.json").write_text(json.dumps(
                {"external_source": source, "bridge_sha256": digest(EXT / "easy_bridge.py"),
                 "bridge_helpers": {name: digest(EXT / name) for name in
                                    ("renellm_flow.py", "tap_compat.py")}}, indent=2))
            emit(status="BOUNDARY_CHECK_PASS_NOT_END_TO_END", cases=checked)
        return
    if args.method is None:
        parser.error("--method is required for run/evaluate")
    if args.ablation is not None and args.method != "ra_cofuzz":
        raise ValueError("invalid_selection_ablation_scope")
    if args.feedback_ablation is not None and args.method != "ra_cofuzz":
        raise ValueError("invalid_feedback_ablation_scope")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run_cell if args.command == "run" else evaluate_cell)(args)


if __name__ == "__main__":
    CONSOLE = os.dup(1)
    try:
        main()
    except Exception as exc:
        # Deliberately do not print exception messages, source lines or model texts.
        emit(status="FAILED", error_type=type(exc).__name__,
             safe_code=str(exc) if str(exc) in SAFE_CODES else None,
             missing_module=exc.name if isinstance(exc, ModuleNotFoundError) else None,
             safe_trace=[dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
                         for f in traceback.extract_tb(exc.__traceback__)[-8:]])
        raise SystemExit(1)
    finally:
        os.close(CONSOLE)
