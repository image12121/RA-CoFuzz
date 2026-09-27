import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # for debugging

from fastchat.model import add_model_args
import argparse
import pandas as pd
from gptfuzzer.fuzzer.selection import (
    MCTSExploreSelectPolicy,
    RAESGuidedSelectPolicy,
    StrongJudgeGuidedSelectPolicy,
    MCTSRAESSelectPolicy,
    HybridRAESSelectPolicy,
)
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.llm import (
    OpenAILLM,
    OpenAICompatibleLLM,
    LocalVLLM,
    LocalLLM,
    PaLM2LLM,
    ClaudeLLM,
)
from gptfuzzer.utils.predict import RoBERTaPredictor
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)


def env_flag(name):
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def main(args):
    initial_seed = pd.read_csv(args.seed_path)['text'].tolist()
    questions = pd.read_csv(args.question_path)['text'].tolist()
    if args.question_limit > 0:
        questions = questions[:args.question_limit]
    if not questions:
        raise ValueError("No questions were loaded from question_path.")

    raes_mode = args.raes_mode or env_flag("GPTFUZZ_RAES_MODE")
    if args.raes_mode:
        os.environ["GPTFUZZ_RAES_MODE"] = "1"
    ra_mode = args.ra_mode or env_flag("GPTFUZZ_RA_MODE") or raes_mode
    strong_judge_in_loop = args.strong_judge_in_loop or env_flag("GPTFUZZ_STRONG_JUDGE_IN_LOOP")
    ra_force_review = args.ra_force_review or env_flag("GPTFUZZ_RA_FORCE_REVIEW")
    selection_policy_name = (
        os.getenv("GPTFUZZ_SELECTION_POLICY")
        or args.selection_policy
        or ("raes" if raes_mode else "strong_judge" if ra_mode else "mcts")
    )

    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY", "")
    if args.mutation_backend == "openai":
        if not openai_key:
            raise ValueError(
                "OpenAI mutation requires --openai_key or OPENAI_API_KEY."
            )
        mutation_model = OpenAILLM(args.model_path, openai_key)
        mutation_model_name = args.model_path
    else:
        mutation_api_key = os.getenv("MUTATION_API_KEY", "")
        mutation_api_base_url = os.getenv("MUTATION_API_BASE_URL", "")
        mutation_api_model_name = os.getenv("MUTATION_API_MODEL_NAME", "")
        if not mutation_api_key or not mutation_api_base_url or not mutation_api_model_name:
            raise ValueError(
                "OpenAI-compatible mutation requires MUTATION_API_KEY, "
                "MUTATION_API_BASE_URL, and MUTATION_API_MODEL_NAME."
            )
        mutation_model = OpenAICompatibleLLM(
            mutation_api_model_name,
            mutation_api_key,
            mutation_api_base_url,
        )
        mutation_model_name = mutation_api_model_name

    if args.target_backend == "local_hf":
        target_model = LocalLLM(
            args.target_model,
            max_gpu_memory=args.target_max_gpu_memory,
            cpu_offloading=args.target_cpu_offloading,
            local_files_only=args.target_local_files_only,
            default_max_new_tokens=args.target_max_new_tokens,
            default_batch_size=args.target_batch_size,
        )
    elif args.target_backend == "local_vllm":
        target_model = LocalVLLM(args.target_model)
    else:
        target_model = OpenAILLM(args.target_model, openai_key)

    roberta_model = RoBERTaPredictor('hubert233/GPTFuzz', device=args.predictor_device)

    if selection_policy_name == "raes":
        select_policy = RAESGuidedSelectPolicy()
    elif selection_policy_name == "strong_judge":
        select_policy = StrongJudgeGuidedSelectPolicy(
            exploration_ratio=args.strong_judge_exploration_ratio
        )
    elif selection_policy_name == "mcts_raes":
        select_policy = MCTSRAESSelectPolicy()
    elif selection_policy_name == "hybrid_raes":
        select_policy = HybridRAESSelectPolicy()
    else:
        select_policy = MCTSExploreSelectPolicy()

    fuzzer = GPTFuzzer(
        questions=questions,
        # target_model=openai_model,
        target=target_model,
        predictor=roberta_model,
        initial_seed=initial_seed,
        mutate_policy=MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(mutation_model, temperature=0.0),  # for reproduction only, if you want better performance, use temperature>0
            OpenAIMutatorExpand(mutation_model, temperature=0.0),
            OpenAIMutatorGenerateSimilar(mutation_model, temperature=0.0),
            OpenAIMutatorRephrase(mutation_model, temperature=0.0),
            OpenAIMutatorShorten(mutation_model, temperature=0.0)],
            concatentate=True,
        ),
        select_policy=select_policy,
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        generate_in_batch=False,
        ra_mode=ra_mode,
        raes_mode=raes_mode,
        strong_judge_in_loop=strong_judge_in_loop,
        ra_force_review=ra_force_review,
        ra_result_file=args.ra_result_file,
        ra_summary_file=args.ra_summary_file,
        mutation_backend=args.mutation_backend,
        mutation_model=mutation_model_name,
        target_model=args.target_model,
        target_backend=args.target_backend,
        log_all_candidates=args.log_all_candidates,
        all_candidates_file=args.all_candidates_file,
    )

    fuzzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai_key', type=str, default='',
                        help='OpenAI API Key. Falls back to OPENAI_API_KEY.')
    parser.add_argument('--claude_key', type=str, default='', help='Claude API Key')
    parser.add_argument('--palm_key', type=str, default='', help='PaLM2 api key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo',
                        help='mutate model path')
    parser.add_argument('--mutation_backend', type=str, default='openai',
                        choices=['openai', 'openai_compatible'],
                        help='Mutation API backend. openai_compatible uses MUTATION_* environment variables.')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--target_backend', type=str, default='local_hf',
                        choices=['local_hf', 'local_vllm', 'openai'],
                        help='Target model backend. local_hf is the Windows-friendly default.')
    parser.add_argument('--predictor_device', type=str, default='cuda:0',
                        help='Device for RoBERTaPredictor, for example cuda:0 or cpu.')
    parser.add_argument('--target_max_new_tokens', type=int, default=32,
                        help='Default maximum new tokens for LocalLLM target generation.')
    parser.add_argument('--target_batch_size', type=int, default=1,
                        help='Default batch size for LocalLLM target batch generation.')
    parser.add_argument('--target_max_gpu_memory', type=str, default='6GiB',
                        help='Maximum GPU memory passed to LocalLLM.')
    parser.add_argument('--target_local_files_only', action='store_true',
                        help='Force LocalLLM to use local model files only.')
    parser.add_argument('--target_cpu_offloading', action='store_true',
                        help='Enable LocalLLM CPU offloading. FastChat requires 8-bit quantization for this to take effect.')
    parser.add_argument('--max_query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument('--selection_policy', type=str, default=None,
                        choices=['mcts', 'strong_judge', 'raes', 'mcts_raes', 'hybrid_raes'],
                        help='Selection policy. Can also be set with GPTFUZZ_SELECTION_POLICY.')
    parser.add_argument('--ra-mode', action='store_true',
                        help='Enable RA-CoFuzz mode. Can also be set with GPTFUZZ_RA_MODE=1.')
    parser.add_argument('--raes-mode', action='store_true',
                        help='Enable RAES-MVP scoring and selection. Can also be set with GPTFUZZ_RAES_MODE=1.')
    parser.add_argument('--strong-judge-in-loop', action='store_true',
                        help='Call the strong judge during fuzzing. Can also be set with GPTFUZZ_STRONG_JUDGE_IN_LOOP=1.')
    parser.add_argument('--ra-force-review', action='store_true',
                        help='Smoke-test only: force all RA candidates through StrongJudge. Can also be set with GPTFUZZ_RA_FORCE_REVIEW=1.')
    parser.add_argument('--strong-judge-exploration-ratio', type=float, default=0.25,
                        help='Random exploration ratio for StrongJudgeGuidedSelectPolicy.')
    parser.add_argument('--ra-result-file', type=str, default='results_ra.jsonl',
                        help='RA-CoFuzz JSONL result path.')
    parser.add_argument('--ra-summary-file', type=str, default='summary_ra.json',
                        help='RA-CoFuzz summary JSON path.')
    parser.add_argument('--log-all-candidates', action='store_true',
                        help='Log every non-RA baseline candidate to a JSONL file.')
    parser.add_argument('--all-candidates-file', type=str, default='baseline_candidates.jsonl',
                        help='JSONL path for --log-all-candidates.')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="datasets/prompts/GPTFuzzer.csv")
    parser.add_argument("--question_path", type=str,
                        default="datasets/questions/question_list.csv")
    parser.add_argument("--question_limit", type=int, default=1,
                        help="Read only the first N questions. Use a value <= 0 for all questions.")
    add_model_args(parser)

    args = parser.parse_args()
    main(args)


