#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${RACOFUZZ_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GPTFUZZ_ROOT="$BASE/GPTFuzz-master"
PAIR_ROOT="$BASE/JailbreakingLLMs-official"
ADAPTER_ROOT="$BASE/pair_official_autodl"
QUESTION_PATH="$GPTFUZZ_ROOT/datasets/questions/gptfuzzer_q20_seed1234_with_targets.csv"
TARGET_MODEL="${TARGET_LLAMA:?Set TARGET_LLAMA to the local Llama-3.2-3B-Instruct path}"
OUT_DIR="$GPTFUZZ_ROOT/pair_official_results"

ENV_FILE="${RACOFUZZ_ENV_FILE:-$BASE/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export PYTHONPATH="$GPTFUZZ_ROOT:$PAIR_ROOT:${PYTHONPATH:-}"

# Reuse the same DeepSeek endpoint used by RA-CoFuzz. Values are never printed.
export PAIR_ATTACK_API_KEY="${MUTATION_API_KEY:-${RACOFUZZ_API_KEY:-}}"
export PAIR_ATTACK_BASE_URL="${MUTATION_API_BASE_URL:-${RACOFUZZ_API_BASE_URL:-https://api.deepseek.com/v1}}"
export PAIR_ATTACK_MODEL="${MUTATION_API_MODEL_NAME:-deepseek-chat}"
export PAIR_JUDGE_API_KEY="${RACOFUZZ_API_KEY:-${MUTATION_API_KEY:-}}"
export PAIR_JUDGE_BASE_URL="${RACOFUZZ_API_BASE_URL:-${MUTATION_API_BASE_URL:-https://api.deepseek.com/v1}}"
export PAIR_JUDGE_MODEL="${RACOFUZZ_API_MODEL_NAME:-deepseek-chat}"

if [ -z "$PAIR_ATTACK_API_KEY" ] || [ -z "$PAIR_JUDGE_API_KEY" ]; then
  echo "PAIR attack/judge API credentials are missing." >&2
  exit 1
fi

if [ ! -s "$QUESTION_PATH" ]; then
  echo "Missing question/target CSV: $QUESTION_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$GPTFUZZ_ROOT"

RUN=0
for SEED in 100 200 300; do
  RUN=$((RUN + 1))
  RAW="$OUT_DIR/results_pair_official_llama32_q20_seed${SEED}.jsonl"
  RUN_SUMMARY="$OUT_DIR/summary_pair_official_llama32_q20_seed${SEED}.json"
  SJ_JSONL="$OUT_DIR/eval_pair_official_llama32_q20_seed${SEED}_strongjudge.jsonl"
  SJ_SUMMARY="$OUT_DIR/summary_pair_official_llama32_q20_seed${SEED}_strongjudge.json"
  ROB_JSONL="$OUT_DIR/results_pair_official_llama32_q20_seed${SEED}_roberta.jsonl"
  ROB_SUMMARY="$OUT_DIR/summary_pair_official_llama32_q20_seed${SEED}_roberta.json"

  python "$ADAPTER_ROOT/pair_official_autodl.py" \
    --official-pair-root "$PAIR_ROOT" \
    --gptfuzz-root "$GPTFUZZ_ROOT" \
    --question-path "$QUESTION_PATH" \
    --require-targets \
    --target-model "$TARGET_MODEL" \
    --output-jsonl "$RAW" \
    --summary-json "$RUN_SUMMARY" \
    --run-id "$RUN" \
    --random-seed "$SEED" \
    --n-streams 2 \
    --n-iterations 5 \
    --keep-last-n 4 \
    --target-max-new-tokens 128 \
    --target-max-gpu-memory 22GiB

  python eval_baseline_strongjudge.py \
    --all-candidates-jsonl "$RAW" \
    --question-file "$QUESTION_PATH" \
    --output-jsonl "$SJ_JSONL" \
    --output-summary "$SJ_SUMMARY"

  python posthoc_roberta_asr.py \
    --input "$RAW" \
    --output-summary "$ROB_SUMMARY" \
    --output-jsonl "$ROB_JSONL" \
    --device cpu
done

python "$ADAPTER_ROOT/summarize_pair_official.py" --results-dir "$OUT_DIR"
