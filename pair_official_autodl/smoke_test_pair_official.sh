#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${RACOFUZZ_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GPTFUZZ_ROOT="$BASE/GPTFuzz-master"
PAIR_ROOT="$BASE/JailbreakingLLMs-official"
ADAPTER_ROOT="$BASE/pair_official_autodl"
OUT_DIR="$GPTFUZZ_ROOT/pair_official_results"

ENV_FILE="${RACOFUZZ_ENV_FILE:-$BASE/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export PYTHONPATH="$GPTFUZZ_ROOT:$PAIR_ROOT:${PYTHONPATH:-}"
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

QUESTION_PATH="$GPTFUZZ_ROOT/datasets/questions/gptfuzzer_q20_seed1234_with_targets.csv"
if [ ! -s "$QUESTION_PATH" ]; then
  echo "Missing question/target CSV: $QUESTION_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python "$ADAPTER_ROOT/pair_official_autodl.py" \
  --official-pair-root "$PAIR_ROOT" \
  --gptfuzz-root "$GPTFUZZ_ROOT" \
  --question-path "$QUESTION_PATH" \
  --require-targets \
  --target-model "${TARGET_LLAMA:?Set TARGET_LLAMA to the local target model path}" \
  --output-jsonl "$OUT_DIR/smoke.jsonl" \
  --summary-json "$OUT_DIR/smoke_summary.json" \
  --run-id 0 \
  --random-seed 1234 \
  --question-limit 1 \
  --n-streams 1 \
  --n-iterations 1

echo "Smoke test completed: $OUT_DIR/smoke_summary.json"
