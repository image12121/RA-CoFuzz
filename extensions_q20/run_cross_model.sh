#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

phase="${1:-all}"
if [[ "$phase" != "core" && "$phase" != "remaining" && "$phase" != "all" ]]; then
    echo "usage: $0 [core|remaining|all]" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python extensions_q20/run.py audit

models=(qwen25_1_5b qwen25_3b qwen25_7b vicuna_7b)
seeds=(100 200 300)
core_methods=(ra_cofuzz strict_gptfuzzer pair)
remaining_methods=(tap renellm deepinception)

run_group() {
    local group="$1"
    shift
    local methods=("$@")
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            for method in "${methods[@]}"; do
                echo "CELL_START phase=${group} model=${model} method=${method} seed=${seed}"
                python extensions_q20/run.py run \
                    --method "$method" --dataset gptfuzzer \
                    --model "$model" --seed "$seed"
                echo "CELL_EVALUATION_START phase=${group} model=${model} method=${method} seed=${seed}"
                python extensions_q20/run.py evaluate \
                    --method "$method" --dataset gptfuzzer \
                    --model "$model" --seed "$seed"
                echo "CELL_DONE phase=${group} model=${model} method=${method} seed=${seed}"
            done
        done
    done
}

if [[ "$phase" == "core" || "$phase" == "all" ]]; then
    run_group core "${core_methods[@]}"
    python extensions_q20/audit_cross_model.py --phase core
    echo "Q20_CROSS_MODEL_CORE_36_CELLS_FINISHED"
fi

if [[ "$phase" == "remaining" ]]; then
    python extensions_q20/audit_cross_model.py --phase core
fi

if [[ "$phase" == "remaining" || "$phase" == "all" ]]; then
    run_group remaining "${remaining_methods[@]}"
    python extensions_q20/audit_cross_model.py --phase all
    echo "Q20_CROSS_MODEL_ALL_72_CELLS_FINISHED"
fi
