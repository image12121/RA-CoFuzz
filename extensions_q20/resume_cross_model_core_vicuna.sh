#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python extensions_q20/run.py audit

echo "CELL_EVALUATION_RETRY_START phase=core model=vicuna_7b method=ra_cofuzz seed=100"
python extensions_q20/run.py evaluate \
    --method ra_cofuzz --dataset gptfuzzer --model vicuna_7b --seed 100
echo "CELL_DONE phase=core model=vicuna_7b method=ra_cofuzz seed=100"

run_cell() {
    local method="$1"
    local seed="$2"
    echo "CELL_START phase=core model=vicuna_7b method=${method} seed=${seed}"
    python extensions_q20/run.py run \
        --method "$method" --dataset gptfuzzer --model vicuna_7b --seed "$seed"
    echo "CELL_EVALUATION_START phase=core model=vicuna_7b method=${method} seed=${seed}"
    python extensions_q20/run.py evaluate \
        --method "$method" --dataset gptfuzzer --model vicuna_7b --seed "$seed"
    echo "CELL_DONE phase=core model=vicuna_7b method=${method} seed=${seed}"
}

run_cell strict_gptfuzzer 100
run_cell pair 100
for seed in 200 300; do
    run_cell ra_cofuzz "$seed"
    run_cell strict_gptfuzzer "$seed"
    run_cell pair "$seed"
done

python extensions_q20/audit_cross_model.py --phase core
echo "Q20_CROSS_MODEL_CORE_36_CELLS_FINISHED"
