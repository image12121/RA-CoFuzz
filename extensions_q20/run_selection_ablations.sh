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

variants=(hybrid_wo_elite hybrid_wo_dp hybrid_wo_mcts)
seeds=(100 200 300)

for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
        echo "CELL_START selection_ablation=${variant} seed=${seed}"
        python extensions_q20/run.py run \
            --method ra_cofuzz \
            --ablation "$variant" \
            --dataset gptfuzzer \
            --model llama32_3b \
            --seed "$seed"
        echo "CELL_EVALUATION_START selection_ablation=${variant} seed=${seed}"
        python extensions_q20/run.py evaluate \
            --method ra_cofuzz \
            --ablation "$variant" \
            --dataset gptfuzzer \
            --model llama32_3b \
            --seed "$seed"
        echo "CELL_DONE selection_ablation=${variant} seed=${seed}"
    done
done

python extensions_q20/audit_selection_ablations.py
echo "Q20_SELECTION_ABLATIONS_WORKFLOW_FINISHED"
