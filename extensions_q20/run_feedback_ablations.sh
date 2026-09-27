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

variants=(wo_label1_reward wo_online_strongjudge_feedback full_online_review)
seeds=(100 200 300)

for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
        echo "CELL_START feedback_ablation=${variant} seed=${seed}"
        python extensions_q20/run.py run \
            --method ra_cofuzz \
            --feedback-ablation "$variant" \
            --dataset gptfuzzer \
            --model llama32_3b \
            --seed "$seed"
        echo "CELL_EVALUATION_START feedback_ablation=${variant} seed=${seed}"
        python extensions_q20/run.py evaluate \
            --method ra_cofuzz \
            --feedback-ablation "$variant" \
            --dataset gptfuzzer \
            --model llama32_3b \
            --seed "$seed"
        echo "CELL_DONE feedback_ablation=${variant} seed=${seed}"
    done
done

python extensions_q20/audit_feedback_ablations.py
echo "Q20_FEEDBACK_ABLATIONS_WORKFLOW_FINISHED"
