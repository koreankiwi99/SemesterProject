#!/bin/bash
cd "$(dirname "$0")/.."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate llm-lean
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_multi_bidirectional.py \
    --api_key "$OPENAI_API_KEY" \
    --data_dir data/multi_logi_original/data \
    --model gpt-5 \
    --num_questions 10 \
    --concurrency 5 \
    --max_iterations 3 \
    --seed 42 \
    --prompt_name bidirectional
