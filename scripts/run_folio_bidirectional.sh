#!/bin/bash
cd "$(dirname "$0")/.."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate llm-lean
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_bidirectional.py \
    --api_key "$OPENAI_API_KEY" \
    --folio_file data/folio_original/folio-validation.json \
    --model gpt-5 \
    --num_questions 0 \
    --concurrency 5 \
    --max_iterations 3 \
    --prompt_name bidirectional
