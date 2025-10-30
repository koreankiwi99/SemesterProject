# Running Tests

## Setup Complete

✓ Codebase refactored and organized
✓ Prompts fixed (CoT and Lean)
✓ API key saved in .env
✓ All test scripts updated with gpt-5 and reasoning_effort support

## Next Steps

### 1. Activate Conda Environment
```bash
conda activate llm-lean
cd /Users/kyuheekim/SemesterProject
```

### 2. Run Tests (in order)

All tests use the same settings except prompts:
- Model: gpt-5
- reasoning_effort: medium
- Multi-LogiEval: 10 samples per logic×depth combination
- FOLIO: all questions

#### Test 1: Multi-LogiEval CoT
```bash
PYTHONPATH=src python src/experiments/test_multi.py \
    --api_key $(grep OPENAI_API_KEY .env | cut -d '=' -f2) \
    --data_dir data/multi_logi_original/data \
    --system_prompt prompts/multilogi/zero_shot_cot_system.txt \
    --user_prompt prompts/multilogi/zero_shot_cot_user.txt \
    --prompt_name zero_shot_cot \
    --samples_per_combination 10 \
    --model gpt-5 \
    --reasoning_effort medium
```

#### Test 2: Multi-LogiEval Lean
```bash
PYTHONPATH=src python src/experiments/test_multi_interact.py \
    --api_key $(grep OPENAI_API_KEY .env | cut -d '=' -f2) \
    --data_dir data/multi_logi_original/data \
    --system_prompt prompts/multilogi/lean_system.txt \
    --user_prompt prompts/multilogi/lean_user.txt \
    --prompt_name lean_test \
    --samples_per_combination 10 \
    --max_iterations 3 \
    --model gpt-5 \
    --reasoning_effort medium
```

#### Test 3: FOLIO CoT
```bash
PYTHONPATH=src python src/experiments/test_folio.py \
    --api_key $(grep OPENAI_API_KEY .env | cut -d '=' -f2) \
    --folio_file data/folio_original/folio-train.jsonl \
    --system_prompt prompts/folio/zero_shot_cot_system.txt \
    --user_prompt prompts/folio/zero_shot_cot_user.txt \
    --prompt_name zero_shot_cot \
    --model gpt-5 \
    --reasoning_effort medium
```

#### Test 4: FOLIO Lean
```bash
PYTHONPATH=src python src/experiments/test_folio_interact.py \
    --api_key $(grep OPENAI_API_KEY .env | cut -d '=' -f2) \
    --folio_file data/folio_original/folio-train.jsonl \
    --system_prompt prompts/folio/lean_system.txt \
    --user_prompt prompts/folio/lean_user.txt \
    --prompt_name lean_test \
    --max_iterations 3 \
    --model gpt-5 \
    --reasoning_effort medium
```

## Expected Results

Results will be saved to:
- `results/multilogieval_zero_shot_cot_<timestamp>/`
- `results/multilogieval_lean_test_<timestamp>/`
- `results/folio_zero_shot_cot_<timestamp>/`
- `results/folio_lean_test_<timestamp>/`

Each directory contains:
- `all_results.json`: Complete results
- `responses/`: Individual question responses
- `summary.txt`: Overall accuracy
- `accuracy_table.txt`: Paper-format table (Multi-LogiEval only)
- `progress.txt`: Real-time progress log
