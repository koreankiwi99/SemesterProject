# Session Status - 2025-10-30

## What Was Completed

### 1. Enhanced FOLIO Analysis
- Fixed FOLIO data structure parsing (nested `results` array)
- Successfully analyzed 200 FOLIO questions comparing CoT vs Lean
- Updated notebook: `nb/analyze_lean_vs_cot.ipynb`

### 2. Analysis Results

#### FOLIO Dataset (200 questions)
- **CoT Accuracy**: 86.0%
- **Lean Accuracy**: 86.0%
- **Both correct**: 166 questions
- **Both wrong**: 22 questions
- **CoT succeeds, Lean fails**: 6 questions
- **Lean succeeds, CoT fails**: 6 questions
- **Finding**: Perfectly balanced - FOLIO shows equal performance

#### Multi-LogiEval Dataset (66 questions)
- **CoT Accuracy**: 75.3%
- **Lean Accuracy**: 83.3%
- **Both correct**: 48 questions
- **Both wrong**: 9 questions
- **CoT succeeds, Lean fails**: 2 questions
- **Lean succeeds, CoT fails**: 7 questions
- **Finding**: Lean outperforms CoT 3.5x in unique successes

### 3. Exported Analysis Files
All in `results/analysis/`:
- `folio_cot_succeeds_lean_fails.json` (6 questions)
- `folio_lean_succeeds_cot_fails.json` (6 questions)
- `mlogi_cot_succeeds_lean_fails.json` (2 questions)
- `mlogi_lean_succeeds_cot_fails.json` (7 questions)
- `summary.json` (complete statistics)

## Currently Running Experiments

Check status with:
```bash
ps aux | grep "test_folio\|test_multi"
```

FOLIO experiments are running in background (started earlier).

## Ready to Run: Depth-5 Experiments

### Script Created: `run_d5_experiments.sh`

**Depth-5 Question Count:**
- FOL: 45 questions
- NM: 20 questions
- PL: 45 questions
- **Total: 110 questions**

**Three experiments configured:**
1. Zero-Shot CoT (d5 only)
2. Lean with 3 iterations (d5 only)
3. Two-Stage Lean (d5 only)

**To start:**
```bash
cd /Users/kyuheekim/SemesterProject
bash run_d5_experiments.sh
```

**Monitor logs:**
```bash
tail -f d5_cot.log
tail -f d5_lean.log
tail -f d5_two_stage.log
```

## Key Files Modified

1. `nb/analyze_lean_vs_cot.ipynb` - Enhanced with FOLIO flattening logic
2. `run_d5_experiments.sh` - New script for depth-5 experiments
3. `results/analysis/` - All comparison results exported

## Next Steps

1. Start depth-5 experiments: `bash run_d5_experiments.sh`
2. Monitor running FOLIO experiments
3. When complete, run analysis notebook on new results
4. Compare depth-5 performance (hardest questions) vs full dataset

## Notes

- FOLIO shows balanced performance (86% both), unlike Multi-LogiEval where Lean wins
- Depth-5 focuses on most complex reasoning chains (5-step proofs)
- All scripts use API key ending in ...NrsA
- Experiments run in parallel to save time
