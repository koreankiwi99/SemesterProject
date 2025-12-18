# Formal Verification for Multi-Step Logical Reasoning with Lean 4

**EPFL NLP Lab Semester Project**

## Overview

This project investigates how formal verification with Lean 4 can improve LLM performance on multi-step logical reasoning tasks. We address a critical limitation: LLM accuracy degrades significantly as reasoning depth increases (single-step: 90%+, 5-step: often <50%).

**Key Innovation**: We introduce **Bidirectional Verification**, which attempts to prove both a statement AND its negation, detecting formalization errors when both succeed.

## Key Results

### Main Finding: Formal Verification Excels on Hard Problems

| Dataset | CoT | Lean | Bidirectional | Best Improvement |
|---------|-----|------|---------------|------------------|
| FOLIO (n=203) | 85.71% | 74.88% | **87.68%** | +1.97% over CoT |
| Multi-LogiEval d1-d5 (n=150) | 76.67% | 78.67% | **82.67%** | +6.00% over CoT |
| Multi-LogiEval d5 only (n=110) | 75.45% | 84.55% | **87.27%** | **+11.82%** over CoT |

**Critical Insight**: As reasoning depth increases, formal verification becomes increasingly valuable:
- Simple tasks (FOLIO): CoT outperforms unidirectional Lean by 10.8pp
- Hard tasks (depth-5): Lean outperforms CoT by **9.1pp**
- Bidirectional verification achieves best results across all benchmarks

### Breakdown by Logic Type (Depth-5)

| Logic Type | CoT | Lean | Bidirectional |
|------------|-----|------|---------------|
| Propositional (PL) | 82.22% | 93.33% | **97.78%** |
| First-Order (FOL) | 76.09% | 86.67% | **88.89%** |
| Non-Monotonic (NM) | 45.00% | 60.00% | 60.00% |

Propositional logic achieves exceptional **97.78%** accuracy with bidirectional verification.

### Performance by Ground Truth Label

Understanding how different methods perform on True vs False vs Unknown answers reveals important biases.

#### FOLIO (n=203: 72 True, 62 False, 69 Unknown)

| Method | True | False | Unknown | Overall |
|--------|------|-------|---------|---------|
| CoT | 60/72 (83.33%) | 52/62 (83.87%) | 62/69 (89.86%) | 85.71% |
| Lean | 68/72 (94.44%) | 35/62 (56.45%) | 49/69 (71.01%) | 74.88% |
| Two-Stage | 60/72 (83.33%) | 48/62 (77.42%) | 53/69 (76.81%) | 79.31% |
| Bidirectional | 64/72 (88.89%) | 54/62 (87.10%) | 60/69 (86.96%) | **87.68%** |

**Key Observations:**
- Lean has strong **True-bias**: 94.44% on True but only 56.45% on False
- This imbalance explains why unidirectional Lean underperforms CoT overall
- Bidirectional achieves balanced performance across all label types
- CoT is best on Unknown cases (open-world reasoning)

#### Multi-LogiEval d1-d5 (n=150: 103 Yes, 47 No)

| Method | Yes (True) | No (False) | Overall |
|--------|------------|------------|---------|
| CoT | 72/103 (69.90%) | 43/47 (91.49%) | 76.67% |
| Lean | 82/103 (79.61%) | 36/47 (76.60%) | 78.67% |
| Bidirectional | 85/103 (82.52%) | 39/47 (82.98%) | **82.67%** |

**Key Observations:**
- CoT shows strong **No-bias** (91.49% on No vs 69.90% on Yes)
- Lean corrects Yes-detection (+9.71pp) but hurts No-detection (-14.89pp)
- Bidirectional achieves balanced performance across both labels

#### Multi-LogiEval Depth-5 Only (n=110: 100 Yes, 10 No)

| Method | Yes (True) | No (False) | Overall |
|--------|------------|------------|---------|
| CoT | 75/100 (75.00%) | 8/10 (80.00%) | 75.45% |
| Lean | 86/100 (86.00%) | 7/10 (70.00%) | 84.55% |
| Bidirectional | 89/100 (89.00%) | 7/10 (70.00%) | **87.27%** |

**Key Observations:**
- Dataset is heavily Yes-biased (91% Yes, 9% No)
- Lean improves Yes-detection (+11pp over CoT)
- All methods struggle with minority No class at depth-5
- Bidirectional achieves best Yes performance (89.00%)

---

## Approach Comparison

### 1. Chain-of-Thought (CoT)
Zero-shot prompting asking the model to reason step-by-step in natural language.
- **Strengths**: Simple, fast, works well on easy problems
- **Weaknesses**: Degrades significantly at higher reasoning depths

### 2. Direct Lean Verification
Translate natural language to Lean 4 formal proofs with iterative refinement (max 3 iterations).
- **Strengths**: Structured reasoning, excellent on hard problems
- **Weaknesses**: Models can "cheat" by axiomatizing conclusions

### 3. Two-Stage (NL → Lean)
First perform natural language reasoning, then translate to Lean.
- **Strengths**: Combines NL intuition with formal verification
- **Weaknesses**: Translation step introduces additional errors

### 4. Bidirectional Verification (Novel)
Attempt to prove BOTH the statement AND its negation in parallel:
- If only TRUE succeeds → Answer is True
- If only FALSE succeeds → Answer is False
- If BOTH succeed → Formalization error detected, fall back to CoT
- If NEITHER succeeds → Fall back to CoT

**Key Innovation**: Detects formalization errors that unidirectional approaches miss.

---

## Error Analysis

### The Axiomatization Problem

Error analysis revealed that **60-78%** of verification failures on simple tasks stem from models "cheating" by axiomatizing conclusions instead of proving them:

| Dataset | Axiomatization Rate | Top Error Type |
|---------|---------------------|----------------|
| FOLIO | 77.6% | Axiomatizes Conclusion (59.2%) |
| Multi-LogiEval d5 | **0%** | Reasoning Failure (63.6%) |

**Key Finding**: Error patterns shift dramatically with task complexity:
- Simple tasks: Models exploit Lean syntax to "cheat"
- Hard tasks: Models cannot cheat—errors are genuine reasoning failures

This validates that strong depth-5 performance reflects real reasoning capability.

### Error Categories
1. **AXIOMATIZES_CONCLUSION**: Directly asserts what should be proven
2. **AXIOMATIZES_CONTRADICTION**: Asserts statements contradicting premises
3. **AXIOMATIZES_UNMENTIONED**: Invents facts about entities not in premises
4. **INCORRECT_FORMALIZATION**: Mistranslates premises to Lean
5. **REASONING_FAILURE**: Correct axioms but can't derive conclusion
6. **OTHER**: Miscellaneous errors

---

## Bidirectional Verification Details

### Agreement Patterns

**FOLIO (n=203):**
| Pattern | Count | Accuracy |
|---------|-------|----------|
| TRUE_ONLY | 69 | 89.86% |
| FALSE_ONLY | 56 | 92.86% |
| BOTH_SUCCESS | 8 | 62.50% (falls back to CoT) |
| NEITHER_SUCCESS | 70 | 84.29% (falls back to CoT) |

**Multi-LogiEval d5 (n=110):**
| Pattern | Count |
|---------|-------|
| TRUE_ONLY | 92 |
| NEITHER_SUCCESS | 18 |
| BOTH_SUCCESS | 0 |
| FALSE_ONLY | 0 |

At depth-5, bidirectional verification achieves high TRUE_ONLY rates with no formalization errors detected.

### CoT vs Bidirectional Gap Analysis by Depth

#### Accuracy Gap (Bidir - CoT) by Depth

| Depth | CoT | Bidir | Gap | Gap Change |
|-------|-----|-------|-----|------------|
| d1 | 76.7% | 73.3% | -3.3% | - |
| d2 | 86.7% | 96.7% | +10.0% | +13.3% |
| d3 | 73.3% | 70.0% | -3.3% | -13.3% |
| d4 | 90.0% | 90.0% | +0.0% | +3.3% |
| d5 | 56.7% | 83.3% | **+26.7%** | **+26.7%** |

**Key Finding**: Gap fluctuates at d1-d4, then **explodes at d5** with a +26.7% jump.

#### Accuracy by Logic Type (All Depths Combined)

| Logic | CoT | Bidir | Gap |
|-------|-----|-------|-----|
| FOL | 82.0% | 86.0% | +4.0% |
| NM | 64.0% | 70.0% | +6.0% |
| PL | 84.0% | 92.0% | +8.0% |

#### Gap by Logic Type × Depth

| Logic | d1 | d2 | d3 | d4 | d5 |
|-------|----|----|----|----|-----|
| PL | 0% | +30% | -10% | -10% | **+30%** |
| FOL | +10% | 0% | 0% | -10% | **+20%** |
| NM | -20% | 0% | 0% | +20% | **+30%** |

All three logic types show **largest Bidirectional advantage at d5**.

#### Disagreement Analysis: When They Disagree, Who Wins?

| Depth | Disagreements | Bidir Wins | CoT Wins | Bidir Win% |
|-------|---------------|------------|----------|------------|
| d1 | 3 | 1 | 2 | 33.3% |
| d2 | 3 | 3 | 0 | **100%** |
| d3 | 11 | 5 | 6 | 45.5% |
| d4 | 4 | 2 | 2 | 50.0% |
| d5 | 8 | 8 | 0 | **100%** |

**Key Finding**: At d2 and d5, when CoT and Bidirectional disagree, **Bidirectional is always correct**. Analysis of d5 disagreements reveals CoT has a false-negative bias (says "No" when answer is "Yes"), which Bidirectional corrects by actually proving the statement.

#### D5-Only Disagreement Analysis by Logic Type (n=110)

| Logic Type | Disagreements | Bidir Wins | CoT Wins | Bidir Win% |
|------------|---------------|------------|----------|------------|
| FOL | 8 | 7 | 1 | 87.5% |
| NM | 3 | 3 | 0 | **100%** |
| PL | 7 | 7 | 0 | **100%** |
| **TOTAL** | **18** | **17** | **1** | **94.4%** |

**Key Finding**: On the depth-5 only subset (110 questions), when CoT and Bidirectional disagree, **Bidirectional wins 94.4% of the time** (17/18). The pattern shows CoT consistently says "No" when the answer is "Yes", while Bidirectional correctly proves the statement. This demonstrates formal verification's strength at high reasoning depths.

---

## Robustness Evaluation

### Tautological Noise (FOLIO)

Adding logically irrelevant but syntactically valid statements:

| Method | Original | k=1 noise | k=2 noise | k=4 noise |
|--------|----------|-----------|-----------|-----------|
| CoT | 85.71% | 84.00% | 85.50% | 85.00% |
| Lean | 74.88% | 69.95% | 70.44% | 70.94% |

- CoT is robust to tautological noise (~0.71% degradation)
- Lean verification degrades ~4-5% with noise (increased formalization complexity)

### Multi-LogiEval Noise Perturbations (GPT-5, December 2024)

Testing robustness to different noise types on Multi-LogiEval depth-5 subset (n=110):

#### Tautological Noise
Injecting logically valid but irrelevant statements (e.g., "All cats are cats"):

| Noise Level | Accuracy | vs Baseline |
|-------------|----------|-------------|
| Baseline (clean) | 83/110 (75.45%) | - |
| k=1 | 86/110 (78.18%) | +2.73% |
| k=2 | 82/110 (74.55%) | -0.90% |
| k=4 | 84/110 (76.36%) | +0.91% |

#### Encyclopedic Noise
Injecting factually true but irrelevant Wikipedia sentences:

| Noise Level | Accuracy | vs Baseline |
|-------------|----------|-------------|
| Baseline (clean) | 83/110 (75.45%) | - |
| k=1 | 79/110 (71.82%) | -3.63% |
| k=2 | 81/110 (73.64%) | -1.81% |
| k=4 | 85/110 (77.27%) | +1.82% |

**Key Finding**: GPT-5 shows **no systematic degradation** with noise perturbations on Multi-LogiEval. Performance variance is within normal sampling noise (~3%), indicating robustness to both tautological and encyclopedic distractors at depth-5 reasoning.

---

## Memorization Detection via Perturbation

To assess whether GPT-5 has memorized the benchmark datasets, we conducted perturbation tests that modify questions in ways that should change the correct answer. If a model has memorized the dataset, it will give the **original answer** even when the question is modified.

### Perturbation Types

1. **Remove Critical Premise**: Remove the premise with highest word overlap with the conclusion
2. **Add Contradiction**: Add a negation of the conclusion as an additional premise

### Results (n=378 test cases)

| Test | Total | Answer Changed | Possible Memorization | Rate |
|------|-------|----------------|----------------------|------|
| FOLIO Remove Premise | 100 | 63.0% | 21 | **21.0%** |
| FOLIO Add Contradiction | 100 | 45.0% | 50 | **50.0%** |
| Multi-LogiEval Remove Premise | 89 | 34.8% | 35 | **39.3%** |
| Multi-LogiEval Add Contradiction | 89 | 15.7% | 51 | **57.3%** |
| **TOTAL** | **378** | - | **157** | **41.5%** |

### Key Findings

1. **High memorization signal for "Add Contradiction" perturbations**:
   - FOLIO: 50% of questions show possible memorization
   - Multi-LogiEval: 57.3% show possible memorization
   - When a direct contradiction is added, a non-memorizing model should change its answer

2. **Lower but significant for "Remove Premise"**:
   - FOLIO: 21% possible memorization
   - Multi-LogiEval: 39.3% possible memorization

3. **Multi-LogiEval appears more memorized than FOLIO** across both perturbation types

4. **Overall**: 41.5% of test cases show possible memorization signals

### Interpretation

When a critical premise is removed or a direct contradiction is added, a reasoning model should change its answer. The high rate of unchanged answers (especially 50-57% for contradiction tests) suggests GPT-5 may have memorized portions of these benchmark datasets.

**Implications**: This finding motivates the use of:
- Hardened/perturbed benchmark variants for more reliable evaluation
- Formal verification methods (Lean) that test actual reasoning capability
- Novel benchmarks that the model has not seen during training

---

## Datasets

### FOLIO
- 203 validation questions (first-order logic)
- Natural language premises and conclusions
- Labels: True / False / Unknown

### Multi-LogiEval
- 150 questions across 3 logic types × 5 depths
- **FOL**: First-order logic (50 questions)
- **NM**: Non-monotonic logic (50 questions)
- **PL**: Propositional logic (50 questions)
- Depths 1-5 (10 questions each per type)

### ProverQA (from ProverGen)
- 1500 evaluation instances: 500 easy, 500 medium, 500 hard
- Generated using Prover9 symbolic prover + LLM translation

#### ProverQA Hard Distribution (n=500)

**Answer Distribution:**
| Answer | Count | Percentage |
|--------|-------|------------|
| A (True) | 143 | 28.6% |
| B (False) | 183 | 36.6% |
| C (Uncertain) | 174 | 34.8% |

**Reasoning Steps Distribution:**
| Steps | Count | Percentage |
|-------|-------|------------|
| 6 | 176 | 35.2% |
| 7 | 116 | 23.2% |
| 8 | 87 | 17.4% |
| 5 | 69 | 13.8% |
| 9 | 33 | 6.6% |
| 4 | 16 | 3.2% |
| 3 | 3 | 0.6% |

- Average: 6.56 steps (range 3-9)
- Paper claims "hard" = 6-9 steps; some 3-5 step problems from augmentation

### Depth-7 Extension (Pilot)
- 25 LLM-generated questions at depth-7
- Currently unlabeled (ground truth unknown)
- Used for cross-method agreement analysis

---

## Project Structure

```
├── data/
│   ├── folio_original/           # FOLIO dataset
│   ├── folio_hardened/           # Robustness variants
│   ├── multi_logi_original/      # Multi-LogiEval benchmark
│   ├── multi_logi_hardened/      # Adversarial variants
│   └── multi_logi_d5_only/       # Depth-5 subset
├── src/
│   ├── experiments/              # All experiment scripts
│   │   ├── test_folio*.py        # FOLIO experiments
│   │   └── test_multi*.py        # Multi-LogiEval experiments
│   ├── datasets/                 # Data loading utilities
│   ├── utils/                    # Lean integration, parsing
│   ├── analysis/                 # Error classification
│   └── hardening/                # Robustness testing
├── prompts/                      # System/user prompts
│   ├── folio/
│   ├── multilogi/
│   └── bidirectional/
├── results/                      # Experiment outputs
│   ├── folio/{cot,lean,bidirectional}/
│   └── multilogieval/{all_depths,d5_only}/
├── scripts/                      # Bash runners
├── docs/                         # Documentation
└── nb/                           # Jupyter analysis notebooks
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install openai lean-interact python-dotenv

# Lean 4 (via elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### Run Experiments

```bash
# CoT Baseline
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_async.py \
    --folio_file data/folio_original/folio-validation.json \
    --system_prompt prompts/folio/zero_shot_cot_system.txt \
    --user_prompt prompts/folio/zero_shot_cot_user.txt \
    --model gpt-4o

# Bidirectional Verification
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_folio_bidirectional.py \
    --folio_file data/folio_original/folio-validation.json \
    --model gpt-4o --max_iterations 3 --concurrency 5

# Multi-LogiEval Bidirectional
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_multi_bidirectional.py \
    --data_dir data/multi_logi_original/data \
    --model gpt-4o --max_iterations 3
```

---

## Summary of Findings

### 1. Complexity Inverts Performance Ordering
```
Simple (FOLIO):      CoT (85.7%) > Two-Stage (79.3%) > Lean (74.9%)
Hard (ML d5):        Lean (84.5%) > Two-Stage (75.5%) > CoT (75.5%)
With Bidirectional:  87.27% (+11.8% over CoT)
```

### 2. Error Patterns Reveal Task Nature
- Simple tasks: 77.6% axiomatization (models cheat)
- Hard tasks: 0% axiomatization (genuine reasoning)

### 3. Propositional Logic is Lean's Sweet Spot
- 97.78% accuracy at depth-5 with bidirectional verification

### 4. Bidirectional Detection Works
- 8 formalization errors caught in FOLIO via BOTH_SUCCESS pattern
- All correctly fell back to CoT

---

## Future Directions

1. **Properly labeled depth-7/10 benchmarks** for evaluating extreme reasoning depths
2. **Process Reward Models** using Lean verification as supervision signal
3. **Reverse curriculum learning** (depth 5→1) with Lean-gated progression
4. **Multi-model verification** using different models for TRUE/FALSE proofs

---

## References

- Han, S., et al. (2022). FOLIO: Natural Language Reasoning with First-Order Logic
- Patel, N., et al. (2024). Multi-LogiEval: Evaluating multi-step logical reasoning
- Jiang, D., et al. (2024). LeanReasoner: Boosting complex logical reasoning with Lean. NAACL
- Xi, Z., et al. (2024). Training LLMs for reasoning through reverse curriculum RL. ICML

---

**Model**: GPT-5
**Formal Verification**: Lean 4
**Last Updated**: November 2025
