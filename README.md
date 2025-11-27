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
| Multi-LogiEval d5 only (n=110) | 72.97% | 84.55% | **87.27%** | **+14.30%** over CoT |

**Critical Insight**: As reasoning depth increases, formal verification becomes increasingly valuable:
- Simple tasks (FOLIO): CoT outperforms unidirectional Lean by 10.8pp
- Hard tasks (depth-5): Lean outperforms CoT by **11.8pp**
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
| CoT | 72/100 (72.00%) | 8/10 (80.00%) | 72.73% |
| Lean | 86/100 (86.00%) | 7/10 (70.00%) | 84.55% |
| Bidirectional | 89/100 (89.00%) | 7/10 (70.00%) | **87.27%** |

**Key Observations:**
- Dataset is heavily Yes-biased (91% Yes, 9% No)
- Lean dramatically improves Yes-detection (+14pp over CoT)
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
Hard (ML d5):        Lean (84.5%) > Two-Stage (75.5%) > CoT (72.7%)
With Bidirectional:  87.27% (+14.3% over CoT)
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
