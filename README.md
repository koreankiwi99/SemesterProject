# Reverse Curriculum Learning for Multi-Step Logical Reasoning with Lean 4

**EPFL NLP Lab Semester Project**

## Overview

This project investigates curriculum learning approaches for multi-step logical reasoning, using Lean 4 formal verification. We address performance degradation in LLMs as reasoning depth increases (single-step: 90%+, 5-step: often <50%).

## Baseline Evaluation (Completed)

**Datasets:**
- Multi-LogiEval: 150 questions (FOL/non-monotonic/propositional logic, depths 1-5)
- FOLIO: 203 validation questions (first-order logic)

**Results (Multi-LogiEval):**

| Approach | Overall | FOL | Non-Monotonic | Propositional |
|----------|---------|-----|---------------|---------------|
| Zero-shot CoT | 76.7% | 82% | 64% | 84% |
| Lean Verification | **84.0%** | **88%** | **68%** | **96%** |

**Findings:** Lean verification adds +7.3% without training. Success rate: 98.7% (148/150).

## Two Proposed Directions

### Option 1: Reverse Curriculum Training

**Hypothesis:** Training depth 5→1 with Lean-gated progression outperforms forward curriculum (1→5).

**Basis:** R³ (Xi et al., 2024) showed reverse curriculum success—starting near complete solutions provides stronger supervision.

**Experiments:**
- Reverse: depth 5→4→3→2→1
- Forward: depth 1→2→3→4→5 (baseline)
- Random: random depth order (control)
- Gated vs non-gated (gate: advance after ≥80% Lean verification)

**Setup:** Fine-tune GPT-4-mini on Lean-verified examples from Multi-LogiEval. Measure accuracy, sample efficiency, depth degradation.

**Contribution:** Empirical validation of reverse curriculum with formal verification.

### Option 2: Process Reward Models with Lean Supervision

**Hypothesis:** Train step-level verifiers using Lean as supervision signal, enabling inference-time search over reasoning paths.

**Background:** Process Reward Models (PRMs) provide feedback at each reasoning step rather than only final answers. OpenAI (2023) showed process supervision outperforms outcome supervision for mathematical reasoning. Recent work (2024-2025) demonstrates PRMs enable smaller models to outperform larger ones through reward-guided search.

**Our Approach:**
- Use Lean verification as ground truth for process supervision
- Train PRM to predict step-level correctness (verified by Lean)
- At inference: generate multiple reasoning paths, use PRM to guide search toward Lean-verifiable solutions
- Unlike human annotation (expensive) or MC estimation (noisy), Lean provides binary, objective step verification

**Advantages:**
- Scalable supervision: Lean automatically labels reasoning steps
- Interpretable: PRM learns what makes reasoning steps formally valid
- Inference-time compute: Search over reasoning paths without retraining base model
- Complementary to Option 1: Can combine with curriculum training

**Implementation:**
1. Collect reasoning traces with Lean verification for each step
2. Train discriminative PRM: input=(context, partial reasoning, next step) → output=probability step is Lean-verifiable
3. At inference: beam search or MCTS over reasoning steps, guided by PRM scores
4. Verify final proof with Lean

**Contribution:** First use of formal verification as supervision signal for process reward models in logical reasoning.

## Current Infrastructure

**Evaluation modes:** Zero-shot CoT (baseline), Direct Lean (iterative refinement), Two-stage (NL → Lean, tracks answer drift)

**Implementation:** Interactive Lean 4 via `lean-interact`, max 3 refinement iterations, comprehensive logging

## Project Structure

```
├── data/          # Multi-LogiEval, FOLIO datasets
├── prompts/       # System/user prompts
├── src/
│   ├── experiments/   # test_multi*.py, test_folio*.py
│   ├── utils/         # Lean integration, parsing
│   └── datasets/      # Data loading
├── results/       # Experiment outputs
└── scripts/       # Bash runners
```

## Dependencies

Python 3.10+, `openai`, `lean-interact`, Lean 4 (via `elan`)

## References

- Xi, Z., et al. (2024). Training large language models for reasoning through reverse curriculum reinforcement learning. ICML.
- Zhang, A. (2025). Recursive Language Models. https://alexzhang13.github.io/blog/2025/rlm/
- Jiang, D., et al. (2024). LeanReasoner: Boosting complex logical reasoning with Lean. NAACL.
- Patel, N., et al. (2024). Multi-LogiEval: Towards evaluating multi-step logical reasoning ability of large language models.
