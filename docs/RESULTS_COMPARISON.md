# Comprehensive Results Comparison

## Overview

This document compares all experimental results across models, methods, benchmarks, reasoning depths, and logic types.

---

## 1. FOLIO Benchmark Results (n=203)

### Overall Accuracy by Model and Method

| Model | CoT | Lean | Bidirectional |
|-------|-----|------|---------------|
| **GPT-4o** (Nov 2024) | 85.71% | 74.88% | **87.68%** |
| **DeepSeek-R1-0528** | **84.97%** | - | - |
| **Mistral Large 2512** | 81.22% | - | - |
| Qwen3-235B | 59.50% | - | - |

**Notes:**
- GPT-4o was used for Lean/Bidirectional experiments (Nov 2024)
- Open-source models tested Dec 2025 with CoT only
- Qwen3 underperforms due to excessive "Unknown" outputs

---

## 2. Multi-LogiEval Results by Depth and Logic Type

### 2.1 GPT-4o CoT (n=150, all depths)

| Logic | d1 | d2 | d3 | d4 | d5 | Average |
|-------|----|----|----|----|-----|---------|
| PL | 100% | 70% | 90% | 90% | 70% | **84%** |
| FOL | 80% | 100% | 70% | 100% | 60% | **82%** |
| NM | 50% | 90% | 60% | 80% | 40% | **64%** |
| **Overall** | 77% | 87% | 73% | 90% | 57% | **76.67%** |

### 2.2 GPT-4o Lean (n=150, all depths)

| Logic | d1 | d2 | d3 | d4 | d5 | Average |
|-------|----|----|----|----|-----|---------|
| PL | 90% | 90% | 90% | 80% | 100% | **90%** |
| FOL | 80% | 90% | 90% | 80% | 80% | **84%** |
| NM | 10% | 90% | 60% | 80% | 70% | **62%** |
| **Overall** | 60% | 90% | 80% | 80% | 83% | **78.67%** |

### 2.3 GPT-4o Bidirectional (n=150, all depths)

**Overall Accuracy: 82.67%**
- Agreement Patterns: TRUE_ONLY=89, FALSE_ONLY=12, BOTH_SUCCESS=18, NEITHER=31
- Formalization Errors Detected: 18

---

## 3. Depth-5 Only Results (n=110)

The hardest subset - depth-5 reasoning only.

### By Model and Method

| Model/Method | Overall | FOL | NM | PL |
|--------------|---------|-----|----|----|
| **GPT-4o Bidirectional** | **87.27%** | 88.89% | 60% | 97.78% |
| GPT-4o Lean | 84.55% | 86.67% | 60% | 93.33% |
| GPT-4o CoT | 75.45% | 77.78% | 55% | 82.22% |
| **GPT-5 CoT (Dec 2024)** | **75.45%** | 77.78% | 55% | 82.22% |

**Key Finding**: Formal verification advantage increases at higher depths:
- CoT → Lean: +9.1pp improvement
- CoT → Bidirectional: +11.8pp improvement

---

## 4. Open-Source Model Comparison (Dec 2025)

### 4.1 FOLIO CoT

| Model | Accuracy |
|-------|----------|
| DeepSeek-R1-0528 | **84.97%** |
| Mistral Large 2512 | 81.22% |
| Qwen3-235B | 59.50% |

### 4.2 Multi-LogiEval CoT by Depth and Logic Type

#### DeepSeek-R1-0528 (Overall: 80.69%)

| Logic | d1 | d2 | d3 | d4 | d5 | Average |
|-------|----|----|----|----|-----|---------|
| PL | 100% | 100% | 90% | 80% | 75% | **89%** |
| FOL | 80% | 100% | 70% | 100% | 80% | **86%** |
| NM | 60% | 100% | 50% | 89% | 33% | **66%** |

#### Mistral Large 2512 (Overall: 78.67%)

| Logic | d1 | d2 | d3 | d4 | d5 | Average |
|-------|----|----|----|----|-----|---------|
| PL | 100% | 90% | 80% | 100% | 90% | **92%** |
| FOL | 90% | 100% | 70% | 80% | 70% | **82%** |
| NM | 50% | 90% | 50% | 70% | 50% | **62%** |

#### Qwen3-235B (Overall: 58.00%)

| Logic | d1 | d2 | d3 | d4 | d5 | Average |
|-------|----|----|----|----|-----|---------|
| PL | 80% | 80% | 60% | 60% | 60% | **68%** |
| FOL | 50% | 70% | 60% | 50% | 70% | **60%** |
| NM | 30% | 40% | 40% | 70% | 50% | **46%** |

---

## 5. Adversarial Perturbation Results (GPT-5, Dec 2024)

Testing robustness on Multi-LogiEval depth-5 subset (n=110).

### 5.1 Tautological Noise
Adding logically valid but irrelevant statements (e.g., "All cats are cats").

| Noise Level | Accuracy | vs Baseline | FOL | NM | PL |
|-------------|----------|-------------|-----|----|----|
| Baseline (clean) | 75.45% | - | 77.78% | 55% | 82.22% |
| k=1 | **78.18%** | +2.73% | 84.44% | 50% | 84.44% |
| k=2 | 74.55% | -0.90% | 77.78% | 50% | 82.22% |
| k=4 | 76.36% | +0.91% | 80% | 55% | 82.22% |

### 5.2 Encyclopedic Noise
Adding factually true but irrelevant Wikipedia sentences.

| Noise Level | Accuracy | vs Baseline | FOL | NM | PL |
|-------------|----------|-------------|-----|----|----|
| Baseline (clean) | 75.45% | - | 77.78% | 55% | 82.22% |
| k=1 | 71.82% | -3.63% | 73.33% | 45% | 82.22% |
| k=2 | 73.64% | -1.81% | 77.78% | 50% | 80% |
| k=4 | **77.27%** | +1.82% | 80% | 55% | 84.44% |

**Key Finding**: GPT-5 shows **no systematic degradation** with noise. Variance is within ~3%, indicating robustness to distractors.

---

## 6. Summary Tables

### 6.1 Best Results by Benchmark

| Benchmark | Best Method | Accuracy | Model |
|-----------|-------------|----------|-------|
| FOLIO | Bidirectional | **87.68%** | GPT-4o |
| Multi-LogiEval (all) | Bidirectional | **82.67%** | GPT-4o |
| Multi-LogiEval (d5) | Bidirectional | **87.27%** | GPT-4o |

### 6.2 Logic Type Difficulty Ranking

| Rank | Logic Type | Avg Accuracy | Notes |
|------|------------|--------------|-------|
| 1 | PL (Propositional) | 84-97% | Easiest - clear rules |
| 2 | FOL (First-Order) | 77-89% | Moderate difficulty |
| 3 | NM (Non-Monotonic) | 45-66% | Hardest - exceptions/defaults |

### 6.3 Model Ranking (CoT, Dec 2025)

| Rank | Model | FOLIO | Multi-LogiEval | Notes |
|------|-------|-------|----------------|-------|
| 1 | DeepSeek-R1-0528 | 84.97% | 80.69% | Best open-source |
| 2 | GPT-4o | 85.71% | 76.67% | Strong baseline |
| 3 | Mistral Large 2512 | 81.22% | 78.67% | Consistent performer |
| 4 | Qwen3-235B | 59.50% | 58.00% | Too cautious |

---

## 7. Overcoming Benchmark Saturation

### 7.1 Evidence of Saturation
- FOLIO: Top models hit 85-88%, marginal gains
- Multi-LogiEval d1-d4: 80%+ for top models
- Variance between runs often exceeds method differences

### 7.2 Strategies Implemented

#### Already Done:
1. **Tautological noise** - Model is robust
2. **Encyclopedic noise** - Model is robust
3. **Depth-5 subset** - Still shows differentiation
4. **Multiple models** - Reveals model-specific weaknesses

#### Recommended Next Steps:
1. **Increase reasoning depth** (d7, d10) - More headroom
2. **Novel logic types** - Temporal, probabilistic
3. **Compositional generalization** - Unseen rule combinations
4. **Process-level evaluation** - Trace quality, not just final answer
5. **Adversarial rephrasing** - Same logic, different wording

---

## 8. Key Insights

1. **Depth matters**: Performance ordering inverts at high depth
   - Simple tasks: CoT > Lean
   - Hard tasks (d5): Lean > CoT by 9pp

2. **Bidirectional wins across all benchmarks** by detecting formalization errors

3. **Non-monotonic logic is hardest** for all models (45-66% vs 80-97% for PL)

4. **Open-source models competitive**: DeepSeek-R1 achieves 85% on FOLIO, comparable to GPT-4o

5. **Noise robustness is high**: GPT-5 handles distractors well (within 3% variance)

---

**Last Updated**: December 2025
