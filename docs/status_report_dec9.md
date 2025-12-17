# Status Report: GPT-5 & Multi-Model Experiments

**Date:** December 9, 2025

---

## 1. GPT-5 Results Summary

### FOLIO Dataset (n=203)
- Simple means w/o CoT

| Method | Reasoning_Effort | Overall | True | False | Unknown |
|--------|---------|---------|------|-------|---------|
| **CoT** | high | 85.7% | 83.3% | 83.9% | 89.9% |
| **Simple** | high | 85.0% | 83.1% | 83.6% | 88.2% |
| **CoT** | minimal | 79.0% | - | - | - |
| **Simple** | minimal | 71.0% | 69.0% | 68.9% | 75.0% |
| **Lean** | high | 74.9% | 94.4% | 56.5% | 71.0% |
| **Bidirectional** | high | **87.7%** | 88.9% | 87.1% | 87.0% |
| **Two-Stage** | high | 79.3% | 83.3% | 77.4% | 76.8% |

**Insight on reasoning_effort:**
- High reasoning (default): 85.7%
- Minimal reasoning: 79.0% (-6.7%)
- Simple + minimal: 71.0% (-14.7%)

### Multi-LogiEval All Depths (d1-d5, n=150)
- Simple means w/o CoT

| Method | Reasoning_Effort | Overall | FOL | NM | PL |
|--------|------------------|---------|-----|----|----|
| **CoT** | high | 76.7% | 82.0% | 64.0% | 84.0% |
| **Simple** | high | 72.7% | 78.0% | 58.0% | 82.0% |
| **CoT** | minimal | 74.7% | 78.0% | 62.0% | 84.0% |
| **Simple** | minimal | 51.3% | 50.0% | 52.0% | 52.0% |
| **Lean** | high | 78.7% | 84.0% | 62.0% | 90.0% |
| **Bidirectional** | high | **82.7%** | 86.0% | 70.0% | 92.0% |
| **Two-Stage** | high | 72.7% | 84.0% | 54.0% | 80.0% |

### Multi-LogiEval by Depth (GPT-5 CoT high reasoning)

| Depth | Overall | FOL | NM | PL |
|-------|---------|-----|----|----|
| d1 | 76.7% | 80% | 50% | 100% |
| d2 | 86.7% | 100% | 90% | 70% |
| d3 | 73.3% | 70% | 60% | 90% |
| d4 | 90.0% | 100% | 80% | 90% |
| d5 | **56.7%** | 60% | 40% | 70% |

### Multi-LogiEval by Depth (GPT-5 Simple minimal reasoning)

| Depth | Overall | FOL | NM | PL |
|-------|---------|-----|----|----|
| d1 | 60.0% | 90% | 20% | 70% |
| d2 | 70.0% | 70% | 90% | 50% |
| d3 | 56.7% | 50% | 60% | 60% |
| d4 | 40.0% | 30% | 50% | 40% |
| d5 | **30.0%** | 10% | 40% | 40% |

**Insight:** Minimal reasoning + no CoT shows dramatic degradation at higher depths (d5: 30% vs 56.7% with high reasoning).

### Multi-LogiEval Depth-5 Only (n=110)

| Method | Overall | FOL | NM | PL |
|--------|---------|-----|----|----|
| **CoT** | 72.7% | 75.6% | 45.0% | 82.2% |
| **Lean** | 84.5% | 86.7% | 60.0% | 93.3% |
| **Bidirectional** | **87.3%** | 88.9% | 60.0% | **97.8%** |

---

## 2. Other Models (CoT Only - All Depths)

### FOLIO

| Model | Overall | True | False | Unknown |
|-------|---------|------|-------|---------|
| GPT-5 | 85.7% | 83.3% | 83.9% | 89.9% |
| DeepSeek-R1 | 83.5% | 91.5% | 88.5% | 70.6% |
| Mistral Large | 80.0% | 87.3% | 82.0% | 70.6% |
| Qwen3-235B | 59.5% | 40.8% | 52.5% | 85.3% |

### Multi-LogiEval (All Depths d1-d5)

| Model | Overall | FOL | NM | PL |
|-------|---------|-----|----|----|
| GPT-5 | 76.7% | 82.0% | 64.0% | 84.0% |
| DeepSeek-R1 | 80.0% | 86.0% | 64.0% | 90.0% |
| Mistral Large | 78.7% | 82.0% | 62.0% | 92.0% |
| Qwen3-235B | 58.0% | 60.0% | 46.0% | 68.0% |

### Multi-LogiEval by Depth (All Models CoT)

| Depth | GPT-5 | DeepSeek-R1 | Mistral Large | Qwen3-235B |
|-------|-------|-------------|---------------|------------|
| d1 | 76.7% | 80.0% | 80.0% | 53.3% |
| d2 | 86.7% | 100.0% | 93.3% | 63.3% |
| d3 | 73.3% | 70.0% | 66.7% | 53.3% |
| d4 | 90.0% | 86.7% | 83.3% | 60.0% |
| d5 | **56.7%** | **63.3%** | **70.0%** | **60.0%** |

❌ d5_only NOT run on other models

---

## 3. Key Insights

### Dataset Saturation Evidence

1. **FOLIO:** Top 3 models at 80-86% ceiling
2. **Multi-LogiEval d1-d4:** 67-97% accuracy (saturated)
3. **Multi-LogiEval d5:** Only challenging subset (30-70% range)

### reasoning_effort Impact (GPT-5)

| Configuration | FOLIO | Multi-LogiEval |
|--------------|-------|----------------|
| CoT + High | 85.7% | 76.7% |
| CoT + Minimal | 79.0% | 74.7% |
| Simple + High | 85.0% | 72.7% |
| Simple + Minimal | 71.0% | **51.3%** |

**Insight:** Minimal reasoning shows largest degradation on Multi-LogiEval (51.3%), especially at d5 (30%).

### Non-Monotonic Logic (NM) is Hardest

| Dataset | FOL | NM | PL |
|---------|-----|----|----|
| Multi-LogiEval d1-d5 | 82-86% | **46-64%** | 84-92% |
| Multi-LogiEval d5 only | 76-89% | **45-60%** | 82-98% |

NM consistently 15-30pp below other logic types across all models.

### Bidirectional Value

| Dataset | Bidir vs CoT |
|---------|--------------|
| FOLIO | +2.0% (87.7% vs 85.7%) |
| Multi-LogiEval d1-d5 | +6.0% (82.7% vs 76.7%) |
| **Multi-LogiEval d5 only** | **+14.6%** (87.3% vs 72.7%) |

Formal verification gains increase with task difficulty.

---

## 4. Conclusion

### Completed
- ✅ GPT-5: All methods (CoT, Lean, Bidirectional, Two-Stage)
- ✅ GPT-5: reasoning_effort variations (high vs minimal)
- ✅ GPT-5: Simple vs CoT prompts
- ✅ Other models: CoT on all depths
- ✅ GPT-5: d5_only subset

### Not Completed
- Other models: d5_only subset
- Other models: Lean/Bidirectional (not recommended due to saturation)