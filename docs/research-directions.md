# Research Directions: Improving LLM Logical Reasoning with Lean

## Current Status

### Baseline Results

| Dataset | CoT | Lean | Delta |
|---------|-----|------|-------|
| FOLIO (203 questions) | **85.7%** | 74.9% | -10.8% |
| Multi-LogiEval (150 questions) | 76.7% | 78.7% | +2.0% |

### Robustness Experiments (Hardened Datasets)

**Tautological Noise (FOLIO):**
| k | CoT | Lean |
|---|-----|------|
| Original | 85.7% | 74.9% |
| k=1 | 84.0% | 70.0% |
| k=2 | 85.5% | 70.4% |
| k=4 | 85.0% | 70.9% |

**Tautological Noise (Multi-LogiEval - CoT):**
- k=1: 76.4%, k=2: 76.1%, k=4: 76.1%

**Encyclopedic Noise (FOLIO - CoT):**
- k=1: 84.0%, k=2: 84.0%, k=4: 85.5%

---

## The Core Problem

### Why Lean Doesn't Help (or Hurts)

Error analysis reveals the root cause: **AXIOMATIZES_CONCLUSION** is the most common error type. The LLM bypasses logical reasoning entirely by directly asserting the conclusion as an axiom:

```lean
-- Example of "cheating" - directly axiomatizes what should be proven
axiom bonnie_performs : PerformsOften Bonnie
```

This means Lean verifies syntactically correct but semantically **invalid** proofs. The LLM isn't doing deduction; it's doing formalization + assertion.

### Why Baselines Are Too High

1. **FOLIO**: Questions are relatively simple (most are 1-2 step reasoning)
2. **Multi-LogiEval**: Even at depth-5, GPT-4 still achieves ~50%+ accuracy
3. **Noise doesn't help**: Tautological/encyclopedic noise only tests robustness to irrelevant information, not reasoning depth

---

## Related Work

### Formal Verification for LLM Reasoning

| Paper | Key Contribution |
|-------|------------------|
| [FoVer (2025)](https://arxiv.org/abs/2505.15960) | Uses Lean/Z3 to automatically label step-level errors for PRM training |
| [Safe (2025)](https://arxiv.org/html/2506.04592v1) | Step-aware formal verification - verifies single steps, not full proofs |
| [LeanReasoner (NAACL 2024)](https://arxiv.org/abs/2403.13312) | Formalizes NL to Lean theorems, uses ReProver for proof search |
| [Autoformalization Survey (2025)](https://arxiv.org/html/2505.23486) | Comprehensive survey of NL-to-formal translation challenges |

### Reasoning Depth and Scaling

| Paper | Key Finding |
|-------|-------------|
| [Multi-LogiEval (2024)](https://arxiv.org/abs/2406.17169) | Performance drops from ~68% (d1) to ~43% (d5) |
| [GSM-Symbolic (ICLR 2025)](https://arxiv.org/abs/2410.05229) | LLMs pattern-match, fail with irrelevant clauses (up to 65% drop) |
| [Reasoning Models Reason Well, Until They Don't (2025)](https://arxiv.org/html/2510.22371) | Sharp depth-correlated failures even on chain graphs |

### Process Reward Models

| Paper | Key Contribution |
|-------|------------------|
| [ThinkPRM (2024)](https://arxiv.org/pdf/2504.16828) | PRMs that generate reasoning before scoring |
| [Math-Shepherd (ACL 2024)](https://aclanthology.org/2024.acl-long.510.pdf) | Step-by-step verification for math reasoning |
| [Rewarding Progress (2024)](https://arxiv.org/abs/2410.08146) | Scaling automated process verifiers |

---

## Proposed Research Directions

### Direction 1: Process Reward Model with Step-Level Lean Supervision (Recommended)

**Motivation**: Instead of verifying whole proofs (which LLMs can "cheat"), verify each reasoning step individually.

**Approach**:
1. Parse Multi-LogiEval/FOLIO into individual reasoning steps
2. Use constrained Lean templates to verify each step (prevents axiomatizing conclusions)
3. Train discriminative PRM: `Input=(context, partial reasoning, next step) → correctness probability`
4. At inference: beam search over reasoning paths, guided by PRM scores

**Why this works**:
- Catches the "axiomatizes conclusion" problem at step-level
- Provides granular feedback instead of binary pass/fail
- Cross-task generalization demonstrated by FoVer

**Contribution**: First use of Lean for step-level supervision in logical reasoning (not just mathematics)

### Direction 2: Deeper Reasoning Benchmarks

**Motivation**: Current benchmarks are too easy. Need depth 7-10 to expose LLM limitations.

**Approach**:
1. Extend Multi-LogiEval to depth 7-10 with harder rule combinations
2. Target baseline performance <30% at depth 10
3. Apply hardening (tautological/encyclopedic noise) at deeper depths
4. Demonstrate that Lean verification gap widens with depth

**Contribution**: First comprehensive evaluation of LLMs on depth 7-10 logical reasoning

### Direction 3: Constrained Formalization Templates

**Motivation**: Prevent LLMs from "cheating" by axiomatizing conclusions.

**Approach**:
1. Design Lean templates with structural constraints:
   - Predicates and implications only
   - No arbitrary axiom declarations
   - Conclusion must be derived, not assumed
2. Force LLM to fill in template slots, not generate free-form code
3. Compare constrained vs. unconstrained on FOLIO/Multi-LogiEval

**Contribution**: Solving the formalization cheating problem for NL logical reasoning

---

## Recommendation

**Start with Direction 1 (PRM with step-level Lean)** because:

1. **Addresses root cause**: Solves the axiomatizes_conclusion problem
2. **Aligns with trends**: FoVer, ThinkPRM, Safe all use step-level verification
3. **Leverages existing infrastructure**: lean-interact, Multi-LogiEval are reusable
4. **High novelty**: Step-level formal verification for logical reasoning (not just math)

### Implementation Plan

1. **Step decomposition**: Parse reasoning chains into individual steps
2. **Template design**: Create Lean templates for common logical operations (modus ponens, conjunction, etc.)
3. **Step verification**: Use templates to verify each step without allowing arbitrary axioms
4. **Data collection**: Generate step-level correctness labels for training
5. **PRM training**: Train discriminator on (context, steps, next_step) → correct/incorrect
6. **Inference pipeline**: Integrate PRM with beam search for guided reasoning

---

## References

- [FoVer: Generalizable PRMs via Formally Verified Training Data](https://arxiv.org/abs/2505.15960)
- [Safe: Step-aware Formal Verification](https://arxiv.org/html/2506.04592v1)
- [Autoformalization Survey](https://arxiv.org/html/2505.23486)
- [LeanReasoner](https://arxiv.org/abs/2403.13312)
- [Multi-LogiEval](https://arxiv.org/abs/2406.17169)
- [GSM-Symbolic](https://arxiv.org/abs/2410.05229)
- [Deductive Reasoning Robustness](https://arxiv.org/abs/2502.04352)
- [Reasoning Models Reason Well, Until They Don't](https://arxiv.org/html/2510.22371)
- [ThinkPRM](https://arxiv.org/pdf/2504.16828)
- [Math-Shepherd](https://aclanthology.org/2024.acl-long.510.pdf)
