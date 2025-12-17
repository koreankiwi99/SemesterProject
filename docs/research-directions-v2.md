# Research Directions v2: Bidirectional Verification for NL Logical Reasoning

## The Problem Summary

1. **Baselines too high**: FOLIO 85.7%, Multi-LogiEval 76.7% - not much room to improve
2. **Lean hurts performance**: 85.7% → 74.9% on FOLIO because LLM "cheats" by axiomatizing conclusions
3. **Noise doesn't help**: Tautological/encyclopedic perturbations barely affect accuracy
4. **Prompt engineering failed**: Constraining the LLM to not axiomatize just breaks the Lean code generation

---

## Existing Approaches (and why they're not enough)

| Method | How it works | Limitation |
|--------|--------------|------------|
| **LINC** | NL → FOL → Prover9 | FOL too limited for complex NL |
| **Logic-LM** | NL → symbolic → Z3/Prover9 + self-refinement | Translation errors cascade |
| **LeanReasoner** | NL → Lean → ReProver proof search | Requires fine-tuned prover; same axiomatization problem |

All follow the same pipeline: `NL → Formalize → Prove → Answer`

**They all trust a single proof attempt.** If the LLM cheats (axiomatizes the conclusion), the solver happily verifies it.

### References
- [LINC (EMNLP 2023)](https://arxiv.org/abs/2310.15164)
- [Logic-LM (EMNLP 2023)](https://arxiv.org/abs/2305.12295)
- [LeanReasoner (NAACL 2024)](https://arxiv.org/abs/2403.13312)

---

## Novel Idea 1: Bidirectional Proof Verification

**Core insight**: Don't trust one proof. Try to prove BOTH the conclusion AND its negation.

```
For "Is X true?":
  1. Try to prove X is TRUE  → success/fail
  2. Try to prove X is FALSE → success/fail
```

### Agreement Logic

| Prove TRUE | Prove FALSE | Interpretation | Action |
|------------|-------------|----------------|--------|
| Success | Fail | Valid TRUE proof | Answer: TRUE |
| Fail | Success | Valid FALSE proof | Answer: FALSE |
| **Success** | **Success** | **Formalization error!** | Reject, fall back to CoT |
| Fail | Fail | Proof generation failed | Fall back to CoT |

### Why This Catches Cheating

If the LLM axiomatizes "X is true" to prove TRUE, and axiomatizes "X is false" to prove FALSE, **both will succeed**. This signals the formalization is broken, and we can reject it.

```lean
-- Attempt 1: Prove TRUE
axiom x_is_true : X  -- LLM cheats
theorem conclusion_true : X := x_is_true  -- Lean accepts ✓

-- Attempt 2: Prove FALSE
axiom x_is_false : ¬X  -- LLM cheats again
theorem conclusion_false : ¬X := x_is_false  -- Lean accepts ✓

-- BOTH succeed → Contradiction detected → Reject formalization
```

### Implementation Sketch

```python
def bidirectional_verify(problem: str) -> str:
    # Generate two proof attempts
    proof_true = llm_generate_lean_proof(problem, target="TRUE")
    proof_false = llm_generate_lean_proof(problem, target="FALSE")

    # Verify with Lean
    true_ok = lean_verify(proof_true)
    false_ok = lean_verify(proof_false)

    # Agreement logic
    if true_ok and not false_ok:
        return "TRUE"
    elif false_ok and not true_ok:
        return "FALSE"
    elif true_ok and false_ok:
        # Formalization error detected!
        return cot_fallback(problem)
    else:
        # Neither provable - likely UNCERTAIN or proof failure
        return "UNCERTAIN"
```

---

## Novel Idea 2: Wild-State Problem Generation

**Problem**: Current benchmarks are too easy. We need problems where:
- CoT systematically fails (<40%)
- But Lean CAN solve them (>80% potential)

### "Wild-State" Characteristics

1. **Negation chains**: "It is not the case that all non-X fail to be Y" (humans lose track)
2. **Many distractors**: 10 premises, only 3 matter
3. **Deep dependencies**: Answer requires 7+ inference steps
4. **Quantifier complexity**: Mix of "all", "some", "none" with tricky scoping

### Example Comparison

**Easy (CoT succeeds)**:
> All dogs bark. Fido is a dog. Does Fido bark?
> → TRUE (CoT handles this easily)

**Wild-State (CoT fails, Lean succeeds)**:
> It is not the case that all animals that are not cats fail to bark. Fido is an animal. Fido is not a cat. No animal that barks is silent. Everything that is not silent makes noise. Does Fido make noise?

The wild-state version requires tracking:
- Double negation: "not...not cats fail to bark" = non-cats bark
- Conditional chain: barks → not silent → makes noise
- Proper scoping of quantifiers

### Generation Pipeline

```python
def generate_wild_state_problem(difficulty: int):
    # 1. Start with valid logical structure
    premises, conclusion, label = generate_valid_fol_problem(depth=difficulty)

    # 2. Add distractors (irrelevant but plausible premises)
    premises = inject_distractors(premises, k=difficulty*2)

    # 3. Add negation complexity
    premises = add_negation_chains(premises)

    # 4. Naturalize to ambiguous NL
    nl_problem = formalize_to_natural_language(premises, conclusion)

    # 5. Verify: CoT fails, Lean succeeds
    assert cot_accuracy(nl_problem) < 0.5
    assert lean_accuracy(nl_problem) > 0.8

    return nl_problem
```

---

## Why This Is Novel

| Aspect | Existing Work | Our Approach |
|--------|---------------|--------------|
| Verification | Single proof | Bidirectional (prove + disprove) |
| Error detection | None (trust the proof) | Automatic via contradiction |
| Benchmarks | Easy (FOLIO ~85%) | Wild-state (target CoT <40%) |
| Training | Some require fine-tuning | **Pure inference-time** |
| Focus | Math theorem proving | **Natural language logic** |

### Project Differentiator

Most Lean+LLM work targets **mathematical theorem proving** (MiniF2F, PutnamBench, IMO problems). We focus on **natural language logical reasoning** - FOL, propositional, and non-monotonic logic expressed in everyday language.

---

## Research Plan

### Phase 1: Implement Bidirectional Verification
- Test on existing FOLIO/Multi-LogiEval
- Measure how often both TRUE and FALSE proofs succeed (= formalization errors caught)
- Compare accuracy vs single-proof approach

### Phase 2: Generate Wild-State Benchmark
- Extend Multi-LogiEval rules to depth 7-10
- Add distractors, negation chains
- Validate: CoT <40%, Lean potential >80%
- Target: 100+ problems across difficulty levels

### Phase 3: Full Comparison

| Method | FOLIO | Multi-LogiEval | Wild-State |
|--------|-------|----------------|------------|
| CoT | 85.7% | 76.7% | <40% (target) |
| LINC | ~85% | ? | ? |
| Logic-LM | ~85% | ? | ? |
| LeanReasoner | ~90% | ? | ? |
| **Bidirectional (Ours)** | ? | ? | >70% (target) |

### Success Criteria

1. **Wild-State benchmark**: Create problems where CoT <40%, Lean >80%
2. **Bidirectional verification**: Catch >80% of formalization errors automatically
3. **Overall improvement**: Outperform single-proof Lean on Wild-State by >15%

---

## Open Questions

1. How to handle the UNCERTAIN case in bidirectional verification?
2. Can we generate wild-state problems that are still natural-sounding?
3. What's the right balance of difficulty vs solvability?
4. Should we combine bidirectional verification with self-refinement (Logic-LM style)?

---

## References

### Neurosymbolic Reasoning
- [LINC: Logical Inference via Neurosymbolic Computation (EMNLP 2023)](https://arxiv.org/abs/2310.15164)
- [Logic-LM: Empowering LLMs with Symbolic Solvers (EMNLP 2023)](https://arxiv.org/abs/2305.12295)
- [LeanReasoner: Boosting Complex Logical Reasoning (NAACL 2024)](https://arxiv.org/abs/2403.13312)

### Benchmarks
- [Multi-LogiEval (2024)](https://arxiv.org/abs/2406.17169)
- [FOLIO: Natural Language Reasoning with FOL](https://arxiv.org/abs/2209.00840)
- [GSM-Symbolic (ICLR 2025)](https://arxiv.org/abs/2410.05229)

### Robustness
- [Investigating Robustness of Deductive Reasoning (2025)](https://arxiv.org/abs/2502.04352)
