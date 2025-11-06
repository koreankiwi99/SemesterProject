# Strategy for Increasing Dataset Difficulty

## Motivation

Current results show LLMs perform well on existing benchmarks:
- **FOLIO CoT**: 85.5% accuracy (running)
- **MultiLogiEval CoT**: 76.7% (all depths), 72.97% (d5 only)
- **With Lean verification**: Even better performance

This indicates current benchmarks are **too easy** for modern LLMs. We need harder benchmarks to:
1. Better distinguish model capabilities
2. Create more challenging training data for curriculum learning
3. Test robustness of reasoning abilities

---

## Strategy 1: Increase Reasoning Depth (6-10 steps)

### Based on LeanReasoner Approach

**Current state:**
- Multi-LogiEval: depths 1-5
- Performance degradation observed: d1 (90%+) → d5 (~50% in paper, but 73% in our tests)

**Proposed extension:**

#### 1.1 Automatic Depth Extension
- **Method**: Chain existing depth-5 problems to create depth 6-10
- **Implementation**:
  - Take two d5 problems (P1, P2)
  - Connect them: conclusion of P1 becomes premise of P2
  - Result: depth-10 problem
  - Similarly: d3 + d3 = d6, d4 + d4 = d8, etc.

#### 1.2 Rule Composition
- **Method**: Add more intermediate reasoning rules
- **Example (depth-6)**:
  ```
  Original d5: A→B, B→C, C∨D, ¬D, E→C ⊢ query
  Extended d6: A→B, B→C, C∨D, ¬D, E→C, F→E ⊢ query
  ```
- **Advantages**: Natural extension, maintains logic type consistency

#### 1.3 Quality Control with Lean
- **Verification**: All extended problems must be Lean-verifiable
- **Process**:
  1. Generate extended problem
  2. Translate to Lean
  3. Verify proof exists
  4. Filter out unprovable or ambiguous cases

#### 1.4 Expected Performance
- Based on LeanReasoner results: each additional depth reduces accuracy by ~10-15%
- Target: d10 should achieve <30% accuracy without Lean

---

## Strategy 2: Surface-Form Perturbations

### Based on GSM-Symbolic Paper (arXiv:2410.05229v2)

**Key insight**: LLMs pattern-match rather than truly reason. Surface changes break patterns.

#### 2.1 Name and Entity Replacement
- **Original**: "Alice is a student. Bob is a professor."
- **Perturbed**: "X47 is a member of class Alpha. Y92 is a member of class Beta."
- **Impact**: Removes semantic priors, forces pure logical reasoning

#### 2.2 Template Variation
- **Method**: Rephrase logical statements while preserving meaning
- **Example**:
  ```
  Original: "If someone is a student, they study."
  Variant 1: "All individuals classified as students engage in studying."
  Variant 2: "The property of studying applies to anyone who is a student."
  Variant 3: "Being a student implies the activity of studying."
  ```

#### 2.3 Symbolic Abstraction
- **Method**: Replace concrete predicates with abstract symbols
- **Example**:
  ```
  Original: "Happy(Alice) ∧ Student(Alice)"
  Symbolic: "P1(x1) ∧ P2(x1)"
  ```
- **Rendering**: Still present in natural language but with generic terms

#### 2.4 Structural Obfuscation
- **Method**: Reorder premises, add irrelevant information
- **Example**:
  ```
  Original premises order: P1, P2, P3, P4, P5
  Shuffled: P3, P1, P5, P2, P4
  + Add 2 irrelevant but valid statements
  ```

#### 2.5 Negation and Complexity
- **Method**: Introduce double negations, complex quantifiers
- **Example**:
  ```
  Original: "All students are hardworking"
  Complex: "There does not exist a student who is not hardworking"
  ```

---

## Implementation Plan

### Phase 1: Depth Extension (Week 1-2)

**Tasks:**
1. Create `data_generation/extend_depth.py`:
   - Load Multi-LogiEval d3-d5 data
   - Implement rule chaining algorithms
   - Generate d6-d10 variants

2. Create `data_generation/verify_extended.py`:
   - Integrate Lean verification
   - Filter for provable problems
   - Quality metrics (uniqueness, difficulty)

3. Target output:
   - 50 questions each for d6, d7, d8, d9, d10 (250 total)
   - All Lean-verified
   - Balanced across FOL, NM, PL

### Phase 2: Surface Perturbations (Week 2-3)

**Tasks:**
1. Create `data_generation/perturb_surface.py`:
   - Implement 5 perturbation types (name, template, symbolic, structural, negation)
   - Apply to existing d1-d5 + new d6-d10
   - Generate multiple variants per problem

2. Perturbation levels:
   - **Level 1 (Mild)**: Name replacement only
   - **Level 2 (Medium)**: Name + template variation
   - **Level 3 (Hard)**: Full symbolic + structural obfuscation

3. Target output:
   - 3 variants per original problem
   - ~450 perturbed problems (150 original × 3 levels)
   - Maintain ground truth labels

### Phase 3: Evaluation (Week 3-4)

**Tasks:**
1. Run baseline experiments:
   - **d6-d10 (unperturbed)**: Test depth impact
   - **d1-d5 (perturbed Level 1-3)**: Test perturbation impact
   - **d6-d10 (perturbed)**: Combined challenge

2. Compare approaches:
   - Zero-shot CoT
   - Lean verification (single-stage)
   - Lean verification (two-stage)

3. Analysis:
   - Plot accuracy vs depth (d1-d10)
   - Plot accuracy vs perturbation level
   - Identify which perturbations are most effective
   - Measure Lean verification success rate on harder problems

---

## Expected Outcomes

### Depth Extension
- **d6**: 60-65% accuracy (CoT), 70-75% (Lean)
- **d7**: 50-55% accuracy (CoT), 60-65% (Lean)
- **d8**: 40-45% accuracy (CoT), 50-55% (Lean)
- **d9**: 30-35% accuracy (CoT), 40-45% (Lean)
- **d10**: 20-25% accuracy (CoT), 30-35% (Lean)

### Perturbation Impact
- **Level 1 (Name)**: -5-10% accuracy drop
- **Level 2 (Name+Template)**: -15-20% accuracy drop
- **Level 3 (Full symbolic)**: -25-35% accuracy drop

### Combined (d10 + Level 3 perturbation)
- **Target**: <15% accuracy (CoT), ~25% (Lean)
- This creates genuinely challenging benchmark for curriculum learning

---

## Technical Considerations

### Lean Verification Challenges
- Deeper proofs may require more Lean iterations
- Increase `max_iterations` from 3 to 5 or 10
- May need more sophisticated Lean prompt engineering

### Data Quality
- Manual validation sample (10% of generated data)
- Check for:
  - Logical consistency
  - Unintended ambiguity
  - Solvability (should be hard but not impossible)

### Baseline Establishment
- Run all perturbation types on small sample first
- Identify which combinations create appropriate difficulty
- Avoid making problems artificially unsolvable

---

## References

1. **LeanReasoner** (Jiang et al., NAACL 2024):
   - Methodology for Lean-based logical reasoning
   - Depth-based problem construction
   - Fine-tuning with <100 samples

2. **GSM-Symbolic** (arXiv:2410.05229v2):
   - Symbolic perturbations for math reasoning
   - Pattern-breaking transformations
   - Performance measurement framework

3. **Multi-LogiEval** (Patel et al., 2024):
   - Original depth 1-5 construction
   - Logic type taxonomy (FOL, NM, PL)

---

## Next Steps

1. ✅ Complete current baseline experiments (d1-d5)
2. ⏳ Implement depth extension scripts
3. ⏳ Implement perturbation scripts
4. ⏳ Generate extended dataset (d6-d10)
5. ⏳ Generate perturbed variants
6. ⏳ Run evaluation experiments
7. ⏳ Analyze results and refine approach
