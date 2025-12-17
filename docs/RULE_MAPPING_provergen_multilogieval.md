# Rule Mapping: ProverGen vs Multi-LogiEval

## 1. Overview

This document maps the logical patterns and inference rules between ProverGen (Prover9-based) and Multi-LogiEval to identify overlaps and extension opportunities.

---

## 2. Multi-LogiEval Inference Rules (Complete List)

### Propositional/FOL Rules (14 rules)

| Abbrev | Full Name | Formal Definition | NL Pattern |
|--------|-----------|-------------------|------------|
| **MP** | Modus Ponens | ((p→q) ∧ p) ⊢ q | "If P then Q. P is true. Therefore Q." |
| **MT** | Modus Tollens | ((p→q) ∧ ¬q) ⊢ ¬p | "If P then Q. Q is false. Therefore not P." |
| **HS** | Hypothetical Syllogism | ((p→q) ∧ (q→r)) ⊢ (p→r) | "If P then Q. If Q then R. Therefore if P then R." |
| **DS** | Disjunctive Syllogism | ((p∨q) ∧ ¬p) ⊢ q | "Either P or Q. Not P. Therefore Q." |
| **CD** | Constructive Dilemma | ((p→q) ∧ (r→s) ∧ (p∨r)) ⊢ (q∨s) | "If P then Q, if R then S. P or R. Therefore Q or S." |
| **DD** | Destructive Dilemma | ((p→q) ∧ (r→s) ∧ (¬q∨¬s)) ⊢ (¬p∨¬r) | "If P then Q, if R then S. Not Q or not S. Therefore not P or not R." |
| **BD** | Bidirectional Dilemma | ((p→q) ∧ (r→s) ∧ (p∨¬s)) ⊢ (q∨¬r) | "If P then Q, if R then S. P or not S. Therefore Q or not R." |
| **CT** | Commutation | (p∨q) ⊣⊢ (q∨p) | Order of disjunction is reversible |
| **DMT** | De Morgan's Theorem | ¬(p∧q) ⊣⊢ (¬p∨¬q) | "Not (P and Q)" equals "Not P or not Q" |
| **CO** | Composition | ((p→q) ∧ (p→r)) ⊢ (p→(q∧r)) | "If P then Q, if P then R. Therefore if P then Q and R." |
| **IM** | Importation | (p→(q→r)) ⊣⊢ ((p∧q)→r) | Nested conditionals to conjunction |
| **MI** | Material Implication | (p→q) ⊣⊢ (¬p∨q) | Implication equals disjunction form |
| **EG** | Existential Generalization | p(a) ⊢ ∃x(p(x)) | Specific instance implies existence |
| **UI** | Universal Instantiation | ∀x(p(x)) ⊢ p(a) | Universal applies to specific |

### Non-Monotonic (NM) Rules

| Type | Description |
|------|-------------|
| Default Reasoning | "Typically X implies Y" with exceptions |
| Exception Handling | "X except when Y" patterns |
| Priority Reasoning | Conflicting defaults with precedence |

---

## 3. ProverGen Rule Patterns (from rules.json)

### Normal Rules (9 patterns)
```
[F] → [F]                    # Simple implication
[F] ⊕ [F]                    # XOR (exclusive or)
[F] ∨ [F]                    # Disjunction
[F] → ([F] ∧ [F])            # Implication with conjunction consequent
[F] → ([F] ∨ [F])            # Implication with disjunction consequent
[F] → ([F] ⊕ [F])            # Implication with XOR consequent
([F] ∧ [F]) → [F]            # Conjunction antecedent
([F] ∨ [F]) → [F]            # Disjunction antecedent
([F] ⊕ [F]) → [F]            # XOR antecedent
```

### Goal Rules (14 patterns)
Extends normal rules with nested structures like:
```
[F] ∧ ([F] ∧ [F])
[F] ∨ ([F] ∨ [F])
[F] ⊕ ([F] ⊕ [F])
```

### Operators Used
- **→** : Implication
- **⊕** : Exclusive OR (XOR)
- **∨** : Disjunction (OR)
- **∧** : Conjunction (AND)
- **¬** : Negation (in generated facts)
- **∀** : Universal quantifier
- **∃** : Existential quantifier

---

## 4. Rule Mapping: Multi-LogiEval → ProverGen

### Direct Matches

| Multi-LogiEval | ProverGen Pattern | Match Type |
|----------------|-------------------|------------|
| **MP** (Modus Ponens) | `[F] → [F]` + fact assertion | **EXACT** |
| **HS** (Hypothetical Syllogism) | Chain of `[F] → [F]` | **EXACT** - via chaining |
| **DS** (Disjunctive Syllogism) | `[F] ⊕ [F]` + negation | **EXACT** - XOR handles this |
| **UI** (Universal Instantiation) | `∀x(...)` in ProverQA | **EXACT** |
| **EG** (Existential Generalization) | `∃x(...)` in ProverQA | **EXACT** |

### Partial Matches (Compositional)

| Multi-LogiEval | ProverGen Equivalent | Notes |
|----------------|---------------------|-------|
| **MT** (Modus Tollens) | `[F] → [F]` + negated consequent | Derived via contrapositive |
| **CD** (Constructive Dilemma) | `([F] ∨ [F]) → [F]` variants | Can be composed |
| **DD** (Destructive Dilemma) | Composed from MT + disjunction | Multi-step derivation |
| **BD** (Bidirectional Dilemma) | XOR + implication combo | ProverGen handles via `⊕` |
| **CO** (Composition) | `[F] → ([F] ∧ [F])` | **DIRECT MATCH** |
| **DMT** (De Morgan) | Implicit in Prover9 | Handled by theorem prover |
| **CT** (Commutation) | Implicit in Prover9 | Logical equivalence |
| **MI** (Material Implication) | Implicit in Prover9 | Logical equivalence |
| **IM** (Importation) | Derived via `([F] ∧ [F]) → [F]` | Compositional |

### ProverGen-Only Features

| Pattern | Description | Multi-LogiEval Gap |
|---------|-------------|-------------------|
| **XOR (⊕)** | "Either X or Y, but not both" | Multi-LogiEval lacks explicit XOR |
| **Nested XOR** | `[F] ⊕ ([F] ⊕ [F])` | Complex exclusion chains |
| **Uncertain outcomes** | Three-valued logic | Multi-LogiEval is binary only |

### Multi-LogiEval-Only Features

| Rule | Description | ProverGen Gap |
|------|-------------|---------------|
| **Non-Monotonic (NM)** | Default reasoning with exceptions | Not in ProverGen |
| **Priority reasoning** | Conflicting defaults | Not in ProverGen |

---

## 5. Multi-LogiEval Chaining Constraints

### Output-Input Compatibility

Multi-LogiEval chains rules based on **logical output compatibility** - the conclusion type of one rule must match the input type required by the next:

```
Rule -> Can be followed by (based on output type):
----------------------------------------
  BD   -> ['C', 'DS']      # Produces disjunction
  CD   -> ['C', 'DS']      # Produces disjunction
  DD   -> ['DS']           # Produces disjunction
  DS   -> ['MP', 'MT']     # Produces atomic fact
  HS   -> ['CD', 'MP', 'MT']  # Produces implication or feeds into dilemma
  MT   -> ['DMT', 'DS', 'MT'] # Produces negated fact
  MP   -> ['MP']           # Produces atomic fact (can chain)
  DMT  -> ['CO', 'DS']     # Produces disjunction
  CO   -> ['MT']           # Produces conjunction
  I    -> ['MT']           # Importation feeds MT
```

### Chain Starter Rules

Most chains begin with rules that create complex structures:

| Rule | # of Chains | Role |
|------|-------------|------|
| **HS** | 8 | Creates implication chains |
| **BD** | 7 | Creates bidirectional dilemmas |
| **CD** | 6 | Creates constructive dilemmas |
| **DD** | 2 | Creates destructive dilemmas |

### Example Chain: HS_MT_DS_MP (Depth 4)

```
Step 1 (HS): (P→Q) ∧ (Q→R) ⊢ (P→R)
         "If extra projects → demonstrate abilities → considered for promotion"

Step 2 (MT): (P→R) ∧ ¬R ⊢ ¬P
         "Not considered for promotion" → "Did NOT do extra projects"

Step 3 (DS): (P∨D) ∧ ¬P ⊢ D
         "Either extra projects OR communicate goals" + "NOT extra projects"
         → "DID communicate goals"

Step 4 (MP): (D→E) ∧ D ⊢ E
         "If communicate goals → manager understands" + "DID communicate"
         → "Manager understands"
```

### Structural Constraint Pattern

The chains follow this general flow:
```
[Dilemma Rules: BD/CD/DD] → [Disjunction Consumer: DS/C] → [Atomic Rules: MP/MT]
       ↓                              ↓                            ↓
   Creates                        Eliminates                   Propagates
   disjunctions                   alternatives                 facts

[Syllogism: HS] → [Atomic Rules: MP/MT] → [DS] → [MP]
       ↓                    ↓                ↓        ↓
   Creates              Eliminates      Uses        Final
   implication chains   via contrapos.  disjunction conclusion
```

### How ProverGen Differs

**Multi-LogiEval**: Explicit chaining rules, manually designed combinations
- Pros: Interpretable, controlled depth
- Cons: Limited scalability, manual effort

**ProverGen**: Implicit chaining via Prover9 validation
- The generator samples ANY logical expression
- Prover9 validates: "Can these premises prove this conclusion?"
- No explicit chaining rules - soundness checked automatically
- Pros: Arbitrary depth, flexible
- Cons: Less interpretable chains

### Mapping Multi-LogiEval Chains to ProverGen Patterns

| Multi-LogiEval Chain | ProverGen Equivalent |
|---------------------|---------------------|
| `HS → MP` | `[F]→[F]` chain + fact |
| `BD/CD → DS` | `([F]⊕[F])` or `([F]∨[F])` + negation |
| `DS → MP` | Fact from XOR elimination → implication |
| `MT → DS` | Negation creates new disjunction branch |
| `MP → MP` | Sequential `[F]→[F]` applications |

**Key insight**: ProverGen's XOR (⊕) pattern subsumes Multi-LogiEval's BD/CD/DD patterns more elegantly!

---

## 6. Depth Comparison

### Multi-LogiEval Depth Structure
- **d1**: Single inference rule (1 step)
- **d2**: Two-rule combinations (e.g., HS_MP = Hyp.Syllogism + Modus Ponens)
- **d3**: Three-rule combinations (e.g., BD_DS_MT)
- **d4**: Four-rule combinations
- **d5**: Five-rule combinations (e.g., BD_C_DS_MP_MP)

### ProverGen Depth Structure
- **Easy**: 1-2 reasoning steps
- **Medium**: 3-5 reasoning steps
- **Hard**: 6-9 reasoning steps

### Approximate Mapping

| Multi-LogiEval | ProverGen | Reasoning Steps |
|----------------|-----------|-----------------|
| d1 | Easy | 1-2 steps |
| d2-d3 | Easy-Medium | 2-4 steps |
| d4-d5 | Medium | 4-6 steps |
| d7 (extended) | Medium-Hard | 6-8 steps |
| -- | Hard | 8-9 steps |
| d10+ (needed) | Beyond Hard | 10+ steps |

---

## 6. Key Patterns in ProverQA Data

### Pattern 1: XOR with Fact Elimination
```
Context: "Either X has property A or property B, but not both."
         "X does not have property A."
Conclusion: "X has property B."
```
**Equivalent Multi-LogiEval rule**: DS (Disjunctive Syllogism)

### Pattern 2: Chained Implications
```
Context: "If X does A, then X gets B."
         "If X gets B, then X achieves C."
         "If X achieves C, then X receives D."
         "X does A."
Conclusion: "X receives D."
```
**Equivalent Multi-LogiEval rule**: HS_MP_MP (3-step chain)

### Pattern 3: Universal + Modus Ponens
```
Context: "All entities with property P have property Q."
         "X has property P."
Conclusion: "X has property Q."
```
**Equivalent Multi-LogiEval rule**: UI + MP

### Pattern 4: Complex XOR Elimination
```
Context: "X either does A or does B (but not both)."
         "If X does A, then X gets C."
         "X does not get C."
Conclusion: "X does B."
```
**Equivalent Multi-LogiEval rule**: BD variant with MT

---

## 7. Gap Analysis: What Multi-LogiEval Needs from ProverGen

### 1. Explicit XOR (⊕) Support
Multi-LogiEval uses DS which is similar but not identical:
- DS: (p∨q) ∧ ¬p ⊢ q (standard OR)
- XOR: (p⊕q) ∧ ¬p ⊢ q AND (p⊕q) ∧ p ⊢ ¬q

**Action**: Add XOR-based problems to Multi-LogiEval

### 2. Three-Valued Logic (True/False/Uncertain)
Multi-LogiEval only has binary answers (yes/no).
ProverQA includes "Uncertain" for problems where conclusion cannot be determined.

**Action**: Add "Unknown" answer type (already in d7 extension)

### 3. Automated Depth Scaling
Multi-LogiEval manually designs rule combinations.
ProverGen automatically generates problems at any depth via iterative expansion.

**Action**: Adopt ProverGen's top-down generation for d10+

### 4. Prover9 Verification
Multi-LogiEval has no formal verification.
ProverGen uses Prover9 to guarantee logical soundness.

**Action**: Add Prover9 verification layer to Multi-LogiEval generation

---

## 8. Extension Plan: Multi-LogiEval → d10+

### Phase 1: Adopt ProverGen Patterns
1. Add XOR (⊕) operator to Multi-LogiEval rule set
2. Implement "Uncertain" answer type
3. Create 9 new rule patterns matching ProverGen's rules.json

### Phase 2: Integrate Prover9
1. Install LADR/Prover9
2. Create Python wrapper for Prover9 invocation
3. Implement top-down skeleton generation algorithm
4. Add verification step to ensure logical soundness

### Phase 3: Generate Deep Problems
1. Target depths: d7, d10, d15
2. Use ProverGen's iterative expansion
3. Translate skeletons to Multi-LogiEval NL format
4. Validate with Prover9

### Phase 4: Dataset Release
1. Format: Multi-LogiEval compatible JSON
2. Include: nl2fol mappings (from ProverGen)
3. Include: Prover9 proofs for verification

---

## 9. Rule Combination Examples for d10

### Proposed d10 Combinations

| Combination | Rules (10 steps) |
|-------------|------------------|
| d10_combo_1 | HS_CD_DS_MP_HS_MP_MP_MT_DS_MP |
| d10_combo_2 | BD_CT_DS_HS_MP_MP_MP_CO_HS_MT |
| d10_combo_3 | UI_MP_HS_DS_BD_MP_MP_MT_HS_MP |

### Pattern: 10-Step Reasoning Chain

```
Step 1: ∀x(P(x) → Q(x)) + P(a) → Q(a)                    [UI + MP]
Step 2: Q(a) → R(a) + Q(a) → R(a)                        [MP]
Step 3: R(a) → S(a) + R(a) → S(a)                        [MP]
Step 4: (S(a) ∨ T(a)) ∧ ¬T(a) → S(a)                    [DS]
Step 5: S(a) → (U(a) ⊕ V(a))                             [Implication]
Step 6: (U(a) ⊕ V(a)) ∧ ¬U(a) → V(a)                    [XOR elimination]
Step 7: V(a) → W(a) + V(a) → W(a)                        [MP]
Step 8: (W(a) → X(a)) ∧ (Y(a) → Z(a)) ∧ (W(a) ∨ Y(a))   [CD setup]
Step 9: W(a) ∨ Y(a) → X(a) ∨ Z(a)                       [CD]
Step 10: (X(a) ∨ Z(a)) ∧ ¬Z(a) → X(a)                   [DS final]
```

---

## 10. Summary

### Key Findings

1. **Strong overlap**: Most Multi-LogiEval rules map to ProverGen patterns
2. **ProverGen advantage**: Automated depth scaling via Prover9
3. **Multi-LogiEval advantage**: Non-monotonic reasoning, richer NL templates
4. **Gap**: No depth-10+ data exists in either framework

### Recommended Integration

1. Use ProverGen's **generation algorithm** for scalable depth
2. Keep Multi-LogiEval's **rule taxonomy** for interpretability
3. Add **Prover9 verification** to ensure soundness
4. Extend to **d10, d15** using combined approach

### Next Steps

1. Install Prover9/LADR
2. Implement ProverGen's top-down generator
3. Create d10 dataset (500+ samples)
4. Validate with existing bidirectional verification pipeline
