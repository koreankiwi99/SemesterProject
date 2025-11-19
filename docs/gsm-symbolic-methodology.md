# GSM-SYMBOLIC Methodology

**Paper**: GSM-SYMBOLIC: Understanding the Limitations of Mathematical Reasoning in Large Language Models
**Source**: `pdf/2410.05229v2.pdf`

## Overview

GSM-SYMBOLIC creates symbolic templates from GSM8K to generate diverse question variants and test robustness of mathematical reasoning in LLMs.

---

## Technique 1: Numerical Value Changes (Section 4.2)

### Purpose
Test sensitivity to numerical changes vs. name changes

### Method
- **Names only**: Change person names, places, foods, currencies
- **Numbers only**: Change numerical values in questions
- **Both**: Change names and numbers simultaneously

### Key Findings
- Models more robust to name changes (low variance)
- **Numerical changes cause larger drops** and higher variance
- Gemma2-9b: Names 88.6% (±2.0) → Numbers 83.1% (±2.2)

### Generation
**Manual** - Template creation with variable identification

---

## Technique 2: Adding Clauses (Section 4.3)

### Purpose
Increase reasoning complexity by adding/removing clauses

### Variants
1. **GSM-M1** (Minus 1): Remove 1 clause (easier)
2. **GSM-Symbolic**: Original
3. **GSM-P1** (Plus 1): Add 1 clause (harder)
4. **GSM-P2** (Plus 2): Add 2 clauses (much harder)

### Example
```
GSM-M1: "Pay $0.6/min. After 10 min, drops to $0.5/min.
         How much for 60-min call?"

GSM-P1: "Pay $0.6/min. After 10 min, drops to $0.5/min.
         After 25 min, drops to $0.3/min.
         How much for 60-min call?"

GSM-P2: "Pay $0.6/min. After 10 min, drops to $0.5/min.
         After 25 min, drops to $0.3/min.
         If total > $10, get 25% discount.
         How much for 60-min call?"
```

### Results
As clauses increase:
- **Accuracy drops** (non-linear decline)
- **Variance increases** significantly

| Model | M1 | Symbolic | P1 | P2 |
|-------|----|----|----|----|
| Gemma2-9b | 84.4% | 79.1% | 68.1% | 41.8% |
| Phi-3.5-mini | 87.6% | 82.1% | 64.8% | 44.8% |

### Generation
**Manual** - Hand-crafted additional clauses to templates

---

## Technique 3: GSM-NoOp (Section 4.4)

### Purpose
Test if models understand vs. pattern-match by adding **seemingly relevant but actually irrelevant** information

### Method
Add statements that:
- Appear related to the problem
- Mention numbers/quantities
- **Do NOT affect the reasoning chain** or answer

### Example
**Original**:
> Oliver picks 44 kiwis on Friday, 58 on Saturday.
> On Sunday, he picks double Friday's amount.
> How many total?

**GSM-NoOp**:
> Oliver picks 44 kiwis on Friday, 58 on Saturday.
> On Sunday, he picks double Friday's amount, **but five of them were smaller than average**.
> How many total?

**Correct reasoning**: Ignore "smaller" → 44 + 58 + (2×44) = 190
**Model error**: Subtract 5 → 44 + 58 + 88 - 5 = 185 ❌

### Common Errors
- Models convert irrelevant details into operations
- "smaller" → subtraction
- "discount" → multiplication (regardless of context)
- **Pattern matching without understanding**

### Results - Catastrophic Drops
- Phi-3-mini: **-65.7%**
- Phi-3-small: **-64.0%**
- Gemma2-9b: **-63.0%**
- GPT-4o: **-32.0%**
- o1-preview: **-17.5%**

### Mitigation Attempts (Failed)
1. **NoOp-Symb**: 8 shots of same question from GSM-Symbolic
   - Performance unchanged (within std dev)
2. **NoOp-NoOp**: 8 shots from different GSM-NoOp questions
   - Marginal improvement for some models

### Generation
**Manual** - Carefully crafted irrelevant clauses added to templates

---

## Template Creation Process (Section 3.1)

### Steps
1. Select GSM8K example
2. **Manually identify** variables and their domains
3. Define conditions (e.g., divisibility for whole numbers)
4. Create parsable template with placeholders
5. Apply automated checks (no original values in template)

### Example Template
```python
Template:
When {name} watches her {family}, she gets out toys.
The bag has {x} blocks. The bin has {y} stuffed animals.
The tower has {z} rings. {name} bought a tube of bouncy balls,
bringing her total to {total}. How many bouncy balls?

Variables:
- name: sample(names)
- family: ["nephew", "cousin", "brother"]
- x, y, z: range(5, 100)
- total: range(100, 500)
- ans: range(85, 200)

Conditions:
- x + y + z + ans == total
```

### Generation
**Manual** annotation + **Automated** instantiation from templates

---

## Key Insights

1. **Data Contamination**: Models perform better on original GSM8K than GSM-Symbolic (likely seen in training)

2. **Pattern Matching > Reasoning**: Models fail when:
   - Numbers change (but logic unchanged)
   - Irrelevant info added (can't filter)
   - Complexity increases (non-linear failure)

3. **Not Formal Reasoning**:
   - Performance variance shouldn't exist if true reasoning
   - Shortcuts based on training distribution
   - Can't ignore obviously irrelevant information

---

## Automation Potential

| Technique | Current | Can Automate? |
|-----------|---------|---------------|
| Symbolic templates | Manual | ✓ Parse + extract patterns |
| Name changes | Automated | ✓ Already automated |
| Number changes | Automated | ✓ Already automated |
| Add clauses (P1/P2) | Manual | ⚠️ Need domain knowledge |
| NoOp clauses | Manual | ⚠️ Need semantic verification |

---

## Application to Logical Reasoning

**Challenges for FOLIO/Multi-LogiEval**:
1. Math operations ≠ logical inference rules
2. NoOp in logic = adding irrelevant premises (harder to verify)
3. Clause addition = extending inference chains (feasible)
4. Need formal verification (not just numerical checks)
