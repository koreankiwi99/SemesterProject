# Error Classification Prompts

Prompts for classifying root causes of false negatives (Lean passed but wrong answer).

## Version History

### v1.txt
Initial version with 6 categories:
- AXIOMATIZES_CONCLUSION
- AXIOMATIZES_CONTRADICTION
- AXIOMATIZES_UNMENTIONED
- INCORRECT_FORMALIZATION
- REASONING_FAILURE
- OTHER

**Issues**: "REASONING_FAILURE" too broad, caught many formalization errors.

### v2.txt
Renamed categories for clarity:
- AXIOMATIZES_CONCLUSION
- AXIOMATIZES_CONTRADICTION
- FABRICATES_ENTITY_FACT (renamed from AXIOMATIZES_UNMENTIONED)
- INCORRECT_FORMALIZATION
- REASONING_GAP (renamed from REASONING_FAILURE)
- OTHER

**Issues**: Same as v1 - classifier still over-uses REASONING_GAP.

### v3.txt
Major restructure with AXIOMATIZE_* naming convention:
- AXIOMATIZE_CONCLUSION - directly states what should be proved
- AXIOMATIZE_CONTRADICTION - creates logical contradiction
- AXIOMATIZE_FABRICATION - invents facts not in premises
- FORMALIZE_INCORRECTLY - wrong translation (quantifiers, implication direction)
- FORMALIZE_INCOMPLETE - missing axioms for stated premises
- PROOF_INCOMPLETE - correct axioms but incomplete proof
- OTHER

**Changes from v2**:
1. Split formalization into INCORRECTLY vs INCOMPLETE
2. Renamed to AXIOMATIZE_* prefix (user preferred over "gaming" framing)
3. Added concrete examples for each category

**Issues found in testing**:
- Classifier too easily said "axioms are correct" → PROOF_INCOMPLETE
- Case 36: Should be FORMALIZE_INCORRECTLY (wrong implication direction for "composers write music")
- Case 5: Should be FORMALIZE_INCOMPLETE (missing "James is an employee" axiom)

**v3 revision** added:
1. IMPORTANT section requiring verification before classifying
2. Explicit checklist: every premise has axiom, implication directions, quantifiers
3. Better example for FORMALIZE_INCORRECTLY showing "Composers write music" error
4. Note that PROOF_INCOMPLETE should be rare

### v4.txt
Uses decision tree approach for more accurate classification:
- AXIOMATIZE_CONCLUSION
- AXIOMATIZE_CONTRADICTION
- AXIOMATIZE_FABRICATION
- FORMALIZE_INCORRECTLY
- MISSING_PREMISES (renamed from FORMALIZE_INCOMPLETE)
- PROOF_INCOMPLETE
- OTHER

**Changes from v3**:
1. Renamed FORMALIZE_INCOMPLETE → MISSING_PREMISES
2. Restructured as decision tree (Q1-Q8) to force systematic checking
3. Q1: Check for `False.elim`, `False.rec`, `absurd` in proof → AXIOMATIZE_CONTRADICTION
4. Q3: Self-negating patterns `P x → ... ∧ ¬P x` in single axiom → AXIOMATIZE_CONTRADICTION
5. Q4: Conflicting axiom pairs → AXIOMATIZE_CONTRADICTION
6. Q6: Explicit distinction between wrong axiom (FORMALIZE_INCORRECTLY) vs missing axiom (MISSING_PREMISES)
7. Added `decision_path` field to trace classification reasoning

**V4 Results (GPT-5 FOLIO baseline)**:
| Dataset | False Negatives | Distribution |
|---------|-----------------|--------------|
| Baseline 1 (no max) | 27 | FORMALIZE_INCORRECTLY: 22, AXIOMATIZE_CONTRADICTION: 3, MISSING_PREMISES: 2 |
| Baseline 2 (4096 max) | 23 | FORMALIZE_INCORRECTLY: 17, AXIOMATIZE_CONTRADICTION: 4, MISSING_PREMISES: 2 |

**Issues fixed from v3_revised**:
- Case 77: ✓ Now correctly AXIOMATIZE_CONTRADICTION (self-negating `Need → ¬Need`)
- Case 83: ✓ Now correctly AXIOMATIZE_CONTRADICTION (proof uses `False.elim`)
- Case 119: ✓ Now correctly FORMALIZE_INCORRECTLY (wrong argument order)

## Category Definitions

### Gaming Categories (AXIOMATIZE_*)
Model "cheats" by adding axioms that make proof trivial:
- **AXIOMATIZE_CONCLUSION**: Directly states what needs to be proved
- **AXIOMATIZE_CONTRADICTION**: Creates False, from which anything follows
- **AXIOMATIZE_FABRICATION**: Invents facts not in premises

### Formalization Categories
Model attempts honest translation but makes errors:
- **FORMALIZE_INCORRECTLY**: Premise exists but mistranslated
- **FORMALIZE_INCOMPLETE** / **MISSING_PREMISES** (v4): Premise exists but no axiom for it

### Other
- **PROOF_INCOMPLETE**: Axioms correct, but proof logic incomplete
- **OTHER**: Doesn't fit other categories

## Usage

```bash
PYTHONPATH=src:$PYTHONPATH python src/analysis/analyze_errors.py \
    --results <results.json> \
    --prompt prompts/error-classification/v3.txt \
    --folio_data data/folio/original/folio-validation.json
```
