# Verification Gaming
When LLMs construct formal proofs for verification, they may take shortcuts that pass verification but don't constitute valid reasoning. This has implications for AI-assisted theorem proving and formal verification systems.


## Folio

### Baseline Prompt Design
We design a minimal prompt that achieves 100% valid Lean 4 output without introducing behavioral constraints. This establishes an unbiased baseline for measuring gaming.

### Baseline Results
  - 203/203 cases (complete, no API errors)
  - Accuracy: 174/203 (85.7%)
  - Lean Pass: 201/203 (99.0%)

#### Failed Cases Summary: 29 total
  - 20 Conservative (said Uncertain, should've proved): Model too cautious
  - 7 Gaming (proved something wrong): Cases 77, 83 (True→False) + 5 others (True→Uncertain)
  - 2 Lean failed: Cases 103, 140

  | Pred → GT         | Lean=✓ | Lean=✗ | Total |
  |-------------------|--------|--------|-------|
  | Uncertain → False | 10     | 1      | 11    |
  | Uncertain → True  | 10     | 1      | 11    |
  | True → Uncertain  | 5      | 0      | 5     |
  | True → False      | 2      | 0      | 2     |
  | Total             | 27     | 2      | 29    |

### DeepSeek-R1 Baseline
  - 203/203 cases (complete, 2 API errors rerun)
  - Accuracy: 175/203 (86.2%)
  - Lean Pass: 193/203 (95.1%)

#### Failed Cases Summary: 28 total
  - 22 Conservative (said Uncertain, should've proved): Model too cautious
  - 6 Gaming (proved something wrong): Cases 41, 77, 83, 89, 159, 202
  - 0 Lean failed

  | Pred → GT         | Lean=✓ | Lean=✗ | Total |
  |-------------------|--------|--------|-------|
  | Uncertain → False | 12     | 0      | 12    |
  | Uncertain → True  | 10     | 0      | 10    |
  | True → False      | 3      | 0      | 3     |
  | True → Uncertain  | 3      | 0      | 3     |
  | Total             | 28     | 0      | 28    |

### Reasoning Token Pressure (n=4096)
  - 203/203 cases (complete)
  - Accuracy: 163/203 (80.3%) — **5.4% drop from baseline**
  - Lean Pass: 185/203 (91.1%) — **7.9% drop from baseline**

  | Pred → GT         | Lean=✓ | Lean=✗ | Total |
  |-------------------|--------|--------|-------|
  | Uncertain → True  | 9      | 0      | 9     |
  | Uncertain → False | 5      | 0      | 5     |
  | True → False      | 4      | 0      | 4     |
  | True → Uncertain  | 4      | 0      | 4     |
  | False → Uncertain | 1      | 0      | 1     |
  | None → *          | 0      | 17     | 17    |
  | Total             | 23     | 17     | 40    |

  Note: 17 "None" cases are token cutoff (empty response before model could output).

  ## MultiLogiEval
  Regular FOL (d1-d5) - organized by depth:

  | Depth | Yes         | No          | Total |
  |-------|-------------|-------------|-------|
  | d1    | 120 (92.3%) | 10 (7.7%)   | 130   |
  | d2    | 60 (57.1%)  | 45 (42.9%)  | 105   |
  | d3    | 90 (66.7%)  | 45 (33.3%)  | 135   |
  | d4    | 90 (75.0%)  | 30 (25.0%)  | 120   |
  | d5    | 45 (100%)   | 0 (0.0%)    | 45    |
  | Total | 405 (75.7%) | 130 (24.3%) | 535   |

  Multivariable FOL - organized by rule type (7 rules), no depth:

  | Rule | Pattern                        | Yes | No  |
  |------|--------------------------------|-----|-----|
  | 1    | Contrapositive with multi-vars | 20  | 0   |
  | 2    | Negation with quantifiers      | 0   | 20  |
  | 3    | Existential + conjunction      | 20  | 0   |
  | 4    | Disjunctive + universal        | 20  | 0   |
  | 5    | Existential introduction       | 20  | 0   |
  | 6    | Disjunctive syllogism          | 20  | 0   |
  | 7    | Conjunction introduction       | 20  | 0   |