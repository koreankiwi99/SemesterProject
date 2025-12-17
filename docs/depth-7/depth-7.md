# Depth-7 Rule Combinations for Multi-LogiEval

## Inference Rules Definition

- **MP (Modus Ponens)**: `(P → Q) ∧ P ⊢ Q` ("If P then Q, and P is true, therefore Q")
- **MT (Modus Tollens)**: `(P → Q) ∧ ¬Q ⊢ ¬P` ("If P then Q, and Q is false, therefore P is false")
- **HS (Hypothetical Syllogism)**: `(P → Q) ∧ (Q → R) ⊢ (P → R)` ("If P then Q, and if Q then R, therefore if P then R")
- **DS (Disjunctive Syllogism)**: `(P ∨ Q) ∧ ¬P ⊢ Q` ("Either P or Q, and not P, therefore Q")
- **CD (Constructive Dilemma)**: `(P → Q) ∧ (R → S) ∧ (P ∨ R) ⊢ (Q ∨ S)`
- **DD (Destructive Dilemma)**: `(P → Q) ∧ (R → S) ∧ (¬Q ∨ ¬S) ⊢ (¬P ∨ ¬R)`
- **BD (Bidirectional Dilemma)**: `(P → Q) ∧ (R → S) ∧ (P ∨ ¬S) ⊢ (Q ∨ ¬R)`
- **CT (Commutation)**: `(P ∨ Q) ⊣⊢ (Q ∨ P)`

## Depth-7 Rule Combinations

### Combination 1
| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **HS**: `(P → Q) ∧ (Q → R) ⊢ (P → R)`<br>**CD**: `(P → R) ∧ (S → T) ∧ (P ∨ S) ⊢ (R ∨ T)`<br>**DS**: `(R ∨ T) ∧ ¬R ⊢ T`<br>**MP**: `(T → U) ∧ T ⊢ U`<br>**HS**: `(U → V) ∧ (V → W) ⊢ (U → W)`<br>**MP**: `(U → W) ∧ U ⊢ W`<br>**MP**: `(W → X) ∧ W ⊢ X` | `(P → Q)`,<br>`(Q → R)`,<br>`(S → T)`,<br>`(P ∨ S)`,<br>`(T → U)`,<br>`(U → V)`,<br>`(V → W)`,<br>`(W → X)` | `¬R` | `X: ✓` |

### Combination 2
| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **BD**: `(P → Q) ∧ (R → S) ∧ (P ∨ ¬S) ⊢ (Q ∨ ¬R)`<br>**CT**: `(Q ∨ ¬R) ⊣⊢ (¬R ∨ Q)`<br>**DS**: `(¬R ∨ Q) ∧ R ⊢ Q`<br>**HS**: `(Q → T) ∧ (T → U) ⊢ (Q → U)`<br>**MP**: `(Q → U) ∧ Q ⊢ U`<br>**MP**: `(U → V) ∧ U ⊢ V`<br>**MP**: `(V → W) ∧ V ⊢ W` | `(P → Q)`,<br>`(R → S)`,<br>`(P ∨ ¬S)`,<br>`(Q → T)`,<br>`(T → U)`,<br>`(U → V)`,<br>`(V → W)` | `R` | `W: ✓` |

### Combination 3
| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **CD**: `(P → Q) ∧ (R → S) ∧ (P ∨ R) ⊢ (Q ∨ S)`<br>**DS**: `(Q ∨ S) ∧ ¬Q ⊢ S`<br>**HS**: `(S → T) ∧ (T → U) ⊢ (S → U)`<br>**CD**: `(S → U) ∧ (V → W) ∧ (S ∨ V) ⊢ (U ∨ W)`<br>**DS**: `(U ∨ W) ∧ ¬U ⊢ W`<br>**MP**: `(W → X) ∧ W ⊢ X`<br>**MP**: `(X → Y) ∧ X ⊢ Y` | `(P → Q)`,<br>`(R → S)`,<br>`(P ∨ R)`,<br>`(S → T)`,<br>`(T → U)`,<br>`(V → W)`,<br>`(S ∨ V)`,<br>`(W → X)`,<br>`(X → Y)` | `¬Q, ¬U` | `Y: ✓` |

### Combination 4
| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **HS**: `(P → Q) ∧ (Q → R) ⊢ (P → R)`<br>**MT**: `(P → R) ∧ ¬R ⊢ ¬P`<br>**DS**: `(P ∨ S) ∧ ¬P ⊢ S`<br>**BD**: `(S → T) ∧ (U → V) ∧ (S ∨ ¬V) ⊢ (T ∨ ¬U)`<br>**CT**: `(T ∨ ¬U) ⊣⊢ (¬U ∨ T)`<br>**DS**: `(¬U ∨ T) ∧ U ⊢ T`<br>**MP**: `(T → W) ∧ T ⊢ W` | `(P → Q)`,<br>`(Q → R)`,<br>`(P ∨ S)`,<br>`(S → T)`,<br>`(U → V)`,<br>`(S ∨ ¬V)`,<br>`(T → W)` | `¬R, U` | `W: ✓` |

### Combination 5
| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **DD**: `(P → Q) ∧ (R → S) ∧ (¬Q ∨ ¬S) ⊢ (¬P ∨ ¬R)`<br>**DS**: `(¬P ∨ ¬R) ∧ P ⊢ ¬R`<br>**HS**: `(T → U) ∧ (U → V) ⊢ (T → V)`<br>**MT**: `(T → V) ∧ ¬V ⊢ ¬T`<br>**DS**: `(T ∨ W) ∧ ¬T ⊢ W`<br>**MP**: `(W → X) ∧ W ⊢ X`<br>**MP**: `(X → Y) ∧ X ⊢ Y` | `(P → Q)`,<br>`(R → S)`,<br>`(¬Q ∨ ¬S)`,<br>`(T → U)`,<br>`(U → V)`,<br>`(T ∨ W)`,<br>`(W → X)`,<br>`(X → Y)` | `P, ¬V` | `Y: ✓` |