# Depth-10 Rule Combinations for Multi-LogiEval

## Inference Rules Definition

- **MP (Modus Ponens)**: `(P → Q) ∧ P ⊢ Q` ("If P then Q, and P is true, therefore Q")
- **MT (Modus Tollens)**: `(P → Q) ∧ ¬Q ⊢ ¬P` ("If P then Q, and Q is false, therefore P is false")
- **HS (Hypothetical Syllogism)**: `(P → Q) ∧ (Q → R) ⊢ (P → R)` ("If P then Q, and if Q then R, therefore if P then R")
- **DS (Disjunctive Syllogism)**: `(P ∨ Q) ∧ ¬P ⊢ Q` ("Either P or Q, and not P, therefore Q")
- **CD (Constructive Dilemma)**: `(P → Q) ∧ (R → S) ∧ (P ∨ R) ⊢ (Q ∨ S)`
- **DD (Destructive Dilemma)**: `(P → Q) ∧ (R → S) ∧ (¬Q ∨ ¬S) ⊢ (¬P ∨ ¬R)`
- **BD (Bidirectional Dilemma)**: `(P → Q) ∧ (R → S) ∧ (P ∨ ¬S) ⊢ (Q ∨ ¬R)`
- **CT (Commutation)**: `(P ∨ Q) ⊣⊢ (Q ∨ P)`

## Extension Strategy: d7 → d10

d10 extends d7 by adding 3 additional reasoning steps, typically using:
- MP chains (simple linear reasoning)
- HS → MP combinations (compress two implications into one, then apply)
- MT → DS combinations (negation-based branching)

## Depth-10 Rule Combinations

### Combination 1: HS → CD → DS → MP → HS → MP → MP → MP → MP → MP

**Base:** d7 Combination 1 (HS → CD → DS → MP → HS → MP → MP)
**Extension:** + MP → MP → MP (3 additional implications)

| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **Step 1 - HS**: `(P → Q) ∧ (Q → R) ⊢ (P → R)`<br>**Step 2 - CD**: `(P → R) ∧ (S → T) ∧ (P ∨ S) ⊢ (R ∨ T)`<br>**Step 3 - DS**: `(R ∨ T) ∧ ¬R ⊢ T`<br>**Step 4 - MP**: `(T → U) ∧ T ⊢ U`<br>**Step 5 - HS**: `(U → V) ∧ (V → W) ⊢ (U → W)`<br>**Step 6 - MP**: `(U → W) ∧ U ⊢ W`<br>**Step 7 - MP**: `(W → X) ∧ W ⊢ X`<br>**Step 8 - MP**: `(X → Y) ∧ X ⊢ Y`<br>**Step 9 - MP**: `(Y → Z) ∧ Y ⊢ Z`<br>**Step 10 - MP**: `(Z → Ω) ∧ Z ⊢ Ω` | `(P → Q)`,<br>`(Q → R)`,<br>`(S → T)`,<br>`(P ∨ S)`,<br>`(T → U)`,<br>`(U → V)`,<br>`(V → W)`,<br>`(W → X)`,<br>`(X → Y)`,<br>`(Y → Z)`,<br>`(Z → Ω)` | `¬R` | `Ω: ✓` |

### Combination 2: BD → CT → DS → HS → MP → MP → MP → HS → MP → MP

**Base:** d7 Combination 2 (BD → CT → DS → HS → MP → MP → MP)
**Extension:** + HS → MP → MP (compress + apply twice)

| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **Step 1 - BD**: `(P → Q) ∧ (R → S) ∧ (P ∨ ¬S) ⊢ (Q ∨ ¬R)`<br>**Step 2 - CT**: `(Q ∨ ¬R) ⊣⊢ (¬R ∨ Q)`<br>**Step 3 - DS**: `(¬R ∨ Q) ∧ R ⊢ Q`<br>**Step 4 - HS**: `(Q → T) ∧ (T → U) ⊢ (Q → U)`<br>**Step 5 - MP**: `(Q → U) ∧ Q ⊢ U`<br>**Step 6 - MP**: `(U → V) ∧ U ⊢ V`<br>**Step 7 - MP**: `(V → W) ∧ V ⊢ W`<br>**Step 8 - HS**: `(W → X) ∧ (X → Y) ⊢ (W → Y)`<br>**Step 9 - MP**: `(W → Y) ∧ W ⊢ Y`<br>**Step 10 - MP**: `(Y → Z) ∧ Y ⊢ Z` | `(P → Q)`,<br>`(R → S)`,<br>`(P ∨ ¬S)`,<br>`(Q → T)`,<br>`(T → U)`,<br>`(U → V)`,<br>`(V → W)`,<br>`(W → X)`,<br>`(X → Y)`,<br>`(Y → Z)` | `R` | `Z: ✓` |

### Combination 3: CD → DS → HS → CD → DS → MP → MP → MP → MP → MP

**Base:** d7 Combination 3 (CD → DS → HS → CD → DS → MP → MP)
**Extension:** + MP → MP → MP (3 additional implications)

| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **Step 1 - CD**: `(P → Q) ∧ (R → S) ∧ (P ∨ R) ⊢ (Q ∨ S)`<br>**Step 2 - DS**: `(Q ∨ S) ∧ ¬Q ⊢ S`<br>**Step 3 - HS**: `(S → T) ∧ (T → U) ⊢ (S → U)`<br>**Step 4 - CD**: `(S → U) ∧ (V → W) ∧ (S ∨ V) ⊢ (U ∨ W)`<br>**Step 5 - DS**: `(U ∨ W) ∧ ¬U ⊢ W`<br>**Step 6 - MP**: `(W → X) ∧ W ⊢ X`<br>**Step 7 - MP**: `(X → Y) ∧ X ⊢ Y`<br>**Step 8 - MP**: `(Y → Z) ∧ Y ⊢ Z`<br>**Step 9 - MP**: `(Z → Α) ∧ Z ⊢ Α`<br>**Step 10 - MP**: `(Α → Β) ∧ Α ⊢ Β` | `(P → Q)`,<br>`(R → S)`,<br>`(P ∨ R)`,<br>`(S → T)`,<br>`(T → U)`,<br>`(V → W)`,<br>`(S ∨ V)`,<br>`(W → X)`,<br>`(X → Y)`,<br>`(Y → Z)`,<br>`(Z → Α)`,<br>`(Α → Β)` | `¬Q, ¬U` | `Β: ✓` |

### Combination 4: HS → MT → DS → BD → CT → DS → MP → MP → MP → MP

**Base:** d7 Combination 4 (HS → MT → DS → BD → CT → DS → MP)
**Extension:** + MP → MP → MP (3 additional implications)

| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **Step 1 - HS**: `(P → Q) ∧ (Q → R) ⊢ (P → R)`<br>**Step 2 - MT**: `(P → R) ∧ ¬R ⊢ ¬P`<br>**Step 3 - DS**: `(P ∨ S) ∧ ¬P ⊢ S`<br>**Step 4 - BD**: `(S → T) ∧ (U → V) ∧ (S ∨ ¬V) ⊢ (T ∨ ¬U)`<br>**Step 5 - CT**: `(T ∨ ¬U) ⊣⊢ (¬U ∨ T)`<br>**Step 6 - DS**: `(¬U ∨ T) ∧ U ⊢ T`<br>**Step 7 - MP**: `(T → W) ∧ T ⊢ W`<br>**Step 8 - MP**: `(W → X) ∧ W ⊢ X`<br>**Step 9 - MP**: `(X → Y) ∧ X ⊢ Y`<br>**Step 10 - MP**: `(Y → Z) ∧ Y ⊢ Z` | `(P → Q)`,<br>`(Q → R)`,<br>`(P ∨ S)`,<br>`(S → T)`,<br>`(U → V)`,<br>`(S ∨ ¬V)`,<br>`(T → W)`,<br>`(W → X)`,<br>`(X → Y)`,<br>`(Y → Z)` | `¬R, U` | `Z: ✓` |

### Combination 5: DD → DS → HS → MT → DS → MP → MP → HS → MP → MP

**Base:** d7 Combination 5 (DD → DS → HS → MT → DS → MP → MP)
**Extension:** + HS → MP → MP (compress + apply twice)

| Rule Combinations | Premises in Story | Premise in Question | Answer |
|-------------------|-------------------|---------------------|---------|
| **Step 1 - DD**: `(P → Q) ∧ (R → S) ∧ (¬Q ∨ ¬S) ⊢ (¬P ∨ ¬R)`<br>**Step 2 - DS**: `(¬P ∨ ¬R) ∧ P ⊢ ¬R`<br>**Step 3 - HS**: `(T → U) ∧ (U → V) ⊢ (T → V)`<br>**Step 4 - MT**: `(T → V) ∧ ¬V ⊢ ¬T`<br>**Step 5 - DS**: `(T ∨ W) ∧ ¬T ⊢ W`<br>**Step 6 - MP**: `(W → X) ∧ W ⊢ X`<br>**Step 7 - MP**: `(X → Y) ∧ X ⊢ Y`<br>**Step 8 - HS**: `(Y → Z) ∧ (Z → Α) ⊢ (Y → Α)`<br>**Step 9 - MP**: `(Y → Α) ∧ Y ⊢ Α`<br>**Step 10 - MP**: `(Α → Β) ∧ Α ⊢ Β` | `(P → Q)`,<br>`(R → S)`,<br>`(¬Q ∨ ¬S)`,<br>`(T → U)`,<br>`(U → V)`,<br>`(T ∨ W)`,<br>`(W → X)`,<br>`(X → Y)`,<br>`(Y → Z)`,<br>`(Z → Α)`,<br>`(Α → Β)` | `P, ¬V` | `Β: ✓` |

## Summary Table

| Combination | Rule Chain | Variables | Question Premise | Answer |
|-------------|------------|-----------|------------------|--------|
| 1 | HS→CD→DS→MP→HS→MP→MP→MP→MP→MP | P,Q,R,S,T,U,V,W,X,Y,Z,Ω | ¬R | Ω: ✓ |
| 2 | BD→CT→DS→HS→MP→MP→MP→HS→MP→MP | P,Q,R,S,T,U,V,W,X,Y,Z | R | Z: ✓ |
| 3 | CD→DS→HS→CD→DS→MP→MP→MP→MP→MP | P,Q,R,S,T,U,V,W,X,Y,Z,Α,Β | ¬Q, ¬U | Β: ✓ |
| 4 | HS→MT→DS→BD→CT→DS→MP→MP→MP→MP | P,Q,R,S,T,U,V,W,X,Y,Z | ¬R, U | Z: ✓ |
| 5 | DD→DS→HS→MT→DS→MP→MP→HS→MP→MP | P,Q,R,S,T,U,V,W,X,Y,Z,Α,Β | P, ¬V | Β: ✓ |

## Design Notes

1. **Variable naming**: For d10, we extend beyond single letters to use Greek letters (Α, Β, Ω) for the final conclusions
2. **Chaining constraint**: Each rule's conclusion must connect to the next rule's premise
3. **Balance**: Mix of complex rules (CD, DD, BD) at the start with simple chains (MP, HS) for extension
4. **Diversity**: Each combination uses different opening patterns while sharing the extended MP chains
