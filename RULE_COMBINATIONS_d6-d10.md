# Rule Combinations for d6-d10

## Analysis of Existing Pattern (d1-d5)

### Rule Definitions:
- **MP** (Modus Ponens): (p→q)∧p ⊢ q
- **MT** (Modus Tollens): (p→q)∧¬q ⊢ ¬p
- **HS** (Hypothetical Syllogism): (p→q)∧(q→r) ⊢ (p→r)
- **DS** (Disjunctive Syllogism): (p∨q)∧¬p ⊢ q
- **CD** (Constructive Dilemma): (p→q)∧(r→s)∧(p∨r) ⊢ (q∨s)
- **DD** (Destructive Dilemma): (p→q)∧(r→s)∧(¬q∨¬s) ⊢ (¬p∨¬r)
- **BD** (Biconditional): p↔q, p ⊢ q
- **C** (Commutation): p∨q ⊢ q∨p
- **DMT** (De Morgan's): ¬(p∧q) ⊢ ¬p∨¬q
- **CO** (Composition): (p→q)∧(p→r) ⊢ p→(q∧r)
- **CT** (Commutation): p∨q ⊢ q∨p
- **IM** (Importation): (p∧q)→r ⊢ p→(q→r)
- **EG** (Existential Generalization): p(a) ⊢ ∃x(p(x))
- **UI** (Universal Instantiation): ∀x(p(x)) ⊢ p(a)
- **I** (probably Implication or similar)

### Observed Patterns:

**Depth 5 (3 combinations):**
- BD_C_DS_MP_MP
- CD_C_DS_MP_MP
- HS_MT_DS_MP_MP

**Pattern:** They end with common suffixes (MP_MP, DS_MP_MP) built on d3/d4 bases

**Depth 4 (8 combinations):**
- BD_C_DS_MP
- BD_DS_MT_DS
- BD_DS_MT_MT
- CD_C_DS_MP
- CD_DS_MP_MP
- HS_CD_DS_MP
- HS_MT_DS_MP
- I_MT_DMT_DS

**Pattern:** Mix of different rule types, often ending in MP or DS

**Depth 3 (9 combinations):**
- BD_C_DS
- BD_DS_MT
- CD_C_DS
- CD_DS_MP
- DD_DS_MT
- DMT_CO_MT
- HS_CD_DS
- HS_MP_MP
- HS_MT_DS

**Pattern:** Various starting points (BD, CD, DD, HS) with different middles

## Proposed d6 Combinations (Based on Pattern Extension)

### Strategy: Extend d5 combinations by adding one more rule

1. **BD_C_DS_MP_MP_DS** (extend BD_C_DS_MP_MP)
   - Biconditional → Commutation → Disjunctive Syll → MP → MP → DS

2. **CD_C_DS_MP_MP_DS** (extend CD_C_DS_MP_MP)
   - Constructive Dilemma → Commutation → DS → MP → MP → DS

3. **HS_MT_DS_MP_MP_DS** (extend HS_MT_DS_MP_MP)
   - Hypothetical Syll → MT → DS → MP → MP → DS

4. **BD_C_DS_MP_MP_MT** (extend BD_C_DS_MP_MP)
   - BD → C → DS → MP → MP → MT

5. **CD_C_DS_MP_MP_MT** (extend CD_C_DS_MP_MP)
   - CD → C → DS → MP → MP → MT

6. **HS_MT_DS_MP_MP_MT** (extend HS_MT_DS_MP_MP)
   - HS → MT → DS → MP → MP → MT

7. **BD_DS_MT_DS_MP_MP** (extend BD_DS_MT_DS from d4)
   - BD → DS → MT → DS → MP → MP

8. **CD_DS_MP_MP_DS_MP** (extend CD_DS_MP_MP from d4)
   - CD → DS → MP → MP → DS → MP

9. **HS_CD_DS_MP_DS_MP** (extend HS_CD_DS_MP from d4)
   - HS → CD → DS → MP → DS → MP

10. **BD_DS_MT_MT_DS_MP** (extend BD_DS_MT_MT from d4)
    - BD → DS → MT → MT → DS → MP

## Proposed d7 Combinations

### Strategy: Further extend d6 or combine patterns

1. **BD_C_DS_MP_MP_DS_MT**
2. **CD_C_DS_MP_MP_DS_MT**
3. **HS_MT_DS_MP_MP_DS_MT**
4. **BD_C_DS_MP_MP_MT_DS**
5. **CD_C_DS_MP_MP_MT_DS**
6. **HS_MT_DS_MP_MP_MT_DS**
7. **BD_DS_MT_DS_MP_MP_DS**
8. **CD_DS_MP_MP_DS_MP_MT**

## Proposed d8 Combinations

1. **BD_C_DS_MP_MP_DS_MT_DS**
2. **CD_C_DS_MP_MP_DS_MT_DS**
3. **HS_MT_DS_MP_MP_DS_MT_DS**
4. **BD_C_DS_MP_MP_MT_DS_MP**
5. **CD_C_DS_MP_MP_MT_DS_MP**
6. **BD_DS_MT_DS_MP_MP_DS_MT**

## Proposed d9 Combinations

1. **BD_C_DS_MP_MP_DS_MT_DS_MP**
2. **CD_C_DS_MP_MP_DS_MT_DS_MP**
3. **HS_MT_DS_MP_MP_DS_MT_DS_MP**
4. **BD_C_DS_MP_MP_MT_DS_MP_MT**

## Proposed d10 Combinations

1. **BD_C_DS_MP_MP_DS_MT_DS_MP_MT**
2. **CD_C_DS_MP_MP_DS_MT_DS_MP_MT**
3. **HS_MT_DS_MP_MP_DS_MT_DS_MP_MT**

## Validation Criteria

For each combination, verify:
1. **Sequential chaining**: Output of rule N feeds into input of rule N+1
2. **Human intuition**: Natural flow, not artificial
3. **No infinite loops**: Avoid rules like simplification
4. **Testable**: Can create yes/no questions

## Notes

- These combinations follow the pattern from d1-d5
- They extend existing d5 bases with additional MP, MT, DS steps
- MP (Modus Ponens) and DS (Disjunctive Syllogism) are most common for extension
- Need to verify each combination chains properly before generation
