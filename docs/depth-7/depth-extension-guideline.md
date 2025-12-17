# Multi-LogiEval Depth Extension Materials
**Based on**: Multi-LogiEval Paper (arXiv:2406.17169v3)

---

## 1. Complete Inference Rules (Table 2)

### Propositional Logic & First-Order Logic Rules

| Rule | Name | PL Form | FOL Form |
|------|------|---------|----------|
| **MP** | Modus Ponens | ((p→q)∧p)⊢q | (∀x(p(x)→q(x))∧p(a))⊢q(a) |
| **MT** | Modus Tollens | ((p→q)∧¬q)⊢¬p | (∀x(p(x)→q(x))∧¬q(a))⊢¬p(a) |
| **HS** | Hypothetical Syllogism | ((p→q)∧(q→r))⊢(p→r) | ∀x((p(x)→q(x))∧(q(x)→r(x)))⊢(p(a)→r(a)) |
| **DS** | Disjunctive Syllogism | ((p∨q)∧¬p)⊢q | (∀x(p(x)∨q(x))∧¬p(a))⊢q(a) |
| **CD** | Constructive Dilemma | ((p→q)∧(r→s)∧(p∨r))⊢(q∨s) | (∀x((p(x)→q(x))∧(r(x)→s(x)))∧(p(a)∨r(a)))⊢(q(a)∨s(a)) |
| **DD** | Destructive Dilemma | ((p→q)∧(r→s)∧(¬q∨¬s))⊢(¬p∨¬r) | (∀x((p(x)→q(x))∧(r(x)→s(x)))∧(¬q(a)∨¬s(a)))⊢(¬p(a)∨¬r(a)) |
| **BD** | Bidirectional Dilemma | ((p→q)∧(r→s)∧(p∨¬s))⊢(q∨¬r) | (∀x((p(x)→q(x))∧(r(x)→s(x)))∧(p(a)∨¬s(a)))⊢(q(a)∨¬r(a)) |
| **CT** | Commutation | (p∨q)⊣⊢(q∨p) | ∀x(p(x)∨q(x))⊣⊢∀x(q(x)∨p(x)) |
| **DMT** | De Morgan's Theorem | ¬(p∧q)⊣⊢¬p∨¬q | ¬∀x(p(x)∧q(x))⊣⊢∃x(¬p(x)∨¬q(x)) |
| **CO** | Composition | ((p→q)∧(p→r))⊢(p→(q∧r)) | ∀x((p(x)→q(x))∧(p(x)→r(x)))⊢∀x(p(x)→(q(x)∧r(x))) |
| **IM** | Importation | (p→(q→r))⊣⊢((p∧q)→r) | ∀x(p(x)→(q(x)→r(x)))⊣⊢∀x((p(x)∧q(x))→r(x)) |
| **MI** | Material Implication | (p→q)⊣⊢(¬p∨q) | (PL only) |
| **EG** | Existential Generalization | - | p(a)⊢∃x(p(x)) |
| **UI** | Universal Instantiation | - | ∀x(p(x))⊢p(a) |

---

## 2. Design Constraints (from Section 3.1)

### Key Quote (page 344-350):
> "We left out inference rules such as **simplification** ((p∧q)⊢p), **conjunction** (p,q⊢(p∧q)), and **addition** (p⊢(p∨q)), as they would lead to **infinite reasoning chains** and it did not make sense to add them as an additional step of reasoning to arrive at a meaningful conclusion."

### Chaining Rule (page 373-378):
> "To generate the combinations, we start with the initial rule and **assess whether the conclusion of this rule aligns with the premise of other rules**. This iterative process results in multi-step combinations/reasoning, with **the conclusion of each step serving as a part of the premise for the subsequent rule**."

### Must Satisfy:
✅ Conclusion of Rule N = Premise of Rule N+1
✅ No infinite loops (no Simplification, Conjunction, Addition)
✅ Human intuitive (understandable to non-logicians)
✅ Meaningful final conclusion

---

## 3. Existing Combinations (Tables 8-11)

### Depth-2 (7 combinations)
1. DS_MP: (P∨Q), (Q→R) | ¬P → R:✓
2. MT_DS: (P→Q), (P∨R) | ¬Q → R:✓
3. HS_MP: (P→Q), (Q→R) | P → R:✓
4. CD_DS: (P→Q), (R→S), (P∨R) | ¬Q → S:✓
5. DD_DS: (P→Q), (R→S), (¬Q∨¬S) | P → R:✗
6. BD_DS: (P→Q), (R→S), (P∨¬S) | ¬Q → R:✗
7. HS_MT: (P→Q), (Q→R) | ¬R → P:✗

### Depth-3 (9 combinations)
1. HS_MP_MP: (P→Q), (Q→R), (R→S) | P → S:✓
2. CD_DS_MP: (P→Q), (R→S), (P∨R), (S→T) | ¬Q → T:✓
3. BD_CT_DS: (P→Q), (R→S), (P∨¬S) | R → Q:✓
4. BD_DS_MT: (P→Q), (R→S), (P∨¬S), (T→R) | ¬Q → T:✗
5. CD_CT_DS: (P→Q), (R→S), (P∨R) | ¬S → Q:✓
6. HS_CD_DS: (P→Q), (Q→R), (S→T), (P∨S) | ¬R → T:✓
7. HS_MT_DS: (P→Q), (Q→R), (P∨S) | ¬R → S:✓
8. DD_DS_MT: (P→Q), (R→S), (¬Q∨¬S), (T→R) | P → T:✗
9. DMT_CO_MT: (P→Q), (P→R) | ¬Q∨¬R → P:✗

### Depth-4 (8 combinations)
1. CD_DS_MP_MP: (P→Q), (R→S), (P∨R), (S→T), (T→U) | ¬Q → U:✓
2. BD_CT_DS_MP: (P→Q), (R→S), (P∨¬S), (Q→T) | R → T:✓
3. BD_DS_MT_DS: (P→Q), (R→S), (P∨¬S), (T→R), (T∨U) | ¬Q → U:✓
4. HS_CD_DS_MP: (P→Q), (Q→R), (S→T), (P∨S), (T→U) | ¬R → U:✓
5. CD_CT_DS_MP: (P→Q), (R→S), (P∨R), (Q→T) | ¬S → T:✓
6. HS_MT_DS_MP: (P→Q), (Q→R), (P∨S), (S→T) | ¬R → T:✓
7. BD_DS_MT_MT: (P→Q), (R→S), (P∨¬S), (T→R), (U→T) | ¬Q → U:✗
8. IM_MT_DMT_DS: (P→(Q∧R)) | Q,¬R → P:✗

### Depth-5 (3 combinations)
1. **HS_MT_DS_MP_MP**: (P→Q), (Q→R), (P∨S), (S→T), (T→U) | ¬R → U:✓
2. **BD_CT_DS_MP_MP**: (P→Q), (R→S), (P∨¬S), (Q→T), (T→U) | R → U:✓
3. **CD_CT_DS_MP_MP**: (P→Q), (R→S), (P∨R), (Q→T), (T→U) | ¬S → U:✓

---

## 4. Prompt Template (Figure 4, Appendix C)

### Complete Prompt Structure

```
Generalized Rule Definition:
[Define each rule in natural language using variables {P}, {Q}, {R}, etc.]
Example: Rule 1: [if {P} is true then {Q} is true]

Formatting Instruction:
Complete the following tasks, only returning text in exactly the format given in the following examples.

Diversity Instruction:
Generate 5 more examples from multiple domains

Task Definition:
Task 1: Generate a short real life story that includes sentences to illustrate the above rules,
replacing the entities P, Q, R, S, T with real values. Do not include the entity labels like
P, Q, R, S, T in the story.

Task 2: Generate the following complex reasoning question using the story and the rules,
by replacing the respective entities.
Q1: [If ... is not true, then is ... true?]

Examples:
[Provide 3 complete examples with propositions, context, and question]
```

### Example for Depth-3 (CD_DS_MP):

```
Generalized Rule Definition:
Rule 1: [if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {P} or {R} or both are true]
Rule 2: [if {S} is true, then {T} is true]

Formatting Instruction:
Complete the following tasks, only returning text in exactly the format given in the following examples.

Diversity Instruction:
Generate 5 more examples from multiple domains

Task Definition:
Task 1: Generate a short real life story that includes sentences to illustrate the above rules, replacing the entities P, Q, R, S, T with real values. Do not include the entity labels like P, Q, R, S, T in the story.
Task 2: Generate the following complex reasoning question using the story and the rules, by replacing the respective entities.
Q1: [If Q is not true, then is T true?]

Examples:
Context: Jeff wants to improve his health and fitness. If Jeff meditates regularly, he will improve his overall mental health. Also, if Jeff eats healthy nutritious meals, he is likely to lose weight. Jeff decides to either meditate regularly, or eat healthy meals, or do both simultaneously. He also knew that if he loses weight, then he will feel more confident about himself.
{P}: Jeff meditates regularly.
{Q}: Jeff improves his mental health.
{R}: Jeff eats healthy meals.
{S}: Jeff loses weight.
{T}: Jeff feels more confident about himself.
Question: If Jeff did not improve his mental health, did he feel more confident about himself?

[Add 2 more similar examples]
```

---

## 5. Data Structure Format

### JSON File Structure

```json
{
    "logic": "pl",
    "rule": "HS_MT_DS_MP_MP",
    "depth": "d5",
    "samples": [
        {
            "id": 1,
            "context": "Liam had a big exam coming up...",
            "question": "If Liam does not pass the exam, then does he run bussiness?",
            "answer": "yes"
        }
    ]
}
```

### Naming Convention
- **File**: `[RULE_COMBINATION].json`
- **Example**: `HS_MT_DS_MP_MP.json` for depth-5
- **For d7**: `HS_MT_DS_MP_MP_MP_MP.json`
- **For d10**: `HS_MT_DS_MP_MP_MP_MP_MP_MP_MP.json`

---

## 6. Validation Checklist (from Section 3.3)

From paper (page 552-561):
> "We examine each context for potential discrepancies throughout the data generation phase, ensuring they are **logically correct** and represent the intended logical relations. We also dedicated considerable effort to eliminating typos and validating the grammar."

---

## 7. Implementation Steps

1. **Define Rule Combinations** (Manual)
   - Select 3-5 base patterns from d5
   - Extend with additional MP/MT/DS steps
   - Verify chaining constraint satisfied

2. **Create Formal Specifications** (Manual)
   - Document premises in story
   - Document premise in question
   - Specify expected answer
   - List all variables

3. **Generate NL Instances** (Claude-2/GPT-4)
   - Use prompt template
   - Generate 10 samples per combination
   - Request diverse domains

4. **Manual Validation** (Critical)
   - Check all validation criteria
   - Fix grammar/typos
   - Verify logical correctness
   - Ensure naturalness

5. **Save to JSON** (Automated)
   - Follow data structure format
   - Save to appropriate directory
   - Maintain naming convention

---

## 9. Key Papers Referenced

1. **Patel, N., et al. (2024)**. Multi-LogiEval: Towards evaluating multi-step logical reasoning ability of large language models. arXiv:2406.17169v3
   - Primary source for methodology

2. **Parmar, M., et al. (2024)**. LogicBench: Towards systematic evaluation of logical reasoning ability of large language models. ACL 2024.
   - Source of 25 baseline inference rules

3. **Lifschitz, V. (1989)**. Benchmark problems for formal nonmonotonic reasoning: Version 2.00.
   - Non-monotonic logic rules (depth-1)

4. **Han, S., et al. (2022)**. FOLIO: Natural language reasoning with first-order logic. arXiv:2209.00840
   - FOL reasoning benchmark

5. **Tafjord, O., et al. (2021)**. ProofWriter: Generating implications, proofs, and abductive statements over natural language. ACL Findings.
   - Multi-hop proof generation