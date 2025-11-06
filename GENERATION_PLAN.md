# Plan for Generating d6-d10 Dataset (Based on Multi-LogiEval Paper)

## What the Paper Says About Generation
- Used Claude-2 in few-shot setting
- 5-component prompt: Rule Definition, Format, Diversity, Task, Examples
- Generated stories replacing P,Q,R with real entities
- Manual validation of all instances
- Generated across domains (education, finance, workplace)

## Answers to Key Questions

### 1) Are we gonna use existing questions?

**NO** - we'll generate NEW questions for d6-d10.

Why not reuse existing?
- Existing only goes to d5
- Can't just concatenate (different stories/entities)
- Need 6-10 rule chains, which don't exist yet

What we WILL reuse:
- The METHODOLOGY (rule combinations → stories)
- The STRUCTURE (context + question format)
- The QUALITY standards (manual validation)

### 2) Are there restrictions on combining rules?

**YES** - From the paper:

**EXCLUDED RULES:**
- Rules creating infinite chains (simplification, conjunction, addition)
- Unintuitive rules (principle of explosion: p∧¬p⊢q)

**VALID COMBINATIONS MUST:**
- "Conclusion of one rule aligns with premise of next rule"
- Follow sequential dependency graph
- Be human-intuitive (not just logically valid)

**For NON-MONOTONIC:**
- Max 1 NM rule per problem (to avoid overly long contexts)
- Rest are classical logic rules (MP, MT, HS, etc.)

**EXAMPLES FROM DATA:**
- d2: HS_MP (valid - HS conclusion feeds MP)
- d3: HS_MP_MP (valid - chains properly)
- d5: BD_C_DS_MP_MP (valid - 5 rules chain)

So for d6-d10, we need to:
- Pick rule sequences where outputs→inputs connect
- Avoid unintuitive combinations
- Ensure human understandability

### 3) How to ensure quality and diversity?

Following paper's approach:

**QUALITY ASSURANCE:**
- Manual validation of ALL instances for logical correctness
- Grammar and typo checking
- Verify rule sequence actually requires d steps
- Check answer is definitively yes/no (not ambiguous)

**DIVERSITY STRATEGIES:**
- Generate across domains:
  - Education (students, exams, studying)
  - Finance (investments, markets, profits)
  - Workplace (projects, promotions, performance)
  - Health (exercise, diet, wellness)
  - Technology (software, updates, bugs)
- Vary entity names (not just "Alice" repeatedly)
- Different story structures for same rule sequence
- Multiple problems per rule combination (10-50 variants)

**QUALITY METRICS:**
- Logical validity: Must follow rules exactly
- Naturalness: Readable, not robotic
- Unambiguity: Clear yes/no answer
- Appropriate difficulty: Requires all d steps

### 4) What model should we use?

Paper used: Claude-2
Our baseline: GPT-5

**RECOMMENDATION: Use GPT-5 for generation**

Why GPT-5?
- It's our experimental baseline anyway
- Stronger than Claude-2 (more accurate generation)
- Better instruction following
- We can compare: "GPT-5 generated, GPT-5 solved"

Alternative: Use GPT-4o for generation, test with GPT-5
- Cheaper to generate
- Tests if GPT-5 can solve GPT-4o's problems

**DECISION NEEDED: GPT-5 or GPT-4o for generation?**

### 5) What prompts were used in experiments?

Two types - DIFFERENT purposes:

**A) GENERATION PROMPTS (what they used to CREATE dataset):**

Paper describes but doesn't show exact text:
- Rule definitions with P,Q,R notation
- "Create real-life story replacing P,Q,R with entities"
- "Generate across education, finance domains"
- 5 in-context examples per rule combination
- Format: Context + Question → Yes/No

**WE NEED TO RECREATE THESE for d6-d10**

**B) TESTING PROMPTS (what we use to EVALUATE LLMs):**

From our codebase:
- zero_shot_cot_system.txt: "Expert in logical reasoning..."
- zero_shot_cot_user.txt: "Given context...perform step-by-step"
- lean prompts: For Lean verification
- two_stage prompts: For our two-stage approach

**THESE STAY THE SAME - already working**

## Concrete Generation Plan

### STEP 1: DEFINE RULE SEQUENCES (Manual, 1 day)

Pick valid 6-10 rule combinations:
- d6: 10 combinations (e.g., HS_MP_MP_DS_MT_MP)
- d7: 8 combinations
- d8: 6 combinations
- d9: 4 combinations
- d10: 3 combinations

Criteria:
- Rules must chain properly (output→input)
- Avoid infinite loops
- Human-intuitive
- Mix of different rule types

### STEP 2: CREATE GENERATION PROMPTS (1 day)

For each rule sequence, create prompt:

```
You are creating a logical reasoning problem.

Rules to use (in order):
1. [Rule 1 definition]: If P then Q
2. [Rule 2 definition]: If Q then R
...
6. [Rule 6 definition]: If U then V

Task:
- Create a real-life story using these 6 rules in sequence
- Replace P,Q,R,S,T,U,V with real entities/events
- Make it natural and readable
- Ensure answering requires all 6 steps

Domain: [Education/Finance/Workplace/etc.]

Format:
Context: [Your story with all rules embedded]
Question: [Yes/no question requiring 6-step reasoning]
Answer: [Yes or No]

Examples: [Show 5 similar problems from d5]
```

### STEP 3: GENERATE WITH GPT-5 (2-3 days)
- 10-50 problems per rule sequence
- Vary domains for each
- Generate ~250 total (50 per depth d6-d10)

### STEP 4: MANUAL VALIDATION (3-4 days)

For each generated problem:
- Verify logical correctness
- Check all d steps are needed
- Fix grammar/typos
- Ensure unambiguous answer
- Discard invalid ones

Keep ~70% → ~175 high-quality problems

### STEP 5: FORMAT AND SAVE (1 day)

Save in Multi-LogiEval JSON format:
- d6_Data/fol/[rule_sequence].json
- d7_Data/fol/[rule_sequence].json
- etc.

### STEP 6: TEST BASELINE (1 day)

Run experiments:
- Zero-shot CoT on d6-d10
- Lean verification on d6-d10
- Two-stage on d6-d10
- Compare accuracy: d5 vs d6 vs d10

## Total Time Estimate: 9-12 days

## Next Steps

1. Review and approve this plan
2. Decide: GPT-5 or GPT-4o for generation?
3. Start with Step 1 (defining rule sequences)
