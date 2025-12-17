## Depth-10 Prompt Rules

### Combination 1: HS → CD → DS → MP → HS → MP → MP → MP → MP → MP

**Generalized Rule Definition:**
```
Rule 1: [if {P} is true then {Q} is true, and if {Q} is true then {R} is true]
Rule 2: [if {P} is true then {R} is true, and if {S} is true then {T} is true, and either {P} or {S} or both are true]
Rule 3: [if {T} is true then {U} is true]
Rule 4: [if {U} is true then {V} is true, and if {V} is true then {W} is true]
Rule 5: [if {W} is true then {X} is true]
Rule 6: [if {X} is true then {Y} is true]
Rule 7: [if {Y} is true then {Z} is true]
Rule 8: [if {Z} is true then {Ω} is true]
```

**Question format:** `[If R is not true, then is Ω true?]`

### Combination 2: BD → CT → DS → HS → MP → MP → MP → HS → MP → MP

**Generalized Rule Definition:**
```
Rule 1: [if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {P} is true or {S} is not true or both]
Rule 2: [if {Q} is true then {T} is true, and if {T} is true then {U} is true]
Rule 3: [if {U} is true then {V} is true]
Rule 4: [if {V} is true then {W} is true]
Rule 5: [if {W} is true then {X} is true, and if {X} is true then {Y} is true]
Rule 6: [if {Y} is true then {Z} is true]
```

**Question format:** `[If R is true, then is Z true?]`

### Combination 3: CD → DS → HS → CD → DS → MP → MP → MP → MP → MP

**Generalized Rule Definition:**
```
Rule 1: [if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {P} or {R} or both are true]
Rule 2: [if {S} is true then {T} is true, and if {T} is true then {U} is true]
Rule 3: [if {S} is true then {U} is true, and if {V} is true then {W} is true, and either {S} or {V} or both are true]
Rule 4: [if {W} is true then {X} is true]
Rule 5: [if {X} is true then {Y} is true]
Rule 6: [if {Y} is true then {Z} is true]
Rule 7: [if {Z} is true then {Α} is true]
Rule 8: [if {Α} is true then {Β} is true]
```

**Question format:** `[If Q is not true and U is not true, then is Β true?]`

### Combination 4: HS → MT → DS → BD → CT → DS → MP → MP → MP → MP

**Generalized Rule Definition:**
```
Rule 1: [if {P} is true then {Q} is true, and if {Q} is true then {R} is true]
Rule 2: [either {P} or {S} or both are true]
Rule 3: [if {S} is true then {T} is true, and if {U} is true then {V} is true, and either {S} is true or {V} is not true or both]
Rule 4: [if {T} is true then {W} is true]
Rule 5: [if {W} is true then {X} is true]
Rule 6: [if {X} is true then {Y} is true]
Rule 7: [if {Y} is true then {Z} is true]
```

**Question format:** `[If R is not true and U is true, then is Z true?]`

### Combination 5: DD → DS → HS → MT → DS → MP → MP → HS → MP → MP

**Generalized Rule Definition:**
```
Rule 1: [if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {Q} is not true or {S} is not true or both]
Rule 2: [if {T} is true then {U} is true, and if {U} is true then {V} is true]
Rule 3: [either {T} or {W} or both are true]
Rule 4: [if {W} is true then {X} is true]
Rule 5: [if {X} is true then {Y} is true]
Rule 6: [if {Y} is true then {Z} is true, and if {Z} is true then {Α} is true]
Rule 7: [if {Α} is true then {Β} is true]
```

**Question format:** `[If P is true and V is not true, then is Β true?]`

## Key Patterns:

1. **Complex rules (CD, BD, DD)** are expressed as compound statements with multiple conditions
2. **Simple chains (HS, MP sequences)** can be combined into single rules to reduce premise count
3. **DS steps** are often implicit, triggered by negations in the question
4. **CT (Commutation)** is handled implicitly in the natural language flow
5. **Extended MP chains** at the end allow reaching deeper conclusions (d10 vs d7)

## Variable Naming Convention

For depth-10, we use:
- Standard letters: P, Q, R, S, T, U, V, W, X, Y, Z (11 variables)
- Greek letters for extended conclusions: Α (Alpha), Β (Beta), Ω (Omega)

## Complexity Comparison

| Depth | Inference Steps | Variables | Premises in Story |
|-------|-----------------|-----------|-------------------|
| d5    | 5               | 5-6       | 4-6               |
| d7    | 7               | 8-9       | 7-9               |
| d10   | 10              | 10-13     | 8-12              |
