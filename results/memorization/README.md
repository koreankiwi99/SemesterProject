# Memorization Test Results

## Test Types

### 1. Baseline Test (`baseline_test/`)
- Tests with premises removed (keep_none mode)
- If model still answers correctly without premises → possible memorization

### 2. Omission Test (`omission_test/`)
- `full`: Normal evaluation with all premises
- `keep_none`: All premises removed
- Compares accuracy drop when premises removed

### 3. Perturbation Test (`perturbation_test/`)
Two perturbation types:
- **remove_premise**: Remove a critical premise needed for the conclusion
- **add_contradiction**: Add a premise that contradicts the conclusion

Verdict logic:
- If answer unchanged after perturbation → POSSIBLE_MEMORIZATION
- If answer changes appropriately → OK (model is reasoning)

### 4. Unknown Expected Test (`unknown_expected_test/`)
- After perturbation, expects "Unknown" as the correct answer
- Tests if model recognizes insufficient/contradictory information

## Key Results Summary

| Test | Possible Memorization Rate |
|------|---------------------------|
| FOLIO remove_premise | 21% |
| FOLIO add_contradiction | 50% |
| Multi-LogiEval remove_premise | 39% |
| Multi-LogiEval add_contradiction | 57% |

**Note**: High memorization rates for "add_contradiction" may indicate the model is robust to noise rather than memorization. Need manual verification.
