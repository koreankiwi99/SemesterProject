# Reverse Curriculum Learning for Multi-Step Logical Reasoning with Lean 4

## Project Overview

This project applies reverse curriculum learning to logical reasoning training, starting with complex multi-step problems and progressively simplifying, using Lean 4 verification as objective gating criteria. This approach addresses the fundamental problem of performance degradation in LLMs as reasoning depth increases.

## Key Innovation

Unlike traditional curriculum learning that starts simple and increases complexity, we train models in reverse:
- Begin with depth-5 logical proofs near the conclusion
- Progressively move backward to earlier reasoning steps
- Use Lean 4's formal verification to gate progression between curriculum stages

This prevents models from learning surface patterns on simple cases that fail when reasoning complexity increases.

## Research Context

### The Problem
Recent benchmarks (Multi-LogiEval, ProofWriter) reveal that LLMs show dramatic performance degradation as reasoning depth increases. Models that achieve 90%+ accuracy on single-step logic problems often drop below 50% on 5-step problems.

### Why Reverse Curriculum?
- **R³ (Xi et al., 2024)**: Demonstrated that reverse curriculum reinforcement learning "progressively slides the start state of reasoning from a demonstration's end to its beginning, facilitating easier model exploration at all stages"
- **Florensa et al. (2017)**: Showed success with reverse curriculum in robotics, where agents learn to reach goals from increasingly distant starting points

### Why Lean 4?
- Provides binary verification for logical validity (unlike subjective mathematical reasoning)
- LeanReasoner (Jiang et al., 2024) already demonstrated successful formalization of ProofWriter problems
- Enables automatic curriculum progression without human annotation
  
## Expected Outcomes

1. **Reduced Depth Degradation**: Models maintain higher accuracy as reasoning depth increases
2. **Verifiable Reasoning**: Every reasoning step can be formally verified
3. **Robust Generalization**: Better performance on out-of-distribution logical problems

## Implementation Status

Currently in the proposal stage for EPFL NLP Lab semester project.

## References

- Xi, Z., et al. (2024). Training large language models for reasoning through reverse curriculum reinforcement learning. ICML.
- Jiang, D., et al. (2024). LeanReasoner: Boosting complex logical reasoning with Lean. NAACL.
- Patel, N., et al. (2024). Multi-LogiEval: Towards evaluating multi-step logical reasoning ability of large language models.
- Florensa, C., et al. (2017). Reverse curriculum generation for reinforcement learning. CoRL.
