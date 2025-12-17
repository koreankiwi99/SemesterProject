#!/usr/bin/env python3
"""
Analyze verified-but-wrong cases using OpenAI API for error classification.
Works for both FOLIO and Multi-LogiEval datasets.
"""

import json
import pandas as pd
import os
import sys
import time
from openai import OpenAI

ANALYSIS_PROMPT = """Analyze this Lean 4 formal verification error.

**Premises**: {premises}

**Conclusion**: {conclusion}

**Ground Truth**: {ground_truth}
**Model Prediction**: {prediction} (WRONG - verified by Lean but produces wrong answer)

**Lean Code**:
```lean
{lean_code}
```

Classify the root cause into ONE category:
1. AXIOMATIZES_CONCLUSION - Directly axiomatizes the conclusion or its negation
2. AXIOMATIZES_CONTRADICTION - Axiomatizes statements that contradict premises
3. AXIOMATIZES_UNMENTIONED - Axiomatizes facts about entities NOT mentioned in premises
4. INCORRECT_FORMALIZATION - Formalizes premises incorrectly in Lean
5. REASONING_FAILURE - Has correct axioms but fails to derive the conclusion
6. OTHER - Other error types

Return ONLY a JSON object:
{{
  "problematic_lines": "line numbers or N/A",
  "root_cause_category": "ERROR_TYPE",
  "error_description": "brief explanation (1-2 sentences)",
  "specific_axiom": "the problematic axiom code or N/A"
}}
"""

def analyze_case(case, client, dataset_type='folio'):
    """Analyze a single case using OpenAI API."""

    # Extract fields based on dataset type
    is_twostage = 'twostage' in dataset_type

    if 'folio' in dataset_type:
        premises = case.get('premises', 'N/A')
        conclusion = case.get('conclusion', 'N/A')
        example_id = case.get('example_id', 'N/A')
    else:  # multilogieval
        premises = case.get('context', 'N/A')
        conclusion = case.get('question', 'N/A')
        example_id = f"{case.get('logic_type', '')}_{case.get('depth_dir', '')}"

    # Get lean code
    if is_twostage and 'all_cycles' in case and case['all_cycles']:
        lean_code = case['all_cycles'][-1].get('lean_code', 'N/A')
    else:
        lean_code = case.get('lean_code', 'N/A')

    prompt = ANALYSIS_PROMPT.format(
        premises=premises,
        conclusion=conclusion,
        ground_truth=case.get('ground_truth'),
        prediction=case.get('prediction'),
        lean_code=lean_code
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )

        response_text = response.choices[0].message.content

        # Extract JSON
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result

        return {
            "problematic_lines": "N/A",
            "root_cause_category": "OTHER",
            "error_description": response_text[:200],
            "specific_axiom": "N/A"
        }

    except Exception as e:
        print(f"Error analyzing case {example_id}: {e}")
        return {
            "problematic_lines": "N/A",
            "root_cause_category": "OTHER",
            "error_description": str(e)[:200],
            "specific_axiom": "N/A"
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <folio|folio_twostage|folio_improved|folio_twostage_improved|multilogieval|multilogieval_twostage|multilogieval_d5|multilogieval_d5_twostage>")
        sys.exit(1)

    dataset = sys.argv[1].lower()

    if dataset == 'folio':
        results_path = 'results/folio/lean/all_results.json'
        output_path = 'results/folio/lean/error_root_cause_analysis.csv'
    elif dataset == 'folio_twostage':
        results_path = 'results/folio/two_stage/all_results.json'
        output_path = 'results/folio/two_stage/error_root_cause_analysis.csv'
    elif dataset == 'folio_improved':
        results_path = 'results/folio/lean/lean_improved_20251122_205931/all_results.json'
        output_path = 'results/folio/lean/lean_improved_20251122_205931/error_root_cause_analysis.csv'
    elif dataset == 'folio_twostage_improved':
        results_path = 'results/folio/two_stage/two_stage_lean_improved_20251122_211058/all_results.json'
        output_path = 'results/folio/two_stage/two_stage_lean_improved_20251122_211058/error_root_cause_analysis.csv'
    elif dataset == 'multilogieval':
        results_path = 'results/multilogieval/all_depths/lean/all_results.json'
        output_path = 'results/multilogieval/all_depths/lean/error_root_cause_analysis.csv'
    elif dataset == 'multilogieval_twostage':
        results_path = 'results/multilogieval/all_depths/two_stage/all_results.json'
        output_path = 'results/multilogieval/all_depths/two_stage/error_root_cause_analysis.csv'
    elif dataset == 'multilogieval_d5':
        results_path = 'results/multilogieval/d5_only/lean/all_results.json'
        output_path = 'results/multilogieval/d5_only/lean/error_root_cause_analysis.csv'
    elif dataset == 'multilogieval_d5_twostage':
        results_path = 'results/multilogieval/d5_only/two_stage/all_results.json'
        output_path = 'results/multilogieval/d5_only/two_stage/error_root_cause_analysis.csv'
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    # Initialize OpenAI client
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load results
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Get verified-but-wrong cases
    verified_wrong = []
    is_twostage = 'twostage' in dataset

    for r in results:
        if r is None:
            continue

        if is_twostage:
            # For two-stage, check the final cycle's lean_success
            if 'all_cycles' in r and r['all_cycles']:
                final_cycle = r['all_cycles'][-1]
                if final_cycle.get('lean_success', False) and not r.get('correct', True):
                    verified_wrong.append(r)
        else:
            # For simple lean
            lean_ver = r.get('lean_verification')
            if lean_ver is not None and lean_ver.get('success', False) and not r.get('correct', True):
                verified_wrong.append(r)

    print(f"Found {len(verified_wrong)} verified-but-wrong cases")

    if len(verified_wrong) == 0:
        print("No cases to analyze!")
        return

    # Analyze each case
    analyses = []
    for i, case in enumerate(verified_wrong):
        print(f"Analyzing case {i+1}/{len(verified_wrong)}...", end=' ')

        analysis = analyze_case(case, client, dataset)

        if 'folio' in dataset:
            row = {
                'Example_ID': case.get('example_id'),
                'Ground_Truth': case.get('ground_truth'),
                'Prediction': case.get('prediction'),
                'Pattern': f"{case.get('ground_truth')} → {case.get('prediction')}",
                'Premises': case.get('premises', 'N/A'),
                'Conclusion': case.get('conclusion', 'N/A'),
                'Problematic_Lines': analysis['problematic_lines'],
                'Root_Cause_Category': analysis['root_cause_category'],
                'Error_Description': analysis['error_description'],
                'Specific_Axiom': analysis['specific_axiom']
            }
        else:  # multilogieval
            row = {
                'Logic_Type': case.get('logic_type'),
                'Depth': case.get('depth_dir'),
                'Ground_Truth': case.get('ground_truth'),
                'Prediction': case.get('prediction'),
                'Pattern': f"{case.get('ground_truth')} → {case.get('prediction')}",
                'Context': case.get('context', 'N/A')[:200],
                'Question': case.get('question', 'N/A')[:100],
                'Problematic_Lines': analysis['problematic_lines'],
                'Root_Cause_Category': analysis['root_cause_category'],
                'Error_Description': analysis['error_description'],
                'Specific_Axiom': analysis['specific_axiom']
            }

        analyses.append(row)
        print(f"✓ {analysis['root_cause_category']}")

        time.sleep(1)  # Rate limiting

    # Save to CSV
    df = pd.DataFrame(analyses)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"Saved {len(df)} analyses to {output_path}")
    print(f"{'='*70}")
    print(f"\nError Type Distribution:")
    print(df['Root_Cause_Category'].value_counts())
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
