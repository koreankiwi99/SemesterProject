#!/usr/bin/env python3
"""
Analyze verified-but-wrong cases using LLM for error classification.

Usage:
    PYTHONPATH=src:$PYTHONPATH python src/analysis/analyze_errors.py \
        --results <path_to_results.json> \
        --prompt <prompt_file> \
        [--output <output_file>] \
        [--condition <condition_name>]

Examples:
    PYTHONPATH=src:$PYTHONPATH python src/analysis/analyze_errors.py \
        --results results/conditions_experiment/folio_full_baseline_20251222_140906/all_results.json \
        --prompt prompts/error-classification/v1.txt \
        --condition baseline
"""

import json
import re
import os
import sys
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def analyze_case(case: dict, client: OpenAI, prompt_template: str, model: str = 'gpt-4o') -> dict:
    """Analyze a single case using LLM."""

    # Extract fields - try both folio and multilogieval formats
    premises = case.get('premises') or case.get('context', 'N/A')
    conclusion = case.get('conclusion') or case.get('question', 'N/A')
    lean_code = case.get('lean_code', 'N/A')

    # Format prompt
    prompt = prompt_template.format(
        premises=premises[:3000],
        conclusion=conclusion,
        ground_truth=case.get('ground_truth'),
        prediction=case.get('prediction'),
        lean_code=lean_code[:4000]
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )

        response_text = response.choices[0].message.content

        # Extract JSON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return {
            "root_cause_category": "OTHER",
            "error_description": response_text[:200],
        }

    except Exception as e:
        return {
            "root_cause_category": "ERROR",
            "error_description": str(e)[:200],
        }


def get_false_negatives(results: list) -> list:
    """Extract false negative cases from results."""
    false_negatives = []

    for r in results:
        if r is None:
            continue

        lean_ver = r.get('lean_verification') or {}
        if lean_ver.get('success', False) and not r.get('correct', True):
            false_negatives.append(r)

    return false_negatives


def get_all_lean_pass(results: list) -> list:
    """Extract all cases where Lean verification passed."""
    cases = []
    for r in results:
        if r is None:
            continue
        lean_ver = r.get('lean_verification') or {}
        if lean_ver.get('success', False):
            cases.append(r)
    return cases


def load_folio_data(folio_path: str = 'data/folio/original/folio-validation.json') -> dict:
    """Load FOLIO data and create lookup by example_id."""
    with open(folio_path, 'r') as f:
        data = json.load(f)
    return {item['example_id']: item for item in data}


def main():
    parser = argparse.ArgumentParser(description='Analyze Lean false negatives')
    parser.add_argument('--results', required=True, help='Path to results JSON file')
    parser.add_argument('--prompt', required=True, help='Path to prompt template file')
    parser.add_argument('--output', help='Output CSV path (auto-generated if not specified)')
    parser.add_argument('--model', default='gpt-4o', help='Model to use for analysis')
    parser.add_argument('--condition', default=None, help='Condition name if results has nested structure')
    parser.add_argument('--folio_data', default='data/folio/original/folio-validation.json',
                        help='Path to FOLIO data for premises/conclusion lookup')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all Lean-pass cases, not just false negatives')

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load prompt template
    print(f"Loading prompt from: {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)

    # Load results
    print(f"Loading results from: {args.results}")
    with open(args.results, 'r') as f:
        content = f.read().strip()

    # Auto-detect JSONL vs JSON format
    if content.startswith('[') or content.startswith('{'):
        # Try JSON first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback to JSONL
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
    else:
        # JSONL format (each line is a JSON object)
        data = [json.loads(line) for line in content.split('\n') if line.strip()]

    # Handle nested structure (conditions experiment) vs flat list
    if args.condition:
        results = data.get(args.condition, [])
        print(f"Using condition: {args.condition}")
    elif isinstance(data, dict) and 'baseline' in data:
        # Auto-detect conditions format
        results = data.get('baseline', [])
        print(f"Auto-detected conditions format, using 'baseline'")
    else:
        results = data

    print(f"Loaded {len(results)} results")

    # Load FOLIO data for premises/conclusion lookup
    print(f"Loading FOLIO data from: {args.folio_data}")
    folio_lookup = load_folio_data(args.folio_data)
    print(f"Loaded {len(folio_lookup)} FOLIO entries")

    # Get cases to analyze
    if args.all:
        cases_to_analyze = get_all_lean_pass(results)
        print(f"Found {len(cases_to_analyze)} Lean-pass cases to analyze (--all mode)")
    else:
        cases_to_analyze = get_false_negatives(results)
        print(f"Found {len(cases_to_analyze)} false negatives")

    # Enrich with FOLIO data
    for case in cases_to_analyze:
        example_id = case.get('example_id')
        if example_id in folio_lookup:
            folio_entry = folio_lookup[example_id]
            case['premises'] = folio_entry.get('premises', '')
            case['conclusion'] = folio_entry.get('conclusion', '')

    if not cases_to_analyze:
        print("No cases to analyze!")
        return

    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        prompt_name = os.path.basename(args.prompt).replace('.txt', '')
        results_name = os.path.basename(os.path.dirname(args.results))
        output_path = f'results/error_analysis/{results_name}_{prompt_name}.csv'

    # Analyze each case
    analyses = []
    for i, case in enumerate(cases_to_analyze):
        print(f"Analyzing {i+1}/{len(cases_to_analyze)}...", end=' ')

        analysis = analyze_case(case, client, prompt_template, args.model)

        # Map field names (v6 uses 'category', older versions use 'root_cause_category')
        category = (analysis.get('root_cause_category') or
                    analysis.get('category', 'OTHER'))
        description = (analysis.get('error_description') or
                       analysis.get('explanation', ''))
        problematic = (analysis.get('problematic_axiom') or
                       analysis.get('problematic_element') or
                       analysis.get('specific_axiom', 'N/A'))

        row = {
            'case_idx': case.get('case_idx'),
            'example_id': case.get('example_id'),
            'story_id': case.get('story_id'),
            'ground_truth': case.get('ground_truth'),
            'prediction': case.get('prediction'),
            'correct': case.get('correct'),
            'pattern': f"{case.get('prediction')} → {case.get('ground_truth')}",
            'num_iterations': case.get('num_iterations'),
            'premises': (case.get('premises') or case.get('context', ''))[:300],
            'conclusion': case.get('conclusion') or case.get('question', ''),
            'root_cause_category': category,
            'error_description': description,
            'problematic_axiom': problematic,
        }

        # Add v6-specific fields if present
        if 'is_gaming' in analysis:
            row['is_gaming'] = analysis.get('is_gaming')
        if 'is_faithful' in analysis:
            row['is_faithful'] = analysis.get('is_faithful')
        if 'is_dataset_issue' in analysis:
            row['is_dataset_issue'] = analysis.get('is_dataset_issue')

        analyses.append(row)
        print(f"✓ {row['root_cause_category']}")

        time.sleep(0.5)  # Rate limiting

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(analyses)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total analyzed: {len(df)}")
    print(f"\nRoot Cause Distribution:")
    print(df['root_cause_category'].value_counts())
    print(f"\nBy Pattern:")
    print(df.groupby('pattern')['root_cause_category'].value_counts())
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
