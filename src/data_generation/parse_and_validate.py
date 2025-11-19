"""
Parse and validate generated Multi-LogiEval samples using GPT-4o.
Checks structure, logical correctness, and diversity.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from openai import OpenAI

COMBINATION_RULES = {
    1: {
        "name": "HS_CD_DS_MP_HS_MP_MP",
        "rules": [
            "Rule 1: if {P} is true then {Q} is true, and if {Q} is true then {R} is true",
            "Rule 2: if {P} is true then {R} is true, and if {S} is true then {T} is true, and either {P} or {S} or both are true",
            "Rule 3: if {T} is true then {U} is true",
            "Rule 4: if {U} is true then {V} is true, and if {V} is true then {W} is true",
            "Rule 5: if {W} is true then {X} is true"
        ],
        "question_format": "If R is not true, then is X true?",
        "expected_vars": 9
    },
    2: {
        "name": "BD_CT_DS_HS_MP_MP_MP",
        "rules": [
            "Rule 1: if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {P} is true or {S} is not true or both",
            "Rule 2: if {Q} is true then {T} is true, and if {T} is true then {U} is true",
            "Rule 3: if {U} is true then {V} is true",
            "Rule 4: if {V} is true then {W} is true"
        ],
        "question_format": "If R is true, then is W true?",
        "expected_vars": 8
    },
    3: {
        "name": "CD_DS_HS_CD_DS_MP_MP",
        "rules": [
            "Rule 1: if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {P} or {R} or both are true",
            "Rule 2: if {S} is true then {T} is true, and if {T} is true then {U} is true",
            "Rule 3: if {S} is true then {U} is true, and if {V} is true then {W} is true, and either {S} or {V} or both are true",
            "Rule 4: if {W} is true then {X} is true",
            "Rule 5: if {X} is true then {Y} is true"
        ],
        "question_format": "If Q is not true and U is not true, then is Y true?",
        "expected_vars": 10
    },
    4: {
        "name": "HS_MT_DS_BD_CT_DS_MP",
        "rules": [
            "Rule 1: if {P} is true then {Q} is true, and if {Q} is true then {R} is true",
            "Rule 2: either {P} or {S} or both are true",
            "Rule 3: if {S} is true then {T} is true, and if {U} is true then {V} is true, and either {S} is true or {V} is not true or both",
            "Rule 4: if {T} is true then {W} is true"
        ],
        "question_format": "If R is not true and U is true, then is W true?",
        "expected_vars": 8
    },
    5: {
        "name": "DD_DS_HS_MT_DS_MP_MP",
        "rules": [
            "Rule 1: if {P} is true then {Q} is true, and if {R} is true then {S} is true, and either {Q} is not true or {S} is not true or both",
            "Rule 2: if {T} is true then {U} is true, and if {U} is true then {V} is true",
            "Rule 3: either {T} or {W} or both are true",
            "Rule 4: if {W} is true then {X} is true",
            "Rule 5: if {X} is true then {Y} is true"
        ],
        "question_format": "If P is true and V is not true, then is Y true?",
        "expected_vars": 10
    }
}


def load_validation_prompts() -> Tuple[str, str]:
    """Load validation prompts from files."""
    prompt_dir = Path(__file__).parent.parent.parent / "prompts" / "multilogi" / "validation"

    with open(prompt_dir / "validation_system.txt", 'r') as f:
        system_prompt = f.read().strip()

    with open(prompt_dir / "validation_user.txt", 'r') as f:
        user_prompt_template = f.read().strip()

    return system_prompt, user_prompt_template


def parse_raw_output(raw_text: str) -> Tuple[List[Dict], List[str]]:
    """Parse raw model output into structured samples."""
    samples = []
    errors = []
    lines = raw_text.strip().split('\n')

    current_props = []
    current_context = None
    current_question = None
    sample_num = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\{[A-Z]\}:', line):
            current_props.append(line)
        elif line.startswith('Context:'):
            current_context = line.replace('Context:', '').strip()
        elif line.startswith('Question:'):
            current_question = line.replace('Question:', '').strip()

            if current_context and current_props:
                sample_num += 1
                samples.append({
                    'id': sample_num,
                    'propositions': current_props.copy(),
                    'context': current_context,
                    'question': current_question,
                    'answer': 'NEEDS_VALIDATION'
                })
                current_props = []
                current_context = None
                current_question = None
            else:
                errors.append(f"Sample {sample_num + 1}: Missing propositions or context")

    return samples, errors


def validate_sample_structure(sample: Dict, combination: int) -> List[str]:
    """Validate a single sample's structure."""
    issues = []

    if not sample.get('propositions'):
        issues.append("Missing propositions")
    if not sample.get('context'):
        issues.append("Missing context")
    if not sample.get('question'):
        issues.append("Missing question")

    expected_vars = COMBINATION_RULES[combination]["expected_vars"]
    if len(sample.get('propositions', [])) != expected_vars:
        issues.append(f"Expected {expected_vars} propositions, got {len(sample.get('propositions', []))}")

    context = sample.get('context', '')
    if len(context) < 50:
        issues.append("Context too short")

    question = sample.get('question', '')
    if not question or len(question) < 10:
        issues.append("Question too short or missing")

    return issues


def validate_logical_correctness(sample: Dict, combination: int,
                                 system_prompt: str, user_prompt_template: str,
                                 model: str = "gpt-4o") -> Dict:
    """Use GPT-4o to validate logical correctness of a sample."""
    client = OpenAI()
    rule_info = COMBINATION_RULES[combination]

    user_prompt = user_prompt_template.format(
        rules='\n'.join(rule_info['rules']),
        question_format=rule_info['question_format'],
        propositions='\n'.join(sample['propositions']),
        context=sample['context'],
        question=sample['question']
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


def check_diversity(samples: List[Dict]) -> Dict:
    """Check diversity across samples."""
    entities = []

    for sample in samples:
        if sample.get('propositions'):
            first_prop = sample['propositions'][0]
            match = re.search(r'\{[A-Z]\}:\s*(\w+)', first_prop)
            if match:
                entities.append(match.group(1))

    entity_counts = Counter(entities)
    duplicate_entities = {k: v for k, v in entity_counts.items() if v > 1}

    return {
        'total_samples': len(samples),
        'unique_entities': len(entity_counts),
        'duplicate_entities': duplicate_entities,
        'diversity_score': len(entity_counts) / len(samples) if samples else 0
    }


def generate_report(samples: List[Dict], errors: List[str], combination: int,
                   diversity: Dict, logical_validations: List[Dict]) -> str:
    """Generate validation report."""
    report = []
    report.append("=" * 70)
    report.append(f"VALIDATION REPORT - Combination {combination}")
    report.append(f"Rule: {COMBINATION_RULES[combination]['name']}")
    report.append("=" * 70)
    report.append("")

    report.append("PARSING RESULTS")
    report.append(f"Total samples parsed: {len(samples)}")
    if errors:
        report.append(f"Parsing errors: {len(errors)}")
        for error in errors[:5]:
            report.append(f"  {error}")
    report.append("")

    report.append("STRUCTURE VALIDATION")
    total_issues = 0
    for sample in samples:
        issues = validate_sample_structure(sample, combination)
        if issues:
            total_issues += len(issues)
            report.append(f"Sample {sample['id']}:")
            for issue in issues:
                report.append(f"  {issue}")

    if total_issues == 0:
        report.append(f"All {len(samples)} samples have valid structure")
    report.append("")

    if logical_validations:
        report.append("LOGICAL VALIDATION")
        valid_count = sum(1 for v in logical_validations if v.get('valid'))
        report.append(f"Logically valid: {valid_count}/{len(logical_validations)}")

        for i, validation in enumerate(logical_validations, 1):
            if not validation.get('valid'):
                report.append(f"Sample {i}: Issues found")
                for issue in validation.get('issues', []):
                    report.append(f"  {issue}")
        report.append("")

    report.append("DIVERSITY CHECK")
    report.append(f"Unique entities: {diversity['unique_entities']}/{diversity['total_samples']}")
    report.append(f"Diversity score: {diversity['diversity_score']:.2%}")

    if diversity['duplicate_entities']:
        report.append("Duplicate entities:")
        for entity, count in diversity['duplicate_entities'].items():
            report.append(f"  {entity}: {count} occurrences")
    report.append("")

    valid_count = sum(1 for v in logical_validations if v.get('valid')) if logical_validations else len(samples)
    status = "PASS" if total_issues == 0 and len(errors) == 0 and valid_count == len(samples) else "NEEDS REVIEW"
    report.append(f"STATUS: {status}")
    report.append("=" * 70)

    return '\n'.join(report)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--combination", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-file", default=None)
    parser.add_argument("--validate-logic", action="store_true")

    args = parser.parse_args()

    with open(args.raw_file, 'r') as f:
        raw_text = f.read()

    samples, errors = parse_raw_output(raw_text)
    diversity = check_diversity(samples)

    logical_validations = []
    if args.validate_logic:
        system_prompt, user_prompt_template = load_validation_prompts()
        print(f"Validating {len(samples)} samples with GPT-4o")

        for i, sample in enumerate(samples, 1):
            print(f"Sample {i}/{len(samples)}")
            validation = validate_logical_correctness(sample, args.combination,
                                                     system_prompt, user_prompt_template)
            logical_validations.append(validation)

            if validation.get('correct_answer') and validation['correct_answer'] != 'uncertain':
                sample['answer'] = validation['correct_answer']

    report = generate_report(samples, errors, args.combination, diversity, logical_validations)
    print("\n" + report)

    if args.report_file:
        with open(args.report_file, 'w') as f:
            f.write(report)

    final_samples = [{
        'id': s['id'],
        'context': s['context'],
        'question': s['question'],
        'answer': s['answer']
    } for s in samples]

    final_data = {
        "logic": "pl",
        "rule": COMBINATION_RULES[args.combination]['name'],
        "depth": args.depth,
        "samples": final_samples
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=4)

    print(f"\nSaved to: {output_path}")
    print(f"Total samples: {len(final_samples)}")


if __name__ == "__main__":
    main()
