#!/usr/bin/env python3
"""
ProverQA format converter.

Converts ProverQA dataset from A/B/C answer format to True/False/Uncertain format
to match FOLIO and MultiLogiEval conventions.
"""

import argparse
import json
import os


# Answer mapping from ProverQA format to standard format
ANSWER_MAP = {
    'A': 'True',
    'B': 'False',
    'C': 'Uncertain'
}

# Reverse mapping for converting back if needed
REVERSE_MAP = {v: k for k, v in ANSWER_MAP.items()}


def convert_proverqa_to_tfu(data: list) -> list:
    """Convert ProverQA data from A/B/C to True/False/Uncertain format.

    Args:
        data: List of ProverQA question dictionaries

    Returns:
        List of converted question dictionaries
    """
    converted = []
    for item in data:
        new_item = item.copy()
        # Convert answer from A/B/C to True/False/Uncertain
        if item['answer'] in ANSWER_MAP:
            new_item['answer'] = ANSWER_MAP[item['answer']]
        # Update options to standard format
        new_item['options'] = ['True', 'False', 'Uncertain']
        converted.append(new_item)
    return converted


def convert_tfu_to_proverqa(data: list) -> list:
    """Convert data from True/False/Uncertain back to A/B/C format.

    Args:
        data: List of question dictionaries with True/False/Uncertain answers

    Returns:
        List of converted question dictionaries with A/B/C answers
    """
    converted = []
    for item in data:
        new_item = item.copy()
        # Convert answer from True/False/Uncertain to A/B/C
        if item['answer'] in REVERSE_MAP:
            new_item['answer'] = REVERSE_MAP[item['answer']]
        # Update options to ProverQA format
        new_item['options'] = ['A) True', 'B) False', 'C) Uncertain']
        converted.append(new_item)
    return converted


def convert_file(input_path: str, output_path: str = None, to_tfu: bool = True) -> str:
    """Convert a ProverQA JSON file.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (auto-generated if None)
        to_tfu: If True, convert to True/False/Uncertain; if False, convert to A/B/C

    Returns:
        Path to output file
    """
    # Load data
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Convert
    if to_tfu:
        converted = convert_proverqa_to_tfu(data)
        suffix = '_tfu'
    else:
        converted = convert_tfu_to_proverqa(data)
        suffix = '_abc'

    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}{suffix}{ext}"

    # Save
    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert ProverQA dataset between A/B/C and True/False/Uncertain formats'
    )
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (auto-generated if not provided)')
    parser.add_argument('--to-abc', action='store_true',
                        help='Convert to A/B/C format (default is to True/False/Uncertain)')

    args = parser.parse_args()

    to_tfu = not args.to_abc
    output_path = convert_file(args.input, args.output, to_tfu=to_tfu)

    # Load and show stats
    with open(output_path, 'r') as f:
        data = json.load(f)

    print(f"Converted {len(data)} questions")
    print(f"Output: {output_path}")

    # Show answer distribution
    from collections import Counter
    answers = Counter(item['answer'] for item in data)
    print(f"\nAnswer distribution:")
    for ans, count in sorted(answers.items()):
        print(f"  {ans}: {count} ({count/len(data)*100:.1f}%)")


if __name__ == '__main__':
    main()
