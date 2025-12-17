#!/usr/bin/env python3
"""
Parse d7 raw text files into Multi-LogiEval JSON format.
"""

import json
import re
from pathlib import Path
import argparse


def parse_d7_raw_file(filepath: Path) -> list[dict]:
    """Parse a single d7 raw file into list of samples."""
    content = filepath.read_text()

    # Split by double newlines to separate blocks
    blocks = re.split(r'\n\n+', content.strip())

    samples = []
    current_vars = {}
    current_context = None

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Check if it's variable definitions (starts with {P})
        if block.startswith('{P}'):
            current_vars = {}
            for line in block.split('\n'):
                var_match = re.match(r'\{(\w+)\}:\s*(.+)', line)
                if var_match:
                    current_vars[var_match.group(1)] = var_match.group(2).strip()
            continue

        # Check if it's context
        if block.startswith('Context:'):
            current_context = block[8:].strip()
            continue

        # Check if it's question
        if block.startswith('Question:'):
            current_question = block[9:].strip()

            # Save the complete sample
            if current_context and current_question:
                samples.append({
                    'context': current_context,
                    'question': current_question,
                    'variables': current_vars.copy(),
                    'answer': 'Unknown'  # To be filled by bidirectional verification
                })

    return samples


def parse_all_d7_files(input_dir: Path, output_path: Path):
    """Parse all d7 raw files and save as JSON."""
    all_samples = []

    for raw_file in sorted(input_dir.glob('combination_*_d7_raw.txt')):
        # Extract combination number
        match = re.search(r'combination_(\d+)', raw_file.name)
        if not match:
            continue
        combo_num = int(match.group(1))

        samples = parse_d7_raw_file(raw_file)

        for i, sample in enumerate(samples):
            sample['id'] = f'combo{combo_num}_q{i}'
            sample['combination'] = combo_num
            sample['logic_type'] = 'pl'
            sample['depth'] = 7
            sample['depth_dir'] = 'd7_Data'
            sample['rule'] = f'combination_{combo_num}'
            all_samples.append(sample)

        print(f"Parsed {raw_file.name}: {len(samples)} samples")

    # Save as JSON in Multi-LogiEval format
    output_data = {
        'logic_type': 'pl',
        'depth': 7,
        'rule': 'd7_combinations',
        'samples': all_samples
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(all_samples)} samples to {output_path}")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Parse d7 raw files to JSON')
    parser.add_argument('--input_dir', type=str,
                        default='data/generated/multilogi',
                        help='Directory containing d7 raw files')
    parser.add_argument('--output', type=str,
                        default='data/generated/d7_Data/pl/d7_all.json',
                        help='Output JSON file path')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    parse_all_d7_files(input_dir, output_path)


if __name__ == '__main__':
    main()
