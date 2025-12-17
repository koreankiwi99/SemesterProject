"""
Add tautological noise to Multi-LogiEval problems.

Based on the methodology from:
"Robustness of LLM-based Deductive Reasoning" (2502.04352)

Tautological (T) perturbations are easily recognizable correct statements
that may include negations and at most one disjunction or conjunction.
These provide the highest semantic similarity to reasoning contexts but
contain no referential information.

The 22 tautologies are from Appendix B of the paper.
"""

import json
import random
import argparse
from pathlib import Path


# 22 tautologies from the paper (Appendix B)
TAUTOLOGIES = [
    "False is not true.",
    "True is not false.",
    "Not false is true.",
    "Not true is false.",
    "False and true is not true.",
    "False and not true is false.",
    "False and not false is false.",
    "Not true and true is false.",
    "Not true and false is false.",
    "True and false is not true.",
    "True and true is not false.",
    "True and not false is true.",
    "Not false and true is true.",
    "Not false and false is false.",
    "True or not true is true.",
    "True or true is true.",
    "False or not false is true.",
    "False or not true is false.",
    "Not true or false is not true.",
    "Not true or true is true.",
    "Not false or false is true.",
    "Not false or true is not false.",
]


def add_tautological_noise(context: str, k: int, seed: int = None) -> tuple[str, list[str]]:
    """Add k tautological sentences to the beginning of context.

    Following the paper: noise is prepended to avoid breaking inter-sentence
    co-references.

    Args:
        context: Original context string
        k: Number of tautologies to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (new_context, list of injected sentences)
    """
    if seed is not None:
        random.seed(seed)

    # Sample k tautologies (with replacement if k > len(TAUTOLOGIES))
    if k <= len(TAUTOLOGIES):
        selected = random.sample(TAUTOLOGIES, k)
    else:
        selected = random.choices(TAUTOLOGIES, k=k)

    # Prepend to context (as per paper methodology)
    noise_text = " ".join(selected)
    new_context = f"{noise_text} {context}"

    return new_context, selected


def process_file(input_path: Path, output_path: Path, k: int, seed: int = None):
    """Process a single Multi-LogiEval file."""
    with open(input_path, encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    new_samples = []
    for i, sample in enumerate(data['samples']):
        # Use sample-specific seed for reproducibility
        sample_seed = seed + i if seed is not None else None

        new_context, injected = add_tautological_noise(
            sample['context'], k, seed=sample_seed
        )

        new_sample = {
            **sample,
            'original_context': sample['context'],
            'context': new_context,
            'perturbation': {
                'type': 'tautological',
                'k': k,
                'injected_sentences': injected
            }
        }
        new_samples.append(new_sample)

    output_data = {
        **data,
        'samples': new_samples,
        'hardening': {
            'type': 'tautological',
            'k': k
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return len(new_samples)


def main():
    parser = argparse.ArgumentParser(description='Add tautological noise to Multi-LogiEval')
    parser.add_argument('--input_dir', type=str, default='data/multi_logi_original/data',
                        help='Input Multi-LogiEval data directory')
    parser.add_argument('--output_dir', type=str, default='data/multi_logi_tautological',
                        help='Output directory for hardened data')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 2, 4],
                        help='Number of tautologies to add (can specify multiple)')
    parser.add_argument('--depths', type=str, nargs='+', default=['d5_Data'],
                        help='Which depths to process')
    parser.add_argument('--logic_types', type=str, nargs='+', default=['pl', 'fol', 'nm'],
                        help='Which logic types to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Max files to process per logic type (for testing)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for k_val in args.k:
        print(f"\n=== Processing k={k_val} ===")
        total_processed = 0
        files_processed = 0

        for depth in args.depths:
            for logic in args.logic_types:
                logic_dir = input_dir / depth / logic
                if not logic_dir.exists():
                    print(f"Skipping {logic_dir} (not found)")
                    continue

                json_files = list(logic_dir.glob('*.json'))
                if args.max_files:
                    json_files = json_files[:args.max_files]

                for json_file in json_files:
                    output_path = output_dir / f"k{k_val}" / depth / logic / json_file.name

                    count = process_file(
                        json_file, output_path, k=k_val, seed=args.seed
                    )
                    total_processed += count
                    files_processed += 1

                print(f"  {logic}: processed {len(json_files)} files")

        print(f"k={k_val}: {total_processed} samples from {files_processed} files")
        print(f"Output: {output_dir / f'k{k_val}'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
