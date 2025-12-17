"""
Add encyclopedic noise to Multi-LogiEval problems.

Based on the methodology from:
"Robustness of LLM-based Deductive Reasoning" (2502.04352)

Encyclopedic (E) perturbations are NL sentences expressing factual information
sampled from Wikipedia abstracts. They express real-world information
following pragmatic principles of language.

Key characteristics:
- High formalisation complexity (difficult to formalize in FOL)
- Low semantic similarity to original context
- Represent world information
- Low logical reasoning depth
"""

import json
import random
import argparse
from pathlib import Path


def load_wikipedia_abstracts(file_path: str) -> list[str]:
    """Load Wikipedia abstracts from JSON file."""
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('abstracts', [])


def add_encyclopedic_noise(context: str, abstracts: list[str], k: int, seed: int = None) -> tuple[str, list[str]]:
    """Add k encyclopedic sentences to the beginning of context.

    Following the paper: noise is prepended to avoid breaking inter-sentence
    co-references.

    Args:
        context: Original context string
        abstracts: List of Wikipedia abstracts to sample from
        k: Number of encyclopedic sentences to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (new_context, list of injected sentences)
    """
    if seed is not None:
        random.seed(seed)

    # Sample k abstracts
    selected = random.sample(abstracts, min(k, len(abstracts)))

    # Prepend to context (as per paper methodology)
    noise_text = " ".join(selected)
    new_context = f"{noise_text} {context}"

    return new_context, selected


def process_file(input_path: Path, output_path: Path, abstracts: list[str], k: int, seed: int = None):
    """Process a single Multi-LogiEval file."""
    with open(input_path, encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    new_samples = []
    for i, sample in enumerate(data['samples']):
        # Use sample-specific seed for reproducibility
        sample_seed = seed + i if seed is not None else None

        new_context, injected = add_encyclopedic_noise(
            sample['context'], abstracts, k, seed=sample_seed
        )

        new_sample = {
            **sample,
            'original_context': sample['context'],
            'context': new_context,
            'perturbation': {
                'type': 'encyclopedic',
                'k': k,
                'injected_sentences': injected
            }
        }
        new_samples.append(new_sample)

    output_data = {
        **data,
        'samples': new_samples,
        'hardening': {
            'type': 'encyclopedic',
            'k': k
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return len(new_samples)


def main():
    parser = argparse.ArgumentParser(description='Add encyclopedic noise to Multi-LogiEval')
    parser.add_argument('--input_dir', type=str, default='data/multi_logi_original/data',
                        help='Input Multi-LogiEval data directory')
    parser.add_argument('--output_dir', type=str, default='data/multi_logi_encyclopedic',
                        help='Output directory for hardened data')
    parser.add_argument('--wikipedia_file', type=str, default='data/wikipedia_abstracts_10k.json',
                        help='Path to Wikipedia abstracts JSON file')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 2, 4],
                        help='Number of encyclopedic sentences to add (can specify multiple)')
    parser.add_argument('--depths', type=str, nargs='+', default=['d5_Data'],
                        help='Which depths to process')
    parser.add_argument('--logic_types', type=str, nargs='+', default=['pl', 'fol', 'nm'],
                        help='Which logic types to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Max files to process per logic type (for testing)')

    args = parser.parse_args()

    # Load Wikipedia abstracts
    print(f"Loading Wikipedia abstracts from {args.wikipedia_file}...")
    abstracts = load_wikipedia_abstracts(args.wikipedia_file)
    print(f"Loaded {len(abstracts)} abstracts")

    if len(abstracts) == 0:
        print("Error: No abstracts found. Please provide a valid Wikipedia abstracts file.")
        return

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
                        json_file, output_path, abstracts, k=k_val, seed=args.seed
                    )
                    total_processed += count
                    files_processed += 1

                print(f"  {logic}: processed {len(json_files)} files")

        print(f"k={k_val}: {total_processed} samples from {files_processed} files")
        print(f"Output: {output_dir / f'k{k_val}'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
