"""
Tautological Noise Injection for Dataset Hardening

Based on "Investigating the Robustness of Deductive Reasoning with LLMs"
- Injects k ∈ {1, 2, 4} random tautologies at beginning of context
- Uses 22 hand-written tautologies from paper (Appendix B)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Complete list of 22 tautologies from paper (Appendix B)
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


def inject_tautological_noise(
    text: str,
    num_sentences: int = 2,
    seed: int = None
) -> tuple[str, List[str]]:
    """
    Inject random tautologies at beginning of text

    Args:
        text: Original context/premises text
        num_sentences: Number of tautologies to inject (k ∈ {1, 2, 4})
        seed: Random seed for reproducibility

    Returns:
        (hardened_text, injected_tautologies)
    """
    if seed is not None:
        random.seed(seed)

    # Sample k tautologies randomly
    selected = random.sample(TAUTOLOGIES, num_sentences)

    # Prepend to beginning
    noise_text = " ".join(selected)
    hardened_text = noise_text + " " + text

    return hardened_text, selected


def harden_folio_dataset(
    input_file: Path,
    output_dir: Path,
    k: int = 2,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Apply tautological noise to FOLIO dataset

    Args:
        input_file: Path to FOLIO JSON file
        output_dir: Output directory for hardened dataset
        k: Number of tautology sentences (1, 2, or 4)
        seed: Random seed

    Returns:
        Statistics dict
    """
    # Load data
    with open(input_file) as f:
        data = json.load(f)

    # Harden each sample
    hardened_data = []
    for i, sample in enumerate(data):
        hardened_sample = sample.copy()

        # Inject noise into premises
        hardened_premises, noise = inject_tautological_noise(
            sample['premises'],
            num_sentences=k,
            seed=seed + i  # Different seed per sample
        )

        hardened_sample['premises'] = hardened_premises

        # Add metadata
        hardened_sample['perturbation'] = {
            'type': 'tautological',
            'k': k,
            'injected_sentences': noise
        }

        hardened_data.append(hardened_sample)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}-taut-k{k}.json"

    with open(output_file, 'w') as f:
        json.dump(hardened_data, f, indent=2)

    stats = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'num_samples': len(hardened_data),
        'k': k,
        'seed': seed
    }

    print(f"✓ Generated {len(hardened_data)} samples: {output_file}")
    return stats


def harden_multilogi_dataset(
    input_file: Path,
    output_dir: Path,
    k: int = 2,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Apply tautological noise to Multi-LogiEval dataset

    Args:
        input_file: Path to Multi-LogiEval JSON file
        output_dir: Output directory
        k: Number of tautology sentences
        seed: Random seed

    Returns:
        Statistics dict
    """
    # Load data
    with open(input_file) as f:
        data = json.load(f)

    # Harden samples
    hardened_samples = []
    for i, sample in enumerate(data['samples']):
        hardened_sample = sample.copy()

        # Inject noise into context
        hardened_context, noise = inject_tautological_noise(
            sample['context'],
            num_sentences=k,
            seed=seed + i
        )

        hardened_sample['context'] = hardened_context
        hardened_sample['perturbation'] = {
            'type': 'tautological',
            'k': k,
            'injected_sentences': noise
        }

        hardened_samples.append(hardened_sample)

    # Create output data
    output_data = {
        'logic': data['logic'],
        'rule': data['rule'],
        'depth': data['depth'],
        'hardening': {
            'type': 'tautological',
            'k': k,
            'seed': seed
        },
        'samples': hardened_samples
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / input_file.name

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    stats = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'rule': data['rule'],
        'depth': data['depth'],
        'num_samples': len(hardened_samples),
        'k': k
    }

    return stats


def process_all_folio(base_dir: Path = None, k_values: List[int] = [2]):
    """Process all FOLIO datasets"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    folio_file = base_dir / 'data' / 'folio_original' / 'folio-validation.json'

    if not folio_file.exists():
        print(f"✗ FOLIO file not found: {folio_file}")
        return

    print(f"\n{'='*60}")
    print(f"Processing FOLIO: {folio_file.name}")
    print(f"{'='*60}\n")

    all_stats = []
    for k in k_values:
        output_dir = base_dir / 'data' / 'folio_hardened' / 'tautological' / f'k{k}'
        stats = harden_folio_dataset(folio_file, output_dir, k=k)
        all_stats.append(stats)

    return all_stats


def process_all_multilogi(base_dir: Path = None, k_values: List[int] = [2]):
    """Process Multi-LogiEval d1-d5, all logic types"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    data_dir = base_dir / 'data' / 'multi_logi_original' / 'data'

    if not data_dir.exists():
        print(f"✗ Multi-LogiEval directory not found: {data_dir}")
        return

    depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']
    logic_types = ['pl', 'fol', 'nm']

    print(f"\n{'='*60}")
    print(f"Processing Multi-LogiEval: d1-d5")
    print(f"{'='*60}\n")

    all_stats = []
    total_files = 0

    for depth in depths:
        for logic in logic_types:
            depth_logic_dir = data_dir / depth / logic

            if not depth_logic_dir.exists():
                continue

            # Process all rule files
            rule_files = sorted(depth_logic_dir.glob('*.json'))

            for rule_file in rule_files:
                total_files += 1

                for k in k_values:
                    output_dir = base_dir / 'data' / 'multi_logi_hardened' / 'tautological' / f'k{k}' / depth / logic

                    try:
                        stats = harden_multilogi_dataset(rule_file, output_dir, k=k)
                        all_stats.append(stats)

                        if total_files % 10 == 0:
                            print(f"  Processed {total_files} files...")
                    except Exception as e:
                        print(f"✗ Error processing {rule_file}: {e}")

    print(f"\n✓ Total files processed: {total_files}")
    return all_stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Inject tautological noise into datasets')
    parser.add_argument('--dataset', choices=['folio', 'multilogi', 'both'], default='both',
                       help='Which dataset to process')
    parser.add_argument('--k', type=int, nargs='+', default=[2],
                       help='Number of tautology sentences to inject (e.g., 1 2 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set global seed
    random.seed(args.seed)

    print(f"\n{'#'*60}")
    print(f"# Tautological Noise Injection")
    print(f"# k values: {args.k}")
    print(f"# Random seed: {args.seed}")
    print(f"{'#'*60}")

    if args.dataset in ['folio', 'both']:
        process_all_folio(k_values=args.k)

    if args.dataset in ['multilogi', 'both']:
        process_all_multilogi(k_values=args.k)

    print(f"\n{'='*60}")
    print(f"✓ Hardening complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
