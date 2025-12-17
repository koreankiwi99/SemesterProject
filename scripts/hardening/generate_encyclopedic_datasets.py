#!/usr/bin/env python3
"""Generate hardened datasets with encyclopedic noise (k=1,2,4)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from hardening.encyclopedic_noise import WikipediaNoiseGenerator, harden_folio_dataset, harden_multilogi_dataset


def main():
    print("=" * 70)
    print("Encyclopedic Noise Dataset Generation")
    print("=" * 70)
    
    # Initialize Wikipedia generator with cache
    cache_file = "data/wikipedia_abstracts_10k.json"
    print(f"\nInitializing Wikipedia generator (cache: {cache_file})")
    generator = WikipediaNoiseGenerator(cache_file=cache_file)
    
    # Fetch/load 10K abstracts
    print("\nFetching Wikipedia abstracts...")
    abstracts = generator.get_abstracts(count=10000)
    print(f"✓ Using {len(abstracts)} Wikipedia abstract sentences\n")
    
    # Generate FOLIO datasets for k=1,2,4
    print("=" * 70)
    print("Hardening FOLIO Dataset")
    print("=" * 70)
    
    folio_input = "data/folio_original/folio-validation.json"
    
    for k in [1, 2, 4]:
        print(f"\n--- k={k} ---")
        output_file = f"data/folio_hardened/encyclopedic/k{k}/folio-validation-encyc-k{k}.json"
        harden_folio_dataset(
            input_file=folio_input,
            output_file=output_file,
            abstracts=abstracts,
            k=k,
            seed=42
        )
    
    # Generate Multi-LogiEval datasets for k=1,2,4
    print("\n" + "=" * 70)
    print("Hardening Multi-LogiEval Dataset")
    print("=" * 70)
    
    multilogi_input = "data/multi_logi_original"
    
    for k in [1, 2, 4]:
        print(f"\n--- k={k} ---")
        output_dir = f"data/multi_logi_hardened/encyclopedic/k{k}"
        harden_multilogi_dataset(
            input_dir=multilogi_input,
            output_dir=output_dir,
            abstracts=abstracts,
            k=k,
            seed=42
        )
    
    print("\n" + "=" * 70)
    print("✓ All encyclopedic datasets generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
