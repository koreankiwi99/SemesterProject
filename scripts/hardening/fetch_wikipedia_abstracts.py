#!/usr/bin/env python3
"""
Fetch Wikipedia abstracts for encyclopedic noise generation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hardening.encyclopedic_noise import WikipediaNoiseGenerator

def main():
    cache_file = "data/wikipedia_abstracts_10k.json"
    count = 10000

    print(f"Fetching {count} Wikipedia abstracts...")
    print(f"Cache file: {cache_file}\n")

    generator = WikipediaNoiseGenerator(cache_file=cache_file)
    abstracts = generator.fetch_random_abstracts(count=count)

    print(f"\n✓ Successfully fetched and cached {len(abstracts)} Wikipedia abstracts")
    print(f"✓ Cache saved to: {cache_file}")

if __name__ == '__main__':
    main()
