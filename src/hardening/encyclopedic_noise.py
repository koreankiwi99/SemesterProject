"""Encyclopedic noise perturbation for robustness testing.

Based on "Investigating the Robustness of Deductive Reasoning with LLMs" methodology.
Adds real-world factual information from Wikipedia to test out-of-domain robustness.
"""

import json
import random
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path


class WikipediaNoiseGenerator:
    """Generates encyclopedic noise from Wikipedia abstracts."""

    def __init__(self, cache_file: Optional[str] = None):
        """Initialize with optional cache file for Wikipedia abstracts."""
        self.cache_file = cache_file
        self.abstracts_cache: List[str] = []

        if cache_file and Path(cache_file).exists():
            self._load_cache()

    def _load_cache(self):
        """Load cached Wikipedia abstracts from file."""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.abstracts_cache = data.get('abstracts', [])
        print(f"Loaded {len(self.abstracts_cache)} Wikipedia abstracts from cache")

    def _save_cache(self):
        """Save Wikipedia abstracts to cache file."""
        if self.cache_file:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({'abstracts': self.abstracts_cache}, f, indent=2)
            print(f"Saved {len(self.abstracts_cache)} Wikipedia abstracts to cache")

    def fetch_random_abstracts(self, count: int = 10000) -> List[str]:
        """Fetch random Wikipedia article abstracts using the API."""
        abstracts = []
        print(f"Fetching {count} random Wikipedia abstracts...")

        api_url = "https://en.wikipedia.org/w/api.php"
        batch_size = 50
        batches = (count + batch_size - 1) // batch_size

        # Add User-Agent header to comply with Wikipedia API requirements
        headers = {
            'User-Agent': 'LogicRobustnessBot/1.0 (Research project; educational use)'
        }

        for i in range(batches):
            try:
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'random',
                    'rnnamespace': 0,
                    'rnlimit': min(batch_size, count - len(abstracts))
                }

                response = requests.get(api_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()

                page_ids = [page['id'] for page in data['query']['random']]

                extract_params = {
                    'action': 'query',
                    'format': 'json',
                    'prop': 'extracts',
                    'exintro': True,
                    'explaintext': True,
                    'pageids': '|'.join(map(str, page_ids))
                }

                extract_response = requests.get(api_url, params=extract_params, headers=headers, timeout=10)
                extract_response.raise_for_status()
                extract_data = extract_response.json()

                for page_id, page_data in extract_data['query']['pages'].items():
                    if 'extract' in page_data and page_data['extract']:
                        text = page_data['extract']
                        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
                        abstracts.extend(sentences)

                if (i + 1) % 10 == 0:
                    print(f"  Fetched {i + 1}/{batches} batches ({len(abstracts)} sentences)")

            except Exception as e:
                print(f"Error fetching batch {i + 1}: {e}")
                continue

        self.abstracts_cache = abstracts[:count]
        if self.cache_file:
            self._save_cache()

        print(f"✓ Fetched {len(self.abstracts_cache)} total sentences")
        return self.abstracts_cache

    def get_abstracts(self, count: int = 10000) -> List[str]:
        """Get Wikipedia abstracts, using cache if available."""
        if len(self.abstracts_cache) >= count:
            return self.abstracts_cache[:count]
        else:
            return self.fetch_random_abstracts(count)


def inject_encyclopedic_noise(
    text: str,
    abstracts: List[str],
    num_sentences: int = 2,
    seed: Optional[int] = None
) -> tuple:
    """Inject encyclopedic noise (Wikipedia facts) into text."""
    if seed is not None:
        random.seed(seed)

    selected = random.sample(abstracts, min(num_sentences, len(abstracts)))
    noise_text = " ".join(selected)
    hardened_text = noise_text + " " + text

    return hardened_text, selected


def harden_folio_dataset(
    input_file: str,
    output_file: str,
    abstracts: List[str],
    k: int = 2,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Harden FOLIO dataset with encyclopedic noise."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {'total_examples': 0, 'hardened_examples': 0, 'k': k}

    for example in data:
        example_seed = seed + example['example_id'] if seed is not None else None
        original_premises = example['premises']
        hardened_premises, selected = inject_encyclopedic_noise(
            original_premises, abstracts, num_sentences=k, seed=example_seed
        )

        example['premises'] = hardened_premises
        example['perturbation'] = {
            'type': 'encyclopedic',
            'k': k,
            'injected_sentences': selected
        }

        stats['total_examples'] += 1
        stats['hardened_examples'] += 1

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Hardened {stats['hardened_examples']} FOLIO examples with k={k}")
    return stats


def harden_multilogi_dataset(
    input_dir: str,
    output_dir: str,
    abstracts: List[str],
    k: int = 2,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Harden Multi-LogiEval dataset with encyclopedic noise."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {'total_files': 0, 'total_examples': 0, 'hardened_examples': 0, 'k': k}

    for json_file in input_path.rglob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Multi-LogiEval files have structure: {"logic": ..., "rule": ..., "samples": [...]}
            if isinstance(data, dict) and 'samples' in data:
                samples = data['samples']
                for example in samples:
                    stats['total_examples'] += 1
                    example_seed = seed + stats['total_examples'] if seed is not None else None

                    original_context = example['context']
                    hardened_context, selected = inject_encyclopedic_noise(
                        original_context, abstracts, num_sentences=k, seed=example_seed
                    )

                    example['context'] = hardened_context
                    example['perturbation'] = {
                        'type': 'encyclopedic',
                        'k': k,
                        'injected_sentences': selected
                    }
                    stats['hardened_examples'] += 1
            else:
                # Fallback for list format
                for example in data:
                    stats['total_examples'] += 1
                    example_seed = seed + stats['total_examples'] if seed is not None else None

                    original_context = example['context']
                    hardened_context, selected = inject_encyclopedic_noise(
                        original_context, abstracts, num_sentences=k, seed=example_seed
                    )

                    example['context'] = hardened_context
                    example['perturbation'] = {
                        'type': 'encyclopedic',
                        'k': k,
                        'injected_sentences': selected
                    }
                    stats['hardened_examples'] += 1

            relative_path = json_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            stats['total_files'] += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"✓ Hardened {stats['hardened_examples']} Multi-LogiEval examples ({stats['total_files']} files)")
    return stats
