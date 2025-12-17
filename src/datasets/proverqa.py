"""ProverQA dataset loading utilities."""

import json
from pathlib import Path


def load_proverqa(file_path):
    """Load ProverQA dataset from JSON file.

    Args:
        file_path: Path to the ProverQA JSON file (easy.json, medium.json, or hard.json)

    Returns:
        list: List of question dictionaries with standardized keys
    """
    file_path = Path(file_path)
    difficulty = file_path.stem  # 'easy', 'medium', or 'hard'

    print(f"Loading ProverQA from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add difficulty metadata and standardize keys
    for sample in data:
        sample['difficulty'] = difficulty

    print(f"Loaded {len(data)} questions from ProverQA {difficulty}")

    return data


def load_proverqa_all(data_dir):
    """Load all ProverQA difficulty levels.

    Args:
        data_dir: Path to directory containing easy.json, medium.json, hard.json

    Returns:
        dict: Dictionary with keys 'easy', 'medium', 'hard' containing question lists
    """
    data_path = Path(data_dir)
    result = {}

    for difficulty in ['easy', 'medium', 'hard']:
        file_path = data_path / f'{difficulty}.json'
        if file_path.exists():
            result[difficulty] = load_proverqa(file_path)
        else:
            print(f"Warning: {file_path} not found, skipping...")

    total = sum(len(v) for v in result.values())
    print(f"\nTotal ProverQA questions loaded: {total}")

    return result


def build_proverqa_prompt(sample, system_template, user_template):
    """Build prompt for a single ProverQA sample using templates.

    Args:
        sample: Question sample dictionary
        system_template: System prompt template
        user_template: User prompt template with placeholders

    Returns:
        tuple: (system_msg, user_prompt)
    """
    system_msg = system_template

    # Build options string
    options_str = '\n'.join(sample['options'])

    # Replace placeholders in user prompt
    user_prompt = user_template.replace("{context}", sample['context'])
    user_prompt = user_prompt.replace("{question}", sample['question'])
    user_prompt = user_prompt.replace("{options}", options_str)

    return system_msg, user_prompt
