"""Multi-LogiEval dataset loading and sampling utilities."""

import json
import random
from pathlib import Path
from collections import defaultdict


def load_multilogieval_file(file_path, depth_dir):
    """Load a single Multi-LogiEval JSON file.

    Args:
        file_path: Path to the JSON file
        depth_dir: Depth directory name (e.g., 'd1_Data')

    Returns:
        list: List of samples with metadata added
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

    logic_type = data.get('logic', 'unknown')
    rule = data.get('rule', 'unknown')
    depth = data.get('depth', 'unknown')
    samples = data.get('samples', [])

    # Add metadata to each sample
    for sample in samples:
        sample['logic_type'] = logic_type
        sample['rule'] = rule
        sample['depth'] = depth
        sample['depth_dir'] = depth_dir
        sample['source_file'] = str(file_path)

    return samples


def load_and_sample_multilogieval(data_dir, logic_types=None, depths=None,
                                   samples_per_combination=10, seed=42):
    """Load Multi-LogiEval dataset and sample individual questions.

    Args:
        data_dir: Path to Multi-LogiEval data directory
        logic_types: List of logic types to include (default: ['fol', 'nm', 'pl'])
        depths: List of depths to include (default: all d1-d5)
        samples_per_combination: Number of questions to sample per logic type × depth
        seed: Random seed for reproducibility

    Returns:
        list: List of sampled question dictionaries
    """
    data_path = Path(data_dir)

    if logic_types is None:
        logic_types = ['fol', 'nm', 'pl']

    if depths is None:
        depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']

    print(f"Loading Multi-LogiEval from: {data_dir}")
    print(f"Logic types: {logic_types}")
    print(f"Depths: {depths}")
    print(f"Sampling {samples_per_combination} questions per logic type × depth combination")
    print(f"Random seed: {seed}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Collect all samples organized by combination
    samples_by_combination = defaultdict(list)

    for depth_dir in depths:
        depth_path = data_path / depth_dir
        if not depth_path.exists():
            print(f"Warning: {depth_path} does not exist, skipping...")
            continue

        for logic_type in logic_types:
            logic_path = depth_path / logic_type
            if not logic_path.exists():
                print(f"Warning: {logic_path} does not exist, skipping...")
                continue

            json_files = list(logic_path.glob('*.json'))
            print(f"Found {len(json_files)} files in {depth_dir}/{logic_type}")

            # Load all samples from all files for this combination
            combination_key = (logic_type, depth_dir)
            for json_file in json_files:
                samples = load_multilogieval_file(json_file, depth_dir)
                samples_by_combination[combination_key].extend(samples)

    # Sample from each combination
    sampled_questions = []
    print(f"\n{'='*70}")
    print("Sampling questions from each combination:")
    print(f"{'='*70}")

    for (logic_type, depth_dir), samples in sorted(samples_by_combination.items()):
        num_available = len(samples)
        num_to_sample = min(samples_per_combination, num_available)

        sampled = random.sample(samples, num_to_sample)
        sampled_questions.extend(sampled)

        print(f"{logic_type}/{depth_dir}: sampled {num_to_sample} from {num_available} available questions")

    print(f"\n{'='*70}")
    print(f"Total questions sampled: {len(sampled_questions)}")
    print(f"{'='*70}")

    # Print distribution
    distribution = defaultdict(lambda: defaultdict(int))
    for sample in sampled_questions:
        distribution[sample['logic_type']][sample['depth']] += 1

    print("\nDistribution of sampled questions:")
    print(f"{'='*70}")
    for logic_type in sorted(distribution.keys()):
        print(f"\n{logic_type}:")
        for depth in sorted(distribution[logic_type].keys()):
            print(f"  {depth}: {distribution[logic_type][depth]} questions")

    return sampled_questions


def build_multilogieval_prompt(sample, system_template, user_template):
    """Build prompt for a single Multi-LogiEval sample using templates.

    Args:
        sample: Question sample dictionary
        system_template: System prompt template
        user_template: User prompt template

    Returns:
        tuple: (system_msg, user_prompt)
    """
    system_msg = system_template

    # Replace placeholders in user prompt
    user_prompt = user_template.replace("{premises}", sample['context'])
    user_prompt = user_prompt.replace("{questions}", sample['question'])

    return system_msg, user_prompt
