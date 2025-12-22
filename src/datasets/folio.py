"""FOLIO dataset loading and grouping utilities."""

import json
from collections import defaultdict


def load_folio(file_path):
    """Load FOLIO dataset from JSON file.

    Args:
        file_path: Path to FOLIO JSON file

    Returns:
        list: List of FOLIO examples
    """
    print(f"Loading FOLIO from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # Count unique stories for info
    unique_stories = len(set(ex.get('story_id') for ex in data if ex.get('story_id')))
    print(f"Found {unique_stories} unique stories")

    return data


def group_folio_by_story(folio_data):
    """Group FOLIO examples by story_id.

    Args:
        folio_data: List of FOLIO examples

    Returns:
        dict: Dictionary mapping story_id to list of examples
    """
    grouped = defaultdict(list)
    for example in folio_data:
        story_id = example.get('story_id')
        if story_id is not None:
            grouped[story_id].append(example)
    return dict(grouped)


def load_and_group_folio(file_path):
    """Load FOLIO and group by story_id.

    Args:
        file_path: Path to FOLIO JSON file

    Returns:
        dict: Dictionary mapping story_id to list of examples
    """
    data = load_folio(file_path)
    grouped = group_folio_by_story(data)

    print(f"Found {len(grouped)} unique stories")
    print(f"Questions per story: min={min(len(v) for v in grouped.values())}, "
          f"max={max(len(v) for v in grouped.values())}")

    return grouped


def build_folio_prompt_single(example, system_template, user_template):
    """Build prompt for a single FOLIO question using templates.

    Args:
        example: FOLIO example dictionary
        system_template: System prompt template
        user_template: User prompt template

    Returns:
        tuple: (system_msg, user_prompt)
    """
    system_prompt = system_template

    user_prompt = user_template.replace("{premises}", example['premises'])
    user_prompt = user_prompt.replace("{questions}", example['conclusion'])

    return system_prompt, user_prompt


def build_folio_prompt_grouped(story_examples, system_template, user_template):
    """Build prompt for multiple questions on the same FOLIO story.

    Args:
        story_examples: List of FOLIO examples from the same story
        system_template: System prompt template (should have {num_questions} placeholder)
        user_template: User prompt template (should have {premises}, {questions}, {num_questions})

    Returns:
        tuple: (system_msg, user_prompt)
    """
    premises = story_examples[0]["premises"]
    num_questions = len(story_examples)

    # Build system message
    system_msg = system_template.replace("{num_questions}", str(num_questions))

    # Build questions section
    questions_text = []
    for i, example in enumerate(story_examples, 1):
        questions_text.append(
            f"Question {i}: Based on the above information, is the following statement true, false, or uncertain? "
            f"{example['conclusion']}"
        )

    # Build user prompt
    user_prompt = user_template.replace("{premises}", premises)
    user_prompt = user_prompt.replace("{questions}", "\n\n".join(questions_text))
    user_prompt = user_prompt.replace("{num_questions}", str(num_questions))

    return system_msg, user_prompt
