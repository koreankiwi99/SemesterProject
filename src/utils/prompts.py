"""Prompt loading utilities."""


def load_prompt(prompt_file):
    """Load prompt from a text file.

    Args:
        prompt_file: Path to the prompt file

    Returns:
        str: The prompt content

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    try:
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
