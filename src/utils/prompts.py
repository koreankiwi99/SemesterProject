"""Prompt loading and formatting utilities."""


# Dataset configurations
DATASET_CONFIG = {
    "folio": {
        "answer_format": "true_false",
        "answer_format_str": "True/False/Uncertain",
        "answer_true": "True",
        "answer_false": "False",
    },
    "multilogieval": {
        "answer_format": "yes_no",
        "answer_format_str": "Yes/No/Uncertain",
        "answer_true": "Yes",
        "answer_false": "No",
    }
}

# Bidirectional condition configurations
BIDIR_CONFIG = {
    "bidir_true": {
        "answer_format": "bidir_true",
        "answer_format_str": "True/Failure",
        "user_prompt_path": "prompts/bidirectional/user.txt",
    },
    "bidir_false": {
        "answer_format": "bidir_false",
        "answer_format_str": "False/Failure",
        "user_prompt_path": "prompts/bidirectional/user.txt",
    }
}


def load_prompt(path: str) -> str:
    """Load prompt from file."""
    with open(path, 'r') as f:
        return f.read().strip()


def format_system_prompt(template: str, dataset: str) -> str:
    """Format system prompt template with dataset-specific answer format."""
    config = DATASET_CONFIG[dataset]
    return template.format(
        answer_format=config["answer_format_str"],
        answer_true=config["answer_true"],
        answer_false=config["answer_false"]
    )


def format_user_prompt(case: dict, dataset: str, condition: str = None) -> str:
    """Format user prompt based on dataset type and condition.

    Args:
        case: The case data
        dataset: "folio" or "multilogieval"
        condition: Optional condition (e.g., "bidir_true", "bidir_false")
    """
    # For bidirectional conditions, use the bidirectional user prompt template
    if condition and condition in BIDIR_CONFIG:
        template_path = BIDIR_CONFIG[condition]["user_prompt_path"]
        template = load_prompt(template_path)
        if dataset == "folio":
            premises = case.get('premises', '')
            conclusion = case.get('conclusion', '')
            return template.format(premises=premises, conclusion=conclusion)
        else:  # multilogieval
            context = case.get('context', '')
            question = case.get('question', '')
            return template.format(premises=context, conclusion=question)

    # Default prompts for standard conditions
    if dataset == "folio":
        premises = case.get('premises', '')
        conclusion = case.get('conclusion', '')
        return f"Textual context: {premises}\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? {conclusion}"
    else:  # multilogieval
        context = case.get('context', '')
        question = case.get('question', '')
        return f"Context: {context}\n\nQuestion: {question}"


def get_answer_format(dataset: str, condition: str = None) -> str:
    """Get answer format for dataset/condition.

    Args:
        dataset: "folio" or "multilogieval"
        condition: Optional condition (e.g., "bidir_true", "bidir_false")

    Returns:
        Answer format string (e.g., "true_false", "bidir_true")
    """
    if condition and condition in BIDIR_CONFIG:
        return BIDIR_CONFIG[condition]["answer_format"]
    return DATASET_CONFIG[dataset]["answer_format"]


def get_answer_format_str(dataset: str, condition: str = None) -> str:
    """Get answer format string for dataset/condition.

    Args:
        dataset: "folio" or "multilogieval"
        condition: Optional condition (e.g., "bidir_true", "bidir_false")

    Returns:
        Human-readable format string (e.g., "True/False/Uncertain", "True/Failure")
    """
    if condition and condition in BIDIR_CONFIG:
        return BIDIR_CONFIG[condition]["answer_format_str"]
    return DATASET_CONFIG[dataset]["answer_format_str"]
