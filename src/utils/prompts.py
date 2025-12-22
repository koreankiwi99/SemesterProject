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


def format_user_prompt(case: dict, dataset: str) -> str:
    """Format user prompt based on dataset type."""
    if dataset == "folio":
        premises = case.get('premises', '')
        conclusion = case.get('conclusion', '')
        return f"Textual context: {premises}\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? {conclusion}"
    else:  # multilogieval
        context = case.get('context', '')
        question = case.get('question', '')
        return f"Context: {context}\n\nQuestion: {question}"


def get_answer_format(dataset: str) -> str:
    """Get answer format for dataset (true_false or yes_no)."""
    return DATASET_CONFIG[dataset]["answer_format"]


def get_answer_format_str(dataset: str) -> str:
    """Get answer format string for dataset (e.g., 'True/False/Unknown')."""
    return DATASET_CONFIG[dataset]["answer_format_str"]
