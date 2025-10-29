"""Answer extraction and normalization utilities."""

import re


def normalize_answer(answer, answer_format="yes_no"):
    """Normalize answer to standard format.

    Args:
        answer: The answer to normalize
        answer_format: Either "yes_no" for Multi-LogiEval or "true_false" for FOLIO

    Returns:
        str: Normalized answer
    """
    if not answer:
        return 'Unknown'

    low = answer.lower().strip()

    if answer_format == "yes_no":
        # Multi-LogiEval uses Yes/No format
        if low in ['yes', 'y', 'true', 't', '1']:
            return 'Yes'
        elif low in ['no', 'n', 'false', 'f', '0']:
            return 'No'
    elif answer_format == "true_false":
        # FOLIO uses True/False/Unknown format
        if low in ['true', 't', 'yes', 'y']:
            return 'True'
        elif low in ['false', 'f', 'no', 'n']:
            return 'False'
        elif low in ['unknown', 'uncertain', 'u']:
            return 'Unknown'

    return 'Unknown'


def parse_multilogieval_answer(response):
    """Extract Yes/No answer from Multi-LogiEval model response.

    Args:
        response: The model's text response

    Returns:
        str: Extracted answer ('Yes', 'No', or 'Unknown')
    """
    # Look for explicit ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*(Yes|No|Unknown)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).capitalize()

    # Fallback: Look for explicit Answer: pattern
    answer_match = re.search(r'Answer:\s*(Yes|No)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).capitalize()

    # Fallback: Look for standalone Yes/No
    yes_no_match = re.findall(r'\b(Yes|No)\b', response, re.IGNORECASE)
    if yes_no_match:
        # Return the last occurrence
        return yes_no_match[-1].capitalize()

    return 'Unknown'


def parse_folio_answer(response):
    """Extract True/False/Unknown answer from FOLIO model response.

    Args:
        response: The model's text response

    Returns:
        str: Extracted answer ('True', 'False', or 'Unknown')
    """
    # Look for ANSWER: format first
    answer_match = re.search(r'ANSWER:\s*(True|False|Unknown)', response, re.IGNORECASE)
    if answer_match:
        return normalize_answer(answer_match.group(1), answer_format="true_false")

    # Fallback: look for last occurrence of True/False/Unknown
    all_answers = re.findall(r'\b(True|False|Unknown)\b', response, re.IGNORECASE)
    if all_answers:
        return normalize_answer(all_answers[-1], answer_format="true_false")

    return 'Unknown'


def parse_folio_multiple_answers(response, num_questions):
    """Extract multiple answers from FOLIO model response for grouped questions.

    Args:
        response: The model's text response
        num_questions: Expected number of answers

    Returns:
        list: List of extracted answers
    """
    # Look for the structured ANSWERS: section first
    answers_match = re.search(r'ANSWERS:\s*\n(.*?)(?:\n\n|\n(?=[A-Za-z])|$)', response, re.DOTALL | re.IGNORECASE)
    if answers_match:
        answers_text = answers_match.group(1)
        numbered_answers = re.findall(r'(\d+):\s*(True|False|Unknown)', answers_text, re.IGNORECASE)
        if len(numbered_answers) >= num_questions:
            numbered_answers.sort(key=lambda x: int(x[0]))
            return [normalize_answer(match[1], answer_format="true_false") for match in numbered_answers[:num_questions]]

    # Fallback 1: Look for consecutive True/False/Unknown lines
    consecutive_answers = []
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if re.match(r'^(True|False|Unknown)$', line, re.IGNORECASE):
            consecutive_answers.append(line)
            if len(consecutive_answers) == num_questions:
                return [normalize_answer(a, answer_format="true_false") for a in consecutive_answers]
        else:
            if consecutive_answers and len(consecutive_answers) < num_questions:
                consecutive_answers = []

    # Fallback 2: Extract any True/False/Unknown in order
    all_answers = re.findall(r'\b(True|False|Unknown)\b', response, re.IGNORECASE)
    if len(all_answers) >= num_questions:
        return [normalize_answer(a, answer_format="true_false") for a in all_answers[:num_questions]]

    # Pad with Unknown if we don't have enough answers
    result = [normalize_answer(a, answer_format="true_false") for a in all_answers]
    while len(result) < num_questions:
        result.append('Unknown')

    return result[:num_questions]
