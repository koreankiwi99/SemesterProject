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


def parse_two_stage_answers(response):
    """Extract both Stage 1 and Stage 2 answers from two-stage model response.

    Args:
        response: The model's text response

    Returns:
        dict: Dictionary with 'stage1_answer' and 'stage2_answer' keys
    """
    result = {
        'stage1_answer': 'Unknown',
        'stage2_answer': 'Unknown'
    }

    # Look for STAGE 1 ANSWER:
    stage1_match = re.search(r'STAGE\s*1\s*ANSWER:\s*(Yes|No|Unknown)', response, re.IGNORECASE)
    if stage1_match:
        result['stage1_answer'] = stage1_match.group(1).capitalize()

    # Look for STAGE 2 ANSWER:
    stage2_match = re.search(r'STAGE\s*2\s*ANSWER:\s*(Yes|No|Unknown)', response, re.IGNORECASE)
    if stage2_match:
        result['stage2_answer'] = stage2_match.group(1).capitalize()

    # Fallback: if no stage-specific answers found, use old ANSWER: pattern for stage2
    if result['stage2_answer'] == 'Unknown':
        answer_match = re.search(r'ANSWER:\s*(Yes|No|Unknown)', response, re.IGNORECASE)
        if answer_match:
            result['stage2_answer'] = answer_match.group(1).capitalize()

    return result


def parse_folio_answer(response, return_status=False):
    """Extract True/False/Unknown answer from FOLIO model response.

    Args:
        response: The model's text response
        return_status: If True, return (answer, status) tuple

    Returns:
        str or tuple: Extracted answer, or (answer, status) if return_status=True
        Status can be: "SUCCESS", "FALLBACK_PARSE", "EMPTY_RESPONSE", "UNPARSEABLE"
    """
    # Type 1: Empty response (API failure)
    if not response or not response.strip():
        if return_status:
            return None, "EMPTY_RESPONSE"
        return None

    # Look for ANSWER: format first
    answer_match = re.search(r'ANSWER:\s*(True|False|Unknown)', response, re.IGNORECASE)
    if answer_match:
        answer = normalize_answer(answer_match.group(1), answer_format="true_false")
        if return_status:
            return answer, "SUCCESS"
        return answer

    # Fallback: look for last occurrence of True/False/Unknown
    all_answers = re.findall(r'\b(True|False|Unknown)\b', response, re.IGNORECASE)
    if all_answers:
        answer = normalize_answer(all_answers[-1], answer_format="true_false")
        if return_status:
            return answer, "FALLBACK_PARSE"
        return answer

    # Type 3: Unparseable (model gave response but no valid answer)
    if return_status:
        return None, "UNPARSEABLE"
    return None


def parse_proverqa_answer(response):
    """Extract A/B/C answer from ProverQA model response.

    Handles both JSON format (ProverGen style) and plain text format.

    Args:
        response: The model's text response

    Returns:
        str: Extracted answer ('A', 'B', 'C', or 'UNKNOWN')
    """
    if not response:
        return 'UNKNOWN'

    # Try JSON parsing first (ProverGen format: {"reasoning": "...", "answer": "A"})
    import json
    # Look for JSON with "answer" key
    json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([ABC])"[^{}]*\}', response, re.IGNORECASE)
    if json_match:
        return json_match.group(1).upper()

    # Try parsing response as JSON
    try:
        # Handle case where response starts with JSON
        json_start = response.find('{')
        if json_start >= 0:
            json_end = response.rfind('}') + 1
            if json_end > json_start:
                data = json.loads(response[json_start:json_end])
                if isinstance(data, dict) and 'answer' in data:
                    answer = str(data['answer']).upper()
                    if answer in ['A', 'B', 'C']:
                        return answer
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    # Look for explicit ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*([ABC])', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Look for "answer is X" or "correct option is X" patterns
    patterns = [
        r'(?:final\s+)?answer(?:\s+is)?[:\s]+([ABC])',
        r'(?:correct\s+)?option(?:\s+is)?[:\s]+([ABC])',
        r'\b([ABC])\)?(?:\s*(?:is\s+)?(?:the\s+)?(?:correct|right|answer))',
        r'(?:^|\n)\s*([ABC])\s*$',
        r'(?:^|\n)\s*([ABC])\)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last resort: find the last occurrence of A, B, or C as standalone
    last_match = None
    for match in re.finditer(r'\b([ABC])\b', response):
        last_match = match.group(1).upper()

    return last_match if last_match else 'UNKNOWN'


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
