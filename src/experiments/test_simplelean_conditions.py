#!/usr/bin/env python3
"""
Test SimpleLean prompt conditions on false negative cases.
Runs 5 different prompt conditions to test gaming controllability.

Based on test_folio_interact_async.py flow, matching SimpleLean iteration structure.

Usage:
    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_simplelean_conditions.py \
        --dataset folio --concurrency 5
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from lean_interact import Command
from openai import AsyncOpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lean_utils import extract_lean_code, create_lean_server
from utils.answer_parsing import parse_folio_answer, normalize_answer
from utils.savers import ConditionsSaver
from utils.api_client import create_client


# Prompt conditions
ALL_CONDITIONS = ["implicit", "explicit"]

SYSTEM_PROMPTS = {
    "implicit": "prompts/simplelean-conditions/system_implicit.txt",
    "explicit": "prompts/simplelean-conditions/system_explicit.txt",
}

# Shared feedback prompts
FEEDBACK_PROMPTS = {
    "lean_error": "prompts/simplelean-shared/lean_error_feedback.txt",
    "no_lean_code": "prompts/simplelean-shared/no_lean_code_feedback.txt",
}

# Answer format for FOLIO (True/False/Unknown)
ANSWER_FORMAT = "True/False/Unknown"
ANSWER_TRUE = "True"
ANSWER_FALSE = "False"


def format_prompt(template: str, dataset: str = "folio") -> str:
    """Format prompt template with dataset-specific answer format."""
    if dataset == "folio":
        return template.format(
            answer_format="True/False/Unknown",
            answer_true="True",
            answer_false="False"
        )
    else:  # multilogieval
        return template.format(
            answer_format="Yes/No/Unknown",
            answer_true="Yes",
            answer_false="No"
        )


def load_full_folio(dataset: str) -> list:
    """Load full FOLIO dataset.

    FOLIO validation data is a flat list where each entry is a question:
    - story_id, premises (string), conclusion, label, example_id
    """
    if dataset == "folio":
        folio_path = "data/folio/original/folio-validation.json"
    else:
        raise ValueError("Full dataset only supported for folio")

    with open(folio_path, 'r') as f:
        data = json.load(f)

    cases = []
    for entry in data:
        cases.append({
            'story_id': entry.get('story_id', 0),
            'example_id': entry.get('example_id', 0),
            'premises': entry.get('premises', ''),  # Already a string
            'conclusion': entry.get('conclusion', ''),
            'ground_truth': entry.get('label', 'Unknown')
        })
    return cases


def load_prompt(path: str) -> str:
    """Load prompt from file."""
    with open(path, 'r') as f:
        return f.read().strip()


def load_false_negatives(dataset: str) -> list:
    """Load false negative cases from SimpleLean results."""
    if dataset == "folio":
        results_path = "results/SimpleLean/folio/all_results.json"
    else:
        results_path = "results/SimpleLean/multilogieval/all_results.json"

    with open(results_path, 'r') as f:
        results = json.load(f)

    false_negatives = []
    for r in results:
        if r is None:
            continue
        lean_ver = r.get('lean_verification') or {}
        if lean_ver.get('success', False) and not r.get('correct', True):
            false_negatives.append(r)

    return false_negatives


def format_user_prompt(case: dict, dataset: str) -> str:
    """Format user prompt for a case."""
    if dataset == "folio":
        premises = case.get('premises', '')
        conclusion = case.get('conclusion', '')
        return f"Textual context: {premises}\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? {conclusion}"
    else:
        context = case.get('context', '')
        question = case.get('question', '')
        return f"Textual context: {context}\n\nQuestion: Based on the above information, is the following statement true, false, or uncertain? {question}"


async def verify_with_lean_async(lean_code, lean_server, verbose=False):
    """Async version of Lean verification."""
    try:
        response = await lean_server.async_run(Command(cmd=lean_code))

        messages = response.messages if hasattr(response, 'messages') else []
        errors = [msg for msg in messages if msg.severity == 'error']
        warnings = [msg for msg in messages if msg.severity == 'warning']

        success = len(errors) == 0

        return {
            'success': success,
            'env': response.env if hasattr(response, 'env') else None,
            'errors': [msg.data for msg in errors],
            'warnings': [msg.data for msg in warnings],
            'all_messages': [{'severity': msg.severity, 'data': msg.data} for msg in messages]
        }
    except Exception as e:
        return {
            'success': False,
            'env': None,
            'errors': [str(e)],
            'warnings': [],
            'all_messages': []
        }


async def run_single_case(
    client: AsyncOpenAI,
    case: dict,
    condition: str,
    system_prompt: str,
    dataset: str,
    lean_server,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-5-2025-08-07",
    max_iterations: int = 3,
    max_retries: int = 3,
    max_completion_tokens: int = 4096
) -> dict:
    """Run a single case with a specific condition and iteration loop.

    Matches SimpleLean iteration structure exactly:
    - iteration
    - llm_response
    - prediction
    - parse_status
    - lean_code
    - lean_verification
    """
    async with semaphore:
        user_prompt = format_user_prompt(case, dataset)
        ground_truth = case.get('ground_truth', case.get('label', ''))

        # Build conversation history
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        iterations = []
        final_prediction = None
        final_parse_status = None
        final_lean_code = None
        final_verification = None
        total_retries = 0

        try:
            for iteration in range(max_iterations):
                # Retry logic for empty responses
                llm_response = ""
                retries = 0
                while retries < max_retries:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=conversation_history,
                        max_completion_tokens=max_completion_tokens
                    )
                    llm_response = response.choices[0].message.content or ""
                    # Capture reasoning traces for reasoning models (e.g., deepseek-r1)
                    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

                    if llm_response.strip():
                        break

                    retries += 1
                    total_retries += 1
                    if retries < max_retries:
                        await asyncio.sleep(1)  # Brief pause before retry

                conversation_history.append({"role": "assistant", "content": llm_response})

                # Parse answer with status tracking
                prediction, parse_status = parse_folio_answer(llm_response, return_status=True)
                lean_code = extract_lean_code(llm_response)

                # Match SimpleLean iteration structure exactly
                iteration_data = {
                    'iteration': iteration + 1,
                    'llm_response': llm_response,
                    'reasoning_content': reasoning_content,
                    'prediction': prediction,
                    'parse_status': parse_status,
                    'retries': retries,
                    'lean_code': lean_code,
                    'lean_verification': None
                }

                # Always update final prediction from LLM (independent of Lean)
                # This allows gaming analysis: LLM answer vs Lean verification
                final_prediction = prediction
                final_parse_status = parse_status
                final_lean_code = lean_code

                if lean_code:
                    lean_verification = await verify_with_lean_async(lean_code, lean_server)
                    iteration_data['lean_verification'] = lean_verification
                    final_verification = lean_verification

                    if lean_verification['success']:
                        # Lean passed - stop iterating
                        iterations.append(iteration_data)
                        break
                    else:
                        # Lean failed - provide feedback for next iteration
                        if iteration < max_iterations - 1:
                            error_messages = '\n'.join(lean_verification['errors'])
                            feedback_template = load_prompt(FEEDBACK_PROMPTS["lean_error"])
                            feedback = feedback_template.format(
                                lean_code=lean_code,
                                error_messages=error_messages,
                                answer_format=ANSWER_FORMAT
                            )
                            conversation_history.append({"role": "user", "content": feedback})
                else:
                    # No Lean code found - prompt for it
                    if iteration < max_iterations - 1:
                        feedback_template = load_prompt(FEEDBACK_PROMPTS["no_lean_code"])
                        feedback = feedback_template.format(answer_format=ANSWER_FORMAT)
                        conversation_history.append({"role": "user", "content": feedback})

                iterations.append(iteration_data)

            # Normalize for comparison
            pred_norm = normalize_answer(final_prediction, answer_format="true_false") if final_prediction else None
            gt_norm = normalize_answer(ground_truth, answer_format="true_false")
            correct = pred_norm == gt_norm

            return {
                "condition": condition,
                "prediction": final_prediction,
                "parse_status": final_parse_status,
                "ground_truth": ground_truth,
                "correct": correct,
                "model": model,
                "iterations": iterations,
                "num_iterations": len(iterations),
                "total_retries": total_retries,
                "lean_code": final_lean_code,
                "lean_verification": final_verification,
                "conversation_history": conversation_history,
            }

        except Exception as e:
            return {
                "condition": condition,
                "prediction": None,
                "parse_status": "ERROR",
                "ground_truth": ground_truth,
                "correct": False,
                "model": model,
                "error": str(e),
                "iterations": iterations,
                "num_iterations": len(iterations),
                "total_retries": total_retries,
            }


async def run_experiment(
    dataset: str,
    concurrency: int = 3,  # Lower concurrency to avoid rate limits
    max_cases: Optional[int] = None,
    model: str = "gpt-5-2025-08-07",
    max_iterations: int = 3,
    max_retries: int = 3,
    max_completion_tokens: int = 4096,
    conditions: Optional[list] = None,
    full_dataset: bool = False,
    resume_dir: Optional[str] = None
):
    """Run the full experiment."""
    load_dotenv()

    # Create client (supports OpenAI, OpenRouter, DeepSeek, etc.)
    client = create_client(model=model)

    # Use specified conditions or all
    active_conditions = conditions if conditions else ALL_CONDITIONS
    print(f"Active conditions: {active_conditions}")

    # Load cases
    if full_dataset:
        print(f"Loading full {dataset} dataset...")
        cases = load_full_folio(dataset)
        print(f"Found {len(cases)} total cases")
    else:
        print(f"Loading {dataset} false negatives...")
        cases = load_false_negatives(dataset)
        print(f"Found {len(cases)} false negative cases")
    print(f"Model: {model}, Max iterations: {max_iterations}, Max tokens: {max_completion_tokens}")

    if max_cases:
        cases = cases[:max_cases]
        print(f"Using first {max_cases} cases")

    # Load prompts with dataset-specific answer format
    prompts = {}
    for cond in active_conditions:
        template = load_prompt(SYSTEM_PROMPTS[cond])
        prompts[cond] = format_prompt(template, dataset)
        print(f"Loaded {cond} prompt")

    # Create Lean server
    print("Creating Lean server...")
    lean_server = create_lean_server()

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Initialize saver with incremental saving
    saver = ConditionsSaver(
        dataset=dataset,
        conditions=active_conditions,
        resume_dir=resume_dir,
        full_dataset=full_dataset,
        model=model,
        max_iterations=max_iterations,
        max_completion_tokens=max_completion_tokens,
        concurrency=concurrency
    )
    output_dir = Path(saver.base_dir)

    # Run all conditions
    total_tasks = len(cases) * len(active_conditions)
    print(f"\nRunning {total_tasks} total tasks ({len(cases)} cases x {len(active_conditions)} conditions)")

    async def run_and_save(case_idx, case, condition):
        """Run a single case and save result incrementally."""
        # Skip if already completed (resume support)
        if saver.is_completed(case_idx, condition):
            return None

        result = await run_single_case(
            client, case, condition, prompts[condition],
            dataset, lean_server, semaphore,
            model=model, max_iterations=max_iterations,
            max_retries=max_retries, max_completion_tokens=max_completion_tokens
        )

        # Add metadata
        result['case_idx'] = case_idx
        if dataset == "folio":
            result['example_id'] = case.get('example_id', case_idx)
            result['story_id'] = case.get('story_id')
        else:
            result['logic_type'] = case.get('logic_type', 'unknown')
            result['depth'] = case.get('depth', case.get('depth_dir', 'unknown'))

        # Save incrementally
        await saver.save_result(result, case_idx, condition)
        return result

    # Build task list
    tasks = []
    for case_idx, case in enumerate(cases):
        for condition in active_conditions:
            tasks.append(run_and_save(case_idx, case, condition))

    # Run with progress bar
    await tqdm_asyncio.gather(*tasks, desc="Running conditions")

    # Finalize and generate summary
    summary = saver.finalize()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    for condition in active_conditions:
        s = summary.get(condition, {})
        n_total = s.get('n_total', 0)
        if n_total > 0:
            print(f"\n{condition.upper()}:")
            print(f"  Accuracy: {s['n_correct']}/{n_total} ({s['accuracy']*100:.1f}%)")
            print(f"  Lean Pass: {s['n_lean_pass']}/{n_total} ({s['lean_pass_rate']*100:.1f}%)")
            print(f"  Retries: {s.get('total_retries', 0)}")
            print(f"  Predictions: {s.get('predictions', {})}")
            print(f"  Parse Status: {s.get('parse_statuses', {})}")

    # Analysis
    print("\n" + "=" * 60)
    print("CONTROLLABILITY ANALYSIS")
    print("=" * 60)

    print(f"\n{'Condition':<15} {'Accuracy':<12} {'Lean Pass':<12}")
    print("-" * 40)
    for cond in active_conditions:
        s = summary.get(cond, {})
        print(f"{cond.upper():<15} {s.get('accuracy', 0)*100:>5.1f}%      {s.get('lean_pass_rate', 0)*100:>5.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Test SimpleLean prompt conditions')
    parser.add_argument('--dataset', choices=['folio', 'multilogieval'], required=True,
                        help='Dataset to test')
    parser.add_argument('--concurrency', type=int, default=3,
                        help='Number of concurrent API calls (default: 3, lower to avoid rate limits)')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Maximum number of cases to test (for quick testing)')
    parser.add_argument('--model', type=str, default='gpt-5-2025-08-07',
                        help='Model to use (default: gpt-5-2025-08-07)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum iterations for Lean verification (default: 3)')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum retries for empty API responses (default: 3)')
    parser.add_argument('--max_completion_tokens', type=int, choices=[4096, 16384], default=4096,
                        help='Max completion tokens (default: 4096)')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated list of conditions to run (default: all)')
    parser.add_argument('--full', action='store_true',
                        help='Run on full dataset instead of false negatives only')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from existing results directory')

    args = parser.parse_args()

    # Parse conditions
    conditions = None
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(',')]
        for c in conditions:
            if c not in ALL_CONDITIONS:
                print(f"Error: Unknown condition '{c}'. Valid: {ALL_CONDITIONS}")
                sys.exit(1)

    asyncio.run(run_experiment(
        dataset=args.dataset,
        concurrency=args.concurrency,
        max_cases=args.max_cases,
        model=args.model,
        max_iterations=args.max_iterations,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        resume_dir=args.resume,
        conditions=conditions,
        full_dataset=args.full
    ))


if __name__ == '__main__':
    main()
