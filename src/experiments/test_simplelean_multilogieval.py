#!/usr/bin/env python3
"""
Test SimpleLean on MultiLogiEval dataset.
Uses Yes/No/Unknown answer format with Lean verification.

Tracks both LLM answers and Lean verification independently.

Usage:
    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_simplelean_multilogieval.py \
        --model deepseek-r1 --concurrency 5 --depths d4,d5
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from lean_interact import Command

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lean_utils import extract_lean_code, create_lean_server
from utils.api_client import create_client
from utils.savers import MultiLogiEvalSaver

# System prompts (shared with FOLIO)
SYSTEM_PROMPTS = {
    "implicit": "prompts/simplelean-conditions/system_implicit.txt",
    "explicit": "prompts/simplelean-conditions/system_explicit.txt",
}

# Shared feedback prompts
FEEDBACK_PROMPTS = {
    "lean_error": "prompts/simplelean-shared/lean_error_feedback.txt",
    "no_lean_code": "prompts/simplelean-shared/no_lean_code_feedback.txt",
}

# Answer format for MultiLogiEval
ANSWER_FORMAT = "Yes/No/Unknown"
ANSWER_TRUE = "Yes"
ANSWER_FALSE = "No"


def format_prompt(template: str) -> str:
    """Format prompt template with MultiLogiEval answer format."""
    return template.format(
        answer_format="Yes/No/Unknown",
        answer_true="Yes",
        answer_false="No"
    )


def load_multilogieval(depths: list = ["d4", "d5"], logic_types: list = ["fol", "nm", "pl"]) -> list:
    """Load MultiLogiEval data for specified depths and logic types."""
    data_dir = Path("data/multilogieval/original/data")
    cases = []

    for depth in depths:
        depth_dir = data_dir / f"{depth}_Data"
        if not depth_dir.exists():
            print(f"Warning: {depth_dir} does not exist")
            continue

        for logic_type in logic_types:
            logic_dir = depth_dir / logic_type
            if not logic_dir.exists():
                continue

            for json_file in logic_dir.glob("*.json"):
                try:
                    with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)
                except UnicodeDecodeError:
                    with open(json_file, encoding='latin-1') as f:
                        data = json.load(f)

                rule = data.get('rule', json_file.stem)
                samples = data.get('samples', [])

                for sample in samples:
                    cases.append({
                        'id': sample.get('id'),
                        'context': sample.get('context', ''),
                        'question': sample.get('question', ''),
                        'answer': sample.get('answer', '').lower(),  # yes/no
                        'logic_type': logic_type,
                        'depth': depth,
                        'rule': rule
                    })

    return cases


def load_prompt(path: str) -> str:
    """Load prompt from file."""
    with open(path, 'r') as f:
        return f.read().strip()


def parse_multilogieval_answer(response: str) -> tuple:
    """Parse Yes/No/Unknown answer from response.

    Returns: (answer, parse_status)
    """
    import re

    if not response:
        return None, "EMPTY"

    # Look for ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*(Yes|No|Unknown)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).capitalize(), "SUCCESS"

    # Fallback: look for standalone yes/no/unknown
    response_lower = response.lower()
    if 'unknown' in response_lower:
        return 'Unknown', "FALLBACK"
    elif 'yes' in response_lower and 'no' not in response_lower:
        return 'Yes', "FALLBACK"
    elif 'no' in response_lower and 'yes' not in response_lower:
        return 'No', "FALLBACK"

    return None, "PARSE_FAILED"


async def verify_with_lean_async(lean_code: str, lean_server) -> dict:
    """Async version of Lean verification."""
    try:
        response = await lean_server.async_run(Command(cmd=lean_code))
        messages = response.messages if hasattr(response, 'messages') else []
        errors = [msg for msg in messages if msg.severity == 'error']
        warnings = [msg for msg in messages if msg.severity == 'warning']

        return {
            'success': len(errors) == 0,
            'errors': [msg.data for msg in errors],
            'warnings': [msg.data for msg in warnings],
            'all_messages': [{'severity': msg.severity, 'data': msg.data} for msg in messages]
        }
    except Exception as e:
        return {'success': False, 'errors': [str(e)], 'warnings': [], 'all_messages': []}


async def run_single_case(
    client,
    case: dict,
    system_prompt: str,
    lean_server,
    semaphore: asyncio.Semaphore,
    model: str,
    max_iterations: int = 3,
    max_completion_tokens: int = 4096
) -> dict:
    """Run a single MultiLogiEval case.

    Iteration Logic:
    1. LLM generates Lean code + answer
    2. Lean verifies code
    3. If Lean PASSES → stop, record result
    4. If Lean FAILS → send error feedback, continue to next iteration
    5. After max_iterations, record final state

    LLM answer is tracked INDEPENDENTLY of Lean verification to observe gaming.
    """
    async with semaphore:
        context = case.get('context', '')
        question = case.get('question', '')
        ground_truth = case.get('answer', '')  # yes/no

        user_prompt = f"Context: {context}\n\nQuestion: {question}"

        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        iterations = []
        final_prediction = None
        final_parse_status = None
        final_lean_code = None
        final_verification = None

        try:
            for iteration in range(max_iterations):
                response = await client.chat.completions.create(
                    model=model,
                    messages=conversation_history,
                    max_completion_tokens=max_completion_tokens
                )
                llm_response = response.choices[0].message.content or ""

                # Capture reasoning traces (for models like DeepSeek-R1)
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

                conversation_history.append({"role": "assistant", "content": llm_response})

                # Parse LLM's answer (independent of Lean)
                prediction, parse_status = parse_multilogieval_answer(llm_response)
                lean_code = extract_lean_code(llm_response)

                iteration_data = {
                    'iteration': iteration + 1,
                    'llm_response': llm_response,
                    'reasoning_content': reasoning_content,
                    'prediction': prediction,
                    'parse_status': parse_status,
                    'lean_code': lean_code,
                    'lean_verification': None
                }

                # Always update final prediction from LLM (independent of Lean)
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
                    # No Lean code found
                    if iteration < max_iterations - 1:
                        feedback_template = load_prompt(FEEDBACK_PROMPTS["no_lean_code"])
                        feedback = feedback_template.format(answer_format=ANSWER_FORMAT)
                        conversation_history.append({"role": "user", "content": feedback})

                iterations.append(iteration_data)

            # Normalize for comparison (yes/no/unknown)
            pred_norm = final_prediction.lower() if final_prediction else None
            gt_norm = ground_truth.lower() if ground_truth else None
            correct = pred_norm == gt_norm

            return {
                "prediction": final_prediction,
                "parse_status": final_parse_status,
                "ground_truth": ground_truth,
                "correct": correct,
                "model": model,
                "iterations": iterations,
                "num_iterations": len(iterations),
                "lean_code": final_lean_code,
                "lean_verification": final_verification,
                "case_id": case.get('id'),
                "logic_type": case.get('logic_type'),
                "depth": case.get('depth'),
                "rule": case.get('rule'),
                "context": context,
                "question": question,
            }

        except Exception as e:
            return {
                "prediction": final_prediction,
                "parse_status": "ERROR",
                "ground_truth": ground_truth,
                "correct": False,
                "model": model,
                "error": str(e),
                "iterations": iterations,
                "num_iterations": len(iterations),
                "case_id": case.get('id'),
                "logic_type": case.get('logic_type'),
                "depth": case.get('depth'),
            }


async def run_experiment(
    model: str,
    concurrency: int = 5,
    max_cases: Optional[int] = None,
    max_iterations: int = 3,
    max_completion_tokens: int = 4096,
    depths: list = ["d4", "d5"],
    logic_types: list = ["fol", "nm", "pl"],
    prompt_type: str = "implicit",
    resume_dir: Optional[str] = None
):
    """Run MultiLogiEval experiment."""
    load_dotenv()

    client = create_client(model=model)

    print(f"Loading MultiLogiEval data (depths: {depths}, logic: {logic_types})...")
    cases = load_multilogieval(depths=depths, logic_types=logic_types)
    print(f"Found {len(cases)} cases")
    print(f"Model: {model}, Prompt: {prompt_type}, Max iterations: {max_iterations}, Max tokens: {max_completion_tokens}")

    if max_cases:
        cases = cases[:max_cases]
        print(f"Using first {max_cases} cases")

    # Load prompt with answer format substitution
    prompt_path = SYSTEM_PROMPTS[prompt_type]
    template = load_prompt(prompt_path)
    system_prompt = format_prompt(template)
    print(f"Loaded {prompt_type} prompt from {prompt_path}")

    # Create Lean server
    print("Creating Lean server...")
    lean_server = create_lean_server()

    semaphore = asyncio.Semaphore(concurrency)

    # Initialize saver
    saver = MultiLogiEvalSaver(
        output_dir="results/simplelean_multilogieval",
        model=model,
        depths=depths,
        resume_dir=resume_dir
    )
    print(f"Output directory: {saver.base_dir}")

    print(f"\nRunning {len(cases)} cases...")

    async def run_and_save(idx, case):
        # Skip if already completed (resume support)
        if saver.is_completed(idx):
            return None

        result = await run_single_case(
            client, case, system_prompt, lean_server, semaphore,
            model=model, max_iterations=max_iterations,
            max_completion_tokens=max_completion_tokens
        )
        result['case_idx'] = idx

        # Save incrementally
        await saver.save_result(result, idx, case)
        return result

    tasks = [run_and_save(idx, case) for idx, case in enumerate(cases)]
    results = await tqdm_asyncio.gather(*tasks, desc="Running MultiLogiEval")

    # Filter None results (from resumed/skipped cases)
    results = [r for r in results if r is not None]

    # Finalize and generate summary
    summary = saver.finalize()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nOverall: {summary['correct']}/{summary['total']} ({100*summary['accuracy']:.1f}%)")
    print(f"Lean Pass: {summary['lean_pass']}/{summary['total']} ({100*summary['lean_pass_rate']:.1f}%)")

    print("\nBy Depth:")
    for d in sorted(summary['by_depth'].keys()):
        s = summary['by_depth'][d]
        acc = 100*s['correct']/s['total'] if s['total'] > 0 else 0
        lp = 100*s['lean_pass']/s['total'] if s['total'] > 0 else 0
        print(f"  {d}: {s['correct']}/{s['total']} ({acc:.1f}%) | Lean: {lp:.1f}%")

    print("\nBy Logic Type:")
    for lt in sorted(summary['by_logic'].keys()):
        s = summary['by_logic'][lt]
        acc = 100*s['correct']/s['total'] if s['total'] > 0 else 0
        lp = 100*s['lean_pass']/s['total'] if s['total'] > 0 else 0
        print(f"  {lt}: {s['correct']}/{s['total']} ({acc:.1f}%) | Lean: {lp:.1f}%")

    print("\nGaming Analysis:")
    ga = summary['gaming_analysis']
    print(f"  Lean Pass but Wrong Answer: {ga['lean_pass_but_wrong']}")
    print(f"  Answer Flips Across Iterations: {ga['answer_flips_across_iterations']}")

    print(f"\nResults saved to: {saver.base_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Test SimpleLean on MultiLogiEval')
    parser.add_argument('--model', type=str, default='deepseek-r1',
                        help='Model to use (default: deepseek-r1)')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Number of concurrent API calls')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Maximum cases to test')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum Lean iterations')
    parser.add_argument('--max_completion_tokens', type=int, default=16384,
                        help='Max completion tokens')
    parser.add_argument('--depths', type=str, default='d4,d5',
                        help='Comma-separated depths (default: d4,d5)')
    parser.add_argument('--logic_types', type=str, default='fol,nm,pl',
                        help='Comma-separated logic types (default: fol,nm,pl)')
    parser.add_argument('--prompt_type', type=str, choices=['implicit', 'explicit'], default='implicit',
                        help='Prompt type: implicit (default) or explicit (Lean-based answer required)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from existing output directory')

    args = parser.parse_args()

    depths = [d.strip() for d in args.depths.split(',')]
    logic_types = [lt.strip() for lt in args.logic_types.split(',')]

    asyncio.run(run_experiment(
        model=args.model,
        concurrency=args.concurrency,
        max_cases=args.max_cases,
        max_iterations=args.max_iterations,
        max_completion_tokens=args.max_completion_tokens,
        depths=depths,
        logic_types=logic_types,
        prompt_type=args.prompt_type,
        resume_dir=args.resume
    ))


if __name__ == '__main__':
    main()
