#!/usr/bin/env python3
"""
Test SimpleLean on MultiLogiEval dataset (depth 4 and 5 only).
Uses Yes/No/Unknown answer format.

Usage:
    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_simplelean_multilogieval.py \
        --model deepseek-r1 --concurrency 5
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lean_utils import extract_lean_code, create_lean_server
from utils.answer_parsing import normalize_answer
from utils.api_client import create_client


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
    """Parse Yes/No/Unknown answer from response."""
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


async def verify_with_lean_async(lean_code, lean_server):
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
        }
    except Exception as e:
        return {'success': False, 'errors': [str(e)], 'warnings': []}


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
    """Run a single MultiLogiEval case."""
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
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

                conversation_history.append({"role": "assistant", "content": llm_response})

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

                if lean_code:
                    lean_verification = await verify_with_lean_async(lean_code, lean_server)
                    iteration_data['lean_verification'] = lean_verification

                    if lean_verification['success']:
                        final_prediction = prediction
                        final_parse_status = parse_status
                        final_lean_code = lean_code
                        final_verification = lean_verification
                        iterations.append(iteration_data)
                        break
                    else:
                        if iteration < max_iterations - 1:
                            error_messages = '\n'.join(lean_verification['errors'])
                            feedback = (
                                f"The Lean code has errors:\n{error_messages}\n\n"
                                f"Please provide corrected Lean code in <lean></lean> tags.\n"
                                f"Then provide: ANSWER: Yes/No/Unknown"
                            )
                            conversation_history.append({"role": "user", "content": feedback})
                else:
                    if iteration < max_iterations - 1:
                        feedback = (
                            f"Please provide Lean code in <lean></lean> tags.\n"
                            f"Then provide: ANSWER: Yes/No/Unknown"
                        )
                        conversation_history.append({"role": "user", "content": feedback})

                iterations.append(iteration_data)

                if iteration == max_iterations - 1:
                    final_prediction = prediction
                    final_parse_status = parse_status
                    final_lean_code = lean_code
                    final_verification = iteration_data.get('lean_verification')

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
            }

        except Exception as e:
            return {
                "prediction": None,
                "parse_status": "ERROR",
                "ground_truth": ground_truth,
                "correct": False,
                "model": model,
                "error": str(e),
                "iterations": iterations,
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
    logic_types: list = ["fol", "nm", "pl"]
):
    """Run MultiLogiEval experiment."""
    load_dotenv()

    client = create_client(model=model)

    print(f"Loading MultiLogiEval data (depths: {depths}, logic: {logic_types})...")
    cases = load_multilogieval(depths=depths, logic_types=logic_types)
    print(f"Found {len(cases)} cases")
    print(f"Model: {model}, Max iterations: {max_iterations}, Max tokens: {max_completion_tokens}")

    if max_cases:
        cases = cases[:max_cases]
        print(f"Using first {max_cases} cases")

    # Load prompt
    prompt_path = "prompts/simplelean-multilogieval/condition1_baseline_system.txt"
    system_prompt = load_prompt(prompt_path)
    print(f"Loaded prompt from {prompt_path}")

    # Create Lean server
    print("Creating Lean server...")
    lean_server = create_lean_server()

    semaphore = asyncio.Semaphore(concurrency)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("/", "-").replace(":", "-")
    output_dir = Path(f"results/simplelean_multilogieval/{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(exist_ok=True)

    print(f"\nRunning {len(cases)} cases...")

    async def run_and_save(idx, case):
        result = await run_single_case(
            client, case, system_prompt, lean_server, semaphore,
            model=model, max_iterations=max_iterations,
            max_completion_tokens=max_completion_tokens
        )
        result['case_idx'] = idx

        # Save individual response
        response_file = responses_dir / f"case_{idx}_{case.get('logic_type')}_{case.get('depth')}.txt"
        with open(response_file, 'w') as f:
            f.write(f"Logic: {case.get('logic_type')}\n")
            f.write(f"Depth: {case.get('depth')}\n")
            f.write(f"Rule: {case.get('rule')}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Correct: {result['correct']}\n")
            f.write(f"Lean Pass: {result.get('lean_verification', {}).get('success', False)}\n")
            f.write("\n" + "="*50 + "\n\n")
            for it in result.get('iterations', []):
                f.write(f"--- Iteration {it['iteration']} ---\n")
                f.write(it.get('llm_response', '')[:2000])
                f.write("\n\n")

        return result

    tasks = [run_and_save(idx, case) for idx, case in enumerate(cases)]
    results = await tqdm_asyncio.gather(*tasks, desc="Running MultiLogiEval")

    # Save all results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate summary
    total = len(results)
    correct = sum(1 for r in results if r.get('correct'))
    lean_pass = sum(1 for r in results if r.get('lean_verification', {}).get('success'))

    # By depth
    by_depth = {}
    for r in results:
        d = r.get('depth', 'unknown')
        if d not in by_depth:
            by_depth[d] = {'total': 0, 'correct': 0, 'lean_pass': 0}
        by_depth[d]['total'] += 1
        if r.get('correct'):
            by_depth[d]['correct'] += 1
        if r.get('lean_verification', {}).get('success'):
            by_depth[d]['lean_pass'] += 1

    # By logic type
    by_logic = {}
    for r in results:
        lt = r.get('logic_type', 'unknown')
        if lt not in by_logic:
            by_logic[lt] = {'total': 0, 'correct': 0, 'lean_pass': 0}
        by_logic[lt]['total'] += 1
        if r.get('correct'):
            by_logic[lt]['correct'] += 1
        if r.get('lean_verification', {}).get('success'):
            by_logic[lt]['lean_pass'] += 1

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nOverall: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Lean Pass: {lean_pass}/{total} ({100*lean_pass/total:.1f}%)")

    print("\nBy Depth:")
    for d in sorted(by_depth.keys()):
        s = by_depth[d]
        print(f"  {d}: {s['correct']}/{s['total']} ({100*s['correct']/s['total']:.1f}%) | Lean: {100*s['lean_pass']/s['total']:.1f}%")

    print("\nBy Logic Type:")
    for lt in sorted(by_logic.keys()):
        s = by_logic[lt]
        print(f"  {lt}: {s['correct']}/{s['total']} ({100*s['correct']/s['total']:.1f}%) | Lean: {100*s['lean_pass']/s['total']:.1f}%")

    # Save summary
    summary = {
        'total': total,
        'correct': correct,
        'accuracy': correct/total,
        'lean_pass': lean_pass,
        'lean_pass_rate': lean_pass/total,
        'by_depth': by_depth,
        'by_logic': by_logic
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
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
        logic_types=logic_types
    ))


if __name__ == '__main__':
    main()
