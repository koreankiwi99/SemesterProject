#!/usr/bin/env python3
"""
Unified SimpleLean experiment for FOLIO and MultiLogiEval datasets.

Usage:
    # FOLIO with baseline prompt
    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_simplelean.py \
        --dataset folio --model gpt-5 --condition baseline

    # MultiLogiEval
    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_simplelean.py \
        --dataset multilogieval --model deepseek-r1 --depths d4,d5
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lean_utils import extract_lean_code, create_lean_server, verify_with_lean_async
from utils.api_client import create_client
from utils.savers import SimpleLeanSaver
from utils.answer_parsing import parse_answer
from utils.prompts import (
    load_prompt, format_system_prompt, format_user_prompt,
    get_answer_format, get_answer_format_str
)
from utils.datasets import load_folio, load_multilogieval, load_multilogieval_sampled


# Prompt paths
SYSTEM_PROMPTS = {
    "system": "prompts/simplelean-conditions/system.txt",
    "baseline": "prompts/simplelean-conditions/baseline.txt",
    # Pilot prompts
    "lean4_specified": "prompts/simplelean-conditions/pilot/system_lean4_specified.txt",
    "lean4_balanced": "prompts/simplelean-conditions/pilot/system_lean4_balanced.txt",
    "lean4_minimal": "prompts/simplelean-conditions/pilot/system_lean4_minimal.txt",
    # Bidirectional prompts
    "bidir_true": "prompts/bidirectional/true_system.txt",
    "bidir_false": "prompts/bidirectional/false_system.txt",
    # Spooky (nudging) prompts for specification gaming experiment
    "spooky_true": "prompts/bidirectional/spooky_true_system.txt",
    "spooky_false": "prompts/bidirectional/spooky_false_system.txt",
}

FEEDBACK_PROMPTS = {
    "lean_error": "prompts/simplelean-shared/lean_error_feedback.txt",
    "no_lean_code": "prompts/simplelean-shared/no_lean_code_feedback.txt",
}


def add_result_metadata(result: dict, case: dict, dataset: str):
    """Add dataset-specific metadata to result."""
    if dataset == "folio":
        result['story_id'] = case.get('story_id')
        result['example_id'] = case.get('example_id')
    else:
        result['case_id'] = case.get('id')
        result['logic_type'] = case.get('logic_type')
        result['depth'] = case.get('depth')
        result['rule'] = case.get('rule')


async def run_single_case(
    client,
    case: dict,
    system_prompt: str,
    dataset: str,
    lean_server,
    semaphore: asyncio.Semaphore,
    model: str,
    max_iterations: int = 3,
    max_completion_tokens: int = 0,
    condition: str = None
) -> dict:
    """Run a single case with SimpleLean iteration logic."""
    answer_format = get_answer_format(dataset, condition)
    answer_format_str = get_answer_format_str(dataset, condition)

    async with semaphore:
        user_prompt = format_user_prompt(case, dataset, condition)
        ground_truth = case.get('ground_truth', '')

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
                api_params = {"model": model, "messages": conversation_history}
                if max_completion_tokens > 0:
                    api_params["max_completion_tokens"] = max_completion_tokens

                response = await client.chat.completions.create(**api_params)
                llm_response = response.choices[0].message.content or ""
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

                # Extract token usage (handle both object and dict formats)
                token_usage = None
                if response.usage:
                    usage = response.usage
                    if isinstance(usage, dict):
                        token_usage = {
                            'prompt_tokens': usage.get('prompt_tokens', 0) or 0,
                            'completion_tokens': usage.get('completion_tokens', 0) or 0,
                            'total_tokens': usage.get('total_tokens', 0) or 0,
                        }
                    else:
                        token_usage = {
                            'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
                            'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
                            'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
                        }

                conversation_history.append({"role": "assistant", "content": llm_response})

                prediction, parse_status = parse_answer(llm_response, answer_format)
                lean_code = extract_lean_code(llm_response)

                iteration_data = {
                    'iteration': iteration + 1,
                    'llm_response': llm_response,
                    'reasoning_content': reasoning_content,
                    'prediction': prediction,
                    'parse_status': parse_status,
                    'lean_code': lean_code,
                    'lean_verification': None,
                    'token_usage': token_usage
                }

                final_prediction = prediction
                final_parse_status = parse_status
                final_lean_code = lean_code

                if lean_code:
                    lean_verification = await verify_with_lean_async(lean_code, lean_server)
                    iteration_data['lean_verification'] = lean_verification
                    final_verification = lean_verification

                    if lean_verification['success']:
                        iterations.append(iteration_data)
                        break
                    elif iteration < max_iterations - 1:
                        error_messages = '\n'.join(lean_verification['errors'])
                        feedback = load_prompt(FEEDBACK_PROMPTS["lean_error"]).format(
                            lean_code=lean_code,
                            error_messages=error_messages,
                            answer_format=answer_format_str
                        )
                        conversation_history.append({"role": "user", "content": feedback})
                elif iteration < max_iterations - 1:
                    feedback = load_prompt(FEEDBACK_PROMPTS["no_lean_code"]).format(
                        answer_format=answer_format_str
                    )
                    conversation_history.append({"role": "user", "content": feedback})

                iterations.append(iteration_data)

            pred_norm = final_prediction.lower() if final_prediction else None
            gt_norm = ground_truth.lower() if ground_truth else None

            # Calculate correctness based on condition
            # Note: gt_norm is "yes"/"no" from dataset
            # pred_norm is normalized: "true"/"failure" for bidir_true, "false"/"failure" for bidir_false
            if answer_format == "bidir_true":
                # bidir_true: success (yes/true) correct if gt=yes, failure correct if gt=no
                is_success = pred_norm in ("yes", "true")
                correct = (is_success and gt_norm == "yes") or \
                          (pred_norm == "failure" and gt_norm == "no")
            elif answer_format == "bidir_false":
                # bidir_false: answer_false correct if gt=no, "Failure" correct if gt=yes
                # pred will be "no" (multilogieval) or "false" (folio), gt is always "yes"/"no"
                is_false_answer = pred_norm in ("no", "false")
                correct = (is_false_answer and gt_norm == "no") or \
                          (pred_norm == "failure" and gt_norm == "yes")
            else:
                correct = pred_norm == gt_norm

            # Aggregate token usage across iterations
            total_tokens = {
                'prompt_tokens': sum(it.get('token_usage', {}).get('prompt_tokens', 0) or 0 for it in iterations),
                'completion_tokens': sum(it.get('token_usage', {}).get('completion_tokens', 0) or 0 for it in iterations),
                'total_tokens': sum(it.get('token_usage', {}).get('total_tokens', 0) or 0 for it in iterations),
            }

            result = {
                "prediction": final_prediction,
                "parse_status": final_parse_status,
                "ground_truth": ground_truth,
                "correct": correct,
                "model": model,
                "iterations": iterations,
                "num_iterations": len(iterations),
                "lean_code": final_lean_code,
                "lean_verification": final_verification,
                "total_tokens": total_tokens,
            }
            add_result_metadata(result, case, dataset)
            return result

        except Exception as e:
            result = {
                "prediction": final_prediction,
                "parse_status": "ERROR",
                "ground_truth": ground_truth,
                "correct": False,
                "model": model,
                "error": str(e),
                "iterations": iterations,
                "num_iterations": len(iterations),
            }
            add_result_metadata(result, case, dataset)
            return result


async def run_experiment(
    dataset: str,
    model: str,
    condition: str = "implicit",
    concurrency: int = 5,
    max_cases: Optional[int] = None,
    max_iterations: int = 3,
    max_completion_tokens: int = 0,
    depths: Optional[list] = None,
    logic_types: Optional[list] = None,
    resume_dir: Optional[str] = None,
    cases_file: Optional[str] = None,
    data_file: Optional[str] = None
):
    """Run SimpleLean experiment."""
    load_dotenv()
    client = create_client(model=model)

    # Load dataset
    print(f"Loading {dataset} dataset...")
    if dataset == "folio":
        cases = load_folio()
    elif data_file:
        cases = load_multilogieval_sampled(data_file)
    else:
        depths = depths or ["d4", "d5"]
        logic_types = logic_types or ["fol", "nm", "pl"]
        cases = load_multilogieval(depths, logic_types)

    print(f"Found {len(cases)} cases")
    print(f"Model: {model}, Condition: {condition}, Max iterations: {max_iterations}")

    if max_cases:
        cases = cases[:max_cases]
        print(f"Using first {max_cases} cases")

    # Filter to specific cases if cases_file provided
    if cases_file:
        import json
        with open(cases_file) as f:
            case_indices = set(json.loads(line)['case_idx'] for line in f)
        cases = [(idx, cases[idx]) for idx in sorted(case_indices)]
        print(f"Filtered to {len(cases)} cases from {cases_file}")
    else:
        cases = list(enumerate(cases))

    # Load and format system prompt
    template = load_prompt(SYSTEM_PROMPTS[condition])
    system_prompt = format_system_prompt(template, dataset)

    # Create Lean server
    print("Creating Lean server...")
    lean_server = create_lean_server()
    semaphore = asyncio.Semaphore(concurrency)

    # Initialize saver
    saver_kwargs = {
        "output_dir": "results/simplelean",
        "dataset": dataset,
        "model": model,
        "condition": condition,
        "max_iterations": max_iterations,
        "max_completion_tokens": max_completion_tokens,
        "concurrency": concurrency,
        "resume_dir": resume_dir,
    }
    if dataset == "multilogieval":
        saver_kwargs["depths"] = depths
        saver_kwargs["logic_types"] = logic_types

    saver = SimpleLeanSaver(**saver_kwargs)
    print(f"Output directory: {saver.base_dir}")

    async def run_and_save(idx, case):
        if saver.is_completed(idx):
            return None
        result = await run_single_case(
            client, case, system_prompt, dataset, lean_server, semaphore,
            model, max_iterations, max_completion_tokens, condition
        )
        result['case_idx'] = idx
        await saver.save_result(result, idx)
        return result

    print(f"\nRunning {len(cases)} cases...")
    tasks = [run_and_save(idx, case) for idx, case in cases]
    await tqdm_asyncio.gather(*tasks, desc=f"SimpleLean {dataset}")

    summary = saver.finalize()

    print("\n" + "=" * 60)
    print(f"RESULTS: {dataset.upper()}")
    print("=" * 60)
    print(f"Accuracy: {summary['correct']}/{summary['total']} ({summary['accuracy']*100:.1f}%)")
    print(f"Lean Pass: {summary['lean_pass']}/{summary['total']} ({summary['lean_pass_rate']*100:.1f}%)")

    if dataset == "multilogieval" and 'by_depth' in summary:
        print("\nBy Depth:")
        for d in sorted(summary['by_depth'].keys()):
            s = summary['by_depth'][d]
            acc = 100*s['correct']/s['total'] if s['total'] > 0 else 0
            print(f"  {d}: {s['correct']}/{s['total']} ({acc:.1f}%)")

    return summary


def main():
    parser = argparse.ArgumentParser(description='SimpleLean experiment')
    parser.add_argument('--dataset', required=True, choices=['folio', 'multilogieval'])
    parser.add_argument('--model', default='deepseek-r1')
    parser.add_argument('--condition', default='baseline', choices=['system', 'baseline', 'lean4_specified', 'lean4_balanced', 'lean4_minimal', 'bidir_true', 'bidir_false', 'spooky_true', 'spooky_false'])
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--max_completion_tokens', type=int, default=0)
    parser.add_argument('--depths', default='d4,d5')
    parser.add_argument('--logic_types', default='fol,nm,pl')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--cases_file', default=None, help='JSONL file with case indices to run')
    parser.add_argument('--data_file', default=None, help='JSON file with sampled data')

    args = parser.parse_args()

    asyncio.run(run_experiment(
        dataset=args.dataset,
        model=args.model,
        condition=args.condition,
        concurrency=args.concurrency,
        max_cases=args.max_cases,
        max_iterations=args.max_iterations,
        max_completion_tokens=args.max_completion_tokens,
        depths=[d.strip() for d in args.depths.split(',')],
        logic_types=[lt.strip() for lt in args.logic_types.split(',')],
        resume_dir=args.resume,
        cases_file=args.cases_file,
        data_file=args.data_file
    ))


if __name__ == '__main__':
    main()
