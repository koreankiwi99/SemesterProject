#!/usr/bin/env python3
"""
ProverQA zero-shot evaluation using async API calls.

Uses simple prompts similar to FOLIO/MultiLogiEval format with True/False/Uncertain answers.
"""

import argparse
import json
import os
import asyncio
import re

from tqdm.asyncio import tqdm_asyncio

from utils.prompts import load_prompt
from utils.api_client import create_client, get_provider
from utils.savers import ProverQASaver


def load_proverqa_tfu(data_file):
    """Load ProverQA data with True/False/Uncertain format."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def parse_tfu_answer(response: str) -> str:
    """Parse True/False/Uncertain answer from model response.

    Args:
        response: Model response text

    Returns:
        str: 'True', 'False', 'Uncertain', or 'UNKNOWN'
    """
    response_lower = response.lower().strip()

    # Check for exact matches first
    if response_lower in ['true', 'false', 'uncertain']:
        return response_lower.capitalize()

    # Check for answer patterns
    patterns = [
        r'answer[:\s]+(\btrue\b|\bfalse\b|\buncertain\b)',
        r'(\btrue\b|\bfalse\b|\buncertain\b)\s*$',
        r'^(\btrue\b|\bfalse\b|\buncertain\b)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            return match.group(1).capitalize()

    # Check if any keyword appears
    if 'uncertain' in response_lower:
        return 'Uncertain'
    if 'false' in response_lower and 'true' not in response_lower:
        return 'False'
    if 'true' in response_lower and 'false' not in response_lower:
        return 'True'

    return 'UNKNOWN'


async def test_question_async(example, client, system_prompt, user_template, model, semaphore):
    """Test a single question asynchronously.

    Args:
        example: Question sample dictionary
        client: Unified async client
        system_prompt: System prompt string
        user_template: User prompt template
        model: Model name
        semaphore: Asyncio semaphore for rate limiting

    Returns:
        dict: Result dictionary
    """
    async with semaphore:
        # Build user prompt
        user_prompt = user_template.format(
            context=example['context'],
            question=example['question']
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            model_response = response.choices[0].message.content.strip()
            prediction = parse_tfu_answer(model_response)
            ground_truth = example['answer']
            is_correct = prediction == ground_truth

            return {
                'id': example['id'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': is_correct,
                'context': example['context'],
                'question': example['question'],
                'options': example.get('options', ['True', 'False', 'Uncertain']),
                'model': model,
                'model_response': model_response
            }

        except Exception as e:
            return {
                'id': example['id'],
                'ground_truth': example['answer'],
                'prediction': 'UNKNOWN',
                'correct': False,
                'context': example['context'],
                'question': example['question'],
                'options': example.get('options', ['True', 'False', 'Uncertain']),
                'model': model,
                'model_response': f'ERROR: {str(e)}',
                'error': str(e)
            }


async def main_async():
    parser = argparse.ArgumentParser(
        description='ProverQA zero-shot evaluation with True/False/Uncertain format'
    )
    parser.add_argument('--api_key', help='API key (uses env var if not provided)')
    parser.add_argument('--data_file', required=True, help='Path to ProverQA TFU JSON file (hard_tfu.json)')
    parser.add_argument('--system_prompt', default='prompts/proverqa/zeroshot_system.txt',
                        help='Path to system prompt file')
    parser.add_argument('--user_prompt', default='prompts/proverqa/zeroshot_user.txt',
                        help='Path to user prompt template file')
    parser.add_argument('--model', default='gpt-5-2025-08-07', help='Model to use')
    parser.add_argument('--output_dir', default='results/proverqa', help='Output directory')
    parser.add_argument('--concurrency', type=int, default=32, help='Number of concurrent requests')
    parser.add_argument('--num_questions', type=int, default=0,
                        help='Number of questions to test (0 = all)')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')

    args = parser.parse_args()

    # Get difficulty from filename
    difficulty = os.path.basename(args.data_file).replace('.json', '').replace('_tfu', '')

    # Load prompts
    system_prompt = load_prompt(args.system_prompt)
    user_template = load_prompt(args.user_prompt)
    print(f"Loaded system prompt from {args.system_prompt}")
    print(f"Loaded user template from {args.user_prompt}")

    # Load dataset
    data = load_proverqa_tfu(args.data_file)

    # Limit questions if specified
    if args.num_questions > 0:
        data = data[:args.num_questions]

    print(f"\nTesting {len(data)} questions from ProverQA {difficulty} (zero-shot)")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")

    # Initialize saver
    output_subdir = f"{difficulty}_zeroshot"
    if args.resume:
        saver = ProverQASaver(args.output_dir, output_subdir, args.model, resume_dir=args.resume)
    else:
        saver = ProverQASaver(args.output_dir, output_subdir, args.model)

    # Initialize unified async client
    provider = get_provider(args.model)
    print(f"Using provider: {provider}")
    client = create_client(api_key=args.api_key, model=args.model)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    # Create tasks for all questions
    tasks = [
        test_question_async(example, client, system_prompt, user_template, args.model, semaphore)
        for example in data
    ]

    # Process all questions with progress bar
    print(f"\nProcessing {len(tasks)} questions...")
    results = await tqdm_asyncio.gather(*tasks, desc="Processing")

    # Save results
    print("\nSaving results...")
    total_correct = 0
    errors = 0

    # Count by answer type
    by_gt = {'True': {'correct': 0, 'total': 0},
             'False': {'correct': 0, 'total': 0},
             'Uncertain': {'correct': 0, 'total': 0}}
    pred_counts = {'True': 0, 'False': 0, 'Uncertain': 0, 'UNKNOWN': 0}

    for i, result in enumerate(results):
        if result['correct']:
            total_correct += 1
        if 'error' in result:
            errors += 1

        gt = result['ground_truth']
        pred = result['prediction']
        by_gt[gt]['total'] += 1
        if result['correct']:
            by_gt[gt]['correct'] += 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

        saver.save_result(result, i, len(data))

        # Progress indicator
        status = '✓' if result['correct'] else '✗'
        if i < 10 or i % 50 == 0:
            print(f"[{i+1}/{len(data)}] GT={result['ground_truth']} Pred={result['prediction']} {status}")

    # Finalize
    saver.finalize(len(data), total_correct)

    # Close client
    await client.close()

    # Print summary
    accuracy = total_correct / len(data) if data else 0
    print(f"\n{'='*70}")
    print(f"ProverQA {difficulty.upper()} Zero-Shot Results ({args.model})")
    print(f"{'='*70}")
    print(f"Total: {len(data)}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2%}")
    if errors > 0:
        print(f"Errors: {errors}")

    print(f"\nBy Ground Truth:")
    for gt, stats in by_gt.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {gt}: {stats['correct']}/{stats['total']} ({acc:.2%})")

    print(f"\nPrediction Distribution:")
    for pred, count in sorted(pred_counts.items()):
        print(f"  {pred}: {count}")

    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main_async())
