#!/usr/bin/env python3
"""
ProverQA CoT evaluation using async API calls for efficient processing.

Uses the official ProverGen evaluation format with 2-shot ICL examples.
"""

import argparse
import json
import os
import asyncio

from tqdm.asyncio import tqdm_asyncio

from utils.prompts import load_prompt
from utils.api_client import create_client, get_provider
from utils.answer_parsing import parse_proverqa_answer
from utils.savers import ProverQASaver
from datasets.proverqa import load_proverqa


# Default system prompt from ProverGen paper
PROVERGEN_SYSTEM_PROMPT = "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."


def load_icl_template(icl_path, difficulty, mode='CoT'):
    """Load ICL template from ProverGen format.

    Args:
        icl_path: Path to ICL examples JSON file
        difficulty: 'easy', 'medium', or 'hard'
        mode: 'CoT' or 'Direct'

    Returns:
        str: ICL template with [[CONTEXT]], [[QUESTION]], [[OPTIONS]] placeholders
    """
    with open(icl_path, 'r') as f:
        icl_data = json.load(f)

    key = f"{difficulty}_{mode}"
    if key not in icl_data:
        # Fallback to non-difficulty-specific key
        key = mode

    return icl_data[key]


def build_provergen_prompt(example, icl_template):
    """Build prompt using ProverGen ICL template.

    Args:
        example: Question sample dictionary
        icl_template: ICL template with placeholders

    Returns:
        str: Complete user prompt
    """
    options_str = '\n'.join(example['options'])

    prompt = icl_template.replace('[[CONTEXT]]', example['context'])
    prompt = prompt.replace('[[QUESTION]]', example['question'])
    prompt = prompt.replace('[[OPTIONS]]', options_str)

    return prompt


async def test_question_async(example, icl_template, client, system_prompt, model, semaphore):
    """Test a single question asynchronously.

    Args:
        example: Question sample dictionary
        icl_template: ICL template with placeholders
        client: Unified async client
        system_prompt: System prompt string
        model: Model name
        semaphore: Asyncio semaphore for rate limiting

    Returns:
        dict: Result dictionary
    """
    async with semaphore:
        user_prompt = build_provergen_prompt(example, icl_template)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            model_response = response.choices[0].message.content.strip()
            prediction = parse_proverqa_answer(model_response)
            ground_truth = example['answer']
            is_correct = prediction == ground_truth

            return {
                'id': example['id'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': is_correct,
                'context': example['context'],
                'question': example['question'],
                'options': example['options'],
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
                'options': example['options'],
                'model': model,
                'model_response': f'ERROR: {str(e)}',
                'error': str(e)
            }


async def main_async():
    parser = argparse.ArgumentParser(
        description='ProverQA CoT evaluation with async API calls (ProverGen format)'
    )
    parser.add_argument('--api_key', help='API key (uses OPENAI_API_KEY env var if not provided)')
    parser.add_argument('--data_file', required=True, help='Path to ProverQA JSON file (easy.json, medium.json, or hard.json)')
    parser.add_argument('--icl_examples', default='prompts/proverqa/icl_examples.json',
                        help='Path to ICL examples JSON file')
    parser.add_argument('--system_prompt', default=None,
                        help='Path to system prompt file (default: use ProverGen format)')
    parser.add_argument('--mode', default='CoT', choices=['CoT', 'Direct'],
                        help='Evaluation mode: CoT or Direct')
    parser.add_argument('--model', default='gpt-5-2025-08-07', help='Model to use (default: gpt-5-2025-08-07 snapshot)')
    parser.add_argument('--output_dir', default='results/proverqa', help='Output directory')
    parser.add_argument('--concurrency', type=int, default=32, help='Number of concurrent requests')
    parser.add_argument('--num_questions', type=int, default=0,
                        help='Number of questions to test (0 = all)')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per request in seconds')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')

    args = parser.parse_args()

    # Get difficulty from filename
    difficulty = os.path.basename(args.data_file).replace('.json', '')

    # Load system prompt
    if args.system_prompt:
        system_template = load_prompt(args.system_prompt)
        print(f"Loaded system prompt from {args.system_prompt}")
    else:
        system_template = PROVERGEN_SYSTEM_PROMPT
        print(f"Using ProverGen system prompt")

    # Load ICL template
    print(f"Loading ICL examples from {args.icl_examples}...")
    icl_template = load_icl_template(args.icl_examples, difficulty, args.mode)
    print(f"Loaded {difficulty}_{args.mode} ICL template")

    # Load dataset
    data = load_proverqa(args.data_file)

    # Limit questions if specified
    if args.num_questions > 0:
        data = data[:args.num_questions]

    print(f"\nTesting {len(data)} questions from ProverQA {difficulty}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Concurrency: {args.concurrency}")

    # Initialize saver
    if args.resume:
        saver = ProverQASaver(args.output_dir, difficulty, args.model, resume_dir=args.resume)
    else:
        saver = ProverQASaver(args.output_dir, difficulty, args.model)

    # Initialize unified async client
    provider = get_provider(args.model)
    print(f"Using provider: {provider}")
    client = create_client(api_key=args.api_key, model=args.model)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    # Create tasks for all questions
    tasks = [
        test_question_async(example, icl_template, client, system_template, args.model, semaphore)
        for example in data
    ]

    # Process all questions with progress bar
    print(f"\nProcessing {len(tasks)} questions...")
    results = await tqdm_asyncio.gather(*tasks, desc="Processing")

    # Save results
    print("\nSaving results...")
    total_correct = 0
    errors = 0

    for i, result in enumerate(results):
        if result['correct']:
            total_correct += 1
        if 'error' in result:
            errors += 1

        saver.save_result(result, i, len(data))

        # Progress indicator
        status = '✓' if result['correct'] else '✗'
        if i < 10 or i % 50 == 0:  # Show first 10 and every 50th
            print(f"[{i+1}/{len(data)}] GT={result['ground_truth']} Pred={result['prediction']} {status}")

    # Finalize
    saver.finalize(len(data), total_correct)

    # Close client
    await client.close()

    # Print summary
    accuracy = total_correct / len(data) if data else 0
    print(f"\n{'='*70}")
    print(f"ProverQA {difficulty.upper()} Results ({args.model})")
    print(f"{'='*70}")
    print(f"Total: {len(data)}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2%}")
    if errors > 0:
        print(f"Errors: {errors}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main_async())
