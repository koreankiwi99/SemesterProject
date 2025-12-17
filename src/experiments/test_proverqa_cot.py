#!/usr/bin/env python3
"""
ProverQA CoT evaluation using GPTBatcher for efficient batch processing.
"""

import argparse
import os

from gpt_batch.batcher import GPTBatcher

from utils.prompts import load_prompt
from utils.answer_parsing import parse_proverqa_answer
from utils.savers import ProverQASaver
from datasets.proverqa import load_proverqa, build_proverqa_prompt


def main():
    parser = argparse.ArgumentParser(
        description='ProverQA CoT evaluation with GPTBatcher'
    )
    parser.add_argument('--api_key', help='API key (uses OPENAI_API_KEY env var if not provided)')
    parser.add_argument('--data_file', required=True, help='Path to ProverQA JSON file (easy.json, medium.json, or hard.json)')
    parser.add_argument('--system_prompt', default='prompts/proverqa/cot_system.txt',
                        help='Path to system prompt file')
    parser.add_argument('--user_prompt', default='prompts/proverqa/cot_user.txt',
                        help='Path to user prompt template file')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    parser.add_argument('--output_dir', default='results/proverqa', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--num_questions', type=int, default=0,
                        help='Number of questions to test (0 = all)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per request in seconds')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key required. Set OPENAI_API_KEY or use --api_key")

    # Load prompts
    print(f"Loading prompts...")
    system_template = load_prompt(args.system_prompt)
    user_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    # Load dataset
    data = load_proverqa(args.data_file)

    # Get difficulty from filename
    difficulty = os.path.basename(args.data_file).replace('.json', '')

    # Limit questions if specified
    if args.num_questions > 0:
        data = data[:args.num_questions]

    print(f"\nTesting {len(data)} questions from ProverQA {difficulty}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.num_workers}")

    # Initialize saver
    if args.resume:
        saver = ProverQASaver(args.output_dir, difficulty, args.model, resume_dir=args.resume)
    else:
        saver = ProverQASaver(args.output_dir, difficulty, args.model)

    # Build prompts
    print("\nBuilding prompts...")
    prompts = []
    for example in data:
        _, user_prompt = build_proverqa_prompt(example, system_template, user_template)
        prompts.append(user_prompt)

    # Initialize GPTBatcher
    print(f"\nInitializing GPTBatcher...")
    batcher = GPTBatcher(
        api_key=api_key,
        model_name=args.model,
        system_prompt=system_template,
        temperature=args.temperature,
        num_workers=args.num_workers,
        timeout_duration=args.timeout,
        retry_attempts=3
    )

    # Process all prompts
    print(f"\nProcessing {len(prompts)} prompts...")
    responses = batcher.handle_message_list(prompts)

    # Parse results and save
    print("\nParsing responses and saving results...")
    total_correct = 0

    for i, (example, response) in enumerate(zip(data, responses)):
        prediction = parse_proverqa_answer(response)
        ground_truth = example['answer']
        is_correct = prediction == ground_truth

        if is_correct:
            total_correct += 1

        result = {
            'id': example['id'],
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': is_correct,
            'context': example['context'],
            'question': example['question'],
            'options': example['options'],
            'model': args.model,
            'model_response': response if response else 'ERROR: No response'
        }

        if not response:
            result['error'] = 'No response from model'

        saver.save_result(result, i, len(data))

        # Progress indicator
        status = '✓' if is_correct else '✗'
        print(f"[{i+1}/{len(data)}] GT={ground_truth} Pred={prediction} {status}")

    # Finalize
    saver.finalize(len(data), total_correct)

    # Print summary
    accuracy = total_correct / len(data) if data else 0
    print(f"\n{'='*70}")
    print(f"ProverQA {difficulty.upper()} Results ({args.model})")
    print(f"{'='*70}")
    print(f"Total: {len(data)}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*70}")

    # Check for failed requests
    miss_indices = batcher.get_miss_index()
    if miss_indices:
        print(f"\nWarning: {len(miss_indices)} requests failed")
        print(f"Failed indices: {miss_indices[:20]}{'...' if len(miss_indices) > 20 else ''}")


if __name__ == "__main__":
    main()
