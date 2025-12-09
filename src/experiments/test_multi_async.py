#!/usr/bin/env python3
"""
Async version of Multi-LogiEval CoT evaluation with concurrent processing.
"""

import argparse
import asyncio
import os

from utils.prompts import load_prompt
from utils.api_client import create_client, get_provider
from utils.answer_parsing import parse_multilogieval_answer, normalize_answer
from utils.savers import MultiLogiEvalSaver
from datasets.multilogieval import load_and_sample_multilogieval, build_multilogieval_prompt
from collections import defaultdict


async def test_question_async(example, client, system_template, user_template,
                              model="gpt-4o", model_config=None, semaphore=None):
    """Test a single Multi-LogiEval question (async).

    Args:
        example: Multi-LogiEval example dictionary
        client: AsyncOpenAI client
        system_template: System prompt template
        user_template: User prompt template
        model: Model name to use
        model_config: Optional model configuration dict
        semaphore: Asyncio semaphore for rate limiting

    Returns:
        dict: Result dictionary
    """
    if semaphore:
        async with semaphore:
            return await _test_question_impl(example, client, system_template, user_template,
                                            model, model_config)
    else:
        return await _test_question_impl(example, client, system_template, user_template,
                                        model, model_config)


async def _test_question_impl(example, client, system_template, user_template, model, model_config):
    """Implementation of async question testing."""
    system_msg, user_prompt = build_multilogieval_prompt(example, system_template, user_template)

    # Build API call parameters
    api_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    }

    # Add model config if provided
    if model_config:
        api_params.update(model_config)

    try:
        response = await client.chat.completions.create(**api_params)

        model_response = response.choices[0].message.content.strip()
        prediction = parse_multilogieval_answer(model_response)
        correct = normalize_answer(prediction) == normalize_answer(example["answer"])

        return {
            'logic_type': example['logic_type'],
            'depth': example['depth'],
            'depth_dir': example['depth_dir'],
            'rule': example['rule'],
            'context': example['context'],
            'question': example['question'],
            'ground_truth': example['answer'],
            'prediction': prediction,
            'correct': correct,
            'model': model,
            'model_config': model_config or {},
            'system_prompt': system_msg,
            'user_prompt': user_prompt,
            'model_response': model_response
        }

    except Exception as e:
        return {
            'logic_type': example.get('logic_type'),
            'depth': example.get('depth'),
            'depth_dir': example.get('depth_dir'),
            'rule': example.get('rule'),
            'error': str(e)
        }


async def main_async():
    parser = argparse.ArgumentParser(
        description='Async Multi-LogiEval CoT evaluation with concurrent processing'
    )
    parser.add_argument('--api_key', help='API key (uses env var if not provided)')
    parser.add_argument('--base_url', help='Override API base URL (e.g., https://openrouter.ai/api/v1)')
    parser.add_argument('--data_dir', required=True, help='Multi-LogiEval data directory')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files')
    parser.add_argument('--num_questions', type=int, default=10,
                        help='Number of questions to test per (logic, depth) combo (0 = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-4o', help='Model to use')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')
    parser.add_argument('--rerun_errors', action='store_true',
                        help='When used with --resume, only rerun questions that had errors')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent requests (default: 10)')

    # Model configuration options
    parser.add_argument('--temperature', type=float, help='Sampling temperature (0.0-2.0)')
    parser.add_argument('--reasoning_effort', type=str, choices=['none', 'minimal', 'low', 'medium', 'high'],
                        help='Reasoning effort (for models like gpt-5/o1/o3). Use "none" to disable reasoning.')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens in response')
    parser.add_argument('--top_p', type=float, help='Nucleus sampling parameter')

    args = parser.parse_args()

    # Build model config from args
    model_config = {}
    if args.temperature is not None:
        model_config['temperature'] = args.temperature
    if args.reasoning_effort is not None:
        model_config['reasoning_effort'] = args.reasoning_effort
    if args.max_tokens is not None:
        model_config['max_tokens'] = args.max_tokens
    if args.top_p is not None:
        model_config['top_p'] = args.top_p

    # Load prompts
    print(f"Loading prompts...")
    system_template = load_prompt(args.system_prompt)
    user_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    # Load Multi-LogiEval data
    all_questions = load_and_sample_multilogieval(args.data_dir, samples_per_combination=args.num_questions)

    # Handle resume functionality
    error_indices = set()  # Indices of questions with errors
    previous_results = []
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            import json as json_module
            with open(all_results_file, 'r') as f:
                previous_results = json_module.load(f)

            # Identify error indices
            for idx, r in enumerate(previous_results):
                if 'error' in r:
                    error_indices.add(idx)

            if args.rerun_errors:
                print(f"Found {len(error_indices)} questions with errors to rerun")
                print(f"Error indices: {sorted(error_indices)}")
            else:
                # Skip already processed (not supported - Multi-LogiEval loads all questions fresh)
                print(f"Note: Normal resume mode not fully supported for Multi-LogiEval")
                print(f"Use --rerun_errors to rerun only failed questions")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found: {all_results_file}")

    # Determine which questions to process
    if args.rerun_errors and args.resume and error_indices:
        questions_to_process = [(idx, all_questions[idx]) for idx in sorted(error_indices) if idx < len(all_questions)]
        print(f"\nRerunning {len(questions_to_process)} questions with {args.model}")
    else:
        questions_to_process = [(idx, q) for idx, q in enumerate(all_questions)]
        print(f"\nTesting {len(questions_to_process)} questions with {args.model}")

    print(f"Concurrency: {args.concurrency} questions in parallel")

    # Initialize saver
    if args.resume:
        saver = MultiLogiEvalSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = MultiLogiEvalSaver(args.output_dir, args.prompt_name)

    # Initialize unified async client (supports OpenAI, OpenRouter, DeepSeek, etc.)
    provider = get_provider(args.model)
    if args.base_url:
        print(f"Using custom base URL: {args.base_url}")
    else:
        print(f"Using provider: {provider}")
    client = create_client(api_key=args.api_key, model=args.model, base_url=args.base_url)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    results = []
    results_by_combination = defaultdict(list)

    is_rerun_mode = args.rerun_errors and args.resume and error_indices

    try:
        # Process questions in batches
        for batch_start in range(0, len(questions_to_process), args.concurrency):
            batch_items = questions_to_process[batch_start:batch_start + args.concurrency]

            print(f"\nProcessing batch {batch_start // args.concurrency + 1} "
                  f"({len(batch_items)} questions)...")

            # Create tasks for this batch
            tasks = [
                test_question_async(
                    question,
                    client,
                    system_template,
                    user_template,
                    args.model,
                    model_config,
                    semaphore
                )
                for _, question in batch_items
            ]

            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Save results with lock
            for i, ((original_idx, question), result) in enumerate(zip(batch_items, batch_results)):
                display_idx = batch_start + i

                if isinstance(result, Exception):
                    result = {
                        'logic_type': question.get('logic_type'),
                        'depth': question.get('depth'),
                        'depth_dir': question.get('depth_dir'),
                        'rule': question.get('rule'),
                        'context': question.get('context', ''),
                        'question': question.get('question', ''),
                        'error': str(result)
                    }

                # Thread-safe save with lock
                async with saver._save_lock:
                    if is_rerun_mode:
                        # Update existing entry in place
                        saver.update_result(result, original_idx)
                    else:
                        saver.save_result(result, original_idx, len(all_questions))

                if 'error' in result:
                    print(f"Question {display_idx + 1}/{len(questions_to_process)} (idx {original_idx}): "
                          f"ERROR - {result['error']}")
                else:
                    print(f"Question {display_idx + 1}/{len(questions_to_process)} (idx {original_idx}): "
                          f"{result['logic_type']}/{result['depth_dir']}/{result['rule']} "
                          f"→ {result['prediction']} {'✓' if result['correct'] else '✗'}")

                    results.append(result)
                    key = (result['logic_type'], result['depth_dir'])
                    results_by_combination[key].append(result)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}. Saving results so far...")
        import traceback
        traceback.print_exc()

    saver.finalize(results, results_by_combination)

    # Print summary
    if results:
        total_correct = sum(r['correct'] for r in results if 'error' not in r)
        total_questions = len([r for r in results if 'error' not in r])
        if total_questions > 0:
            overall_accuracy = total_correct / total_questions
            print(f"\nOverall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
    else:
        print("\nNo questions were completed successfully.")


if __name__ == "__main__":
    asyncio.run(main_async())
