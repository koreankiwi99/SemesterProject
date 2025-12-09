#!/usr/bin/env python3
"""
Async version of FOLIO CoT evaluation with concurrent story processing.
"""

import argparse
import json
import os
import asyncio

from utils.prompts import load_prompt
from utils.api_client import create_client, get_provider
from utils.answer_parsing import parse_folio_multiple_answers, normalize_answer
from utils.savers import FOLIOSaver
from datasets.folio import load_and_group_folio, build_folio_prompt_grouped


async def test_story_async(story_examples, client, system_prompt_template, user_prompt_template,
                           model="gpt-4o", model_config=None, semaphore=None):
    """Test model on grouped questions from a single story (async).

    Args:
        story_examples: List of FOLIO examples from the same story
        client: AsyncOpenAI client
        system_prompt_template: System prompt template
        user_prompt_template: User prompt template
        model: Model name to use
        model_config: Optional model configuration dict
        semaphore: Asyncio semaphore for rate limiting

    Returns:
        dict: Results for this story including all questions
    """
    if semaphore:
        async with semaphore:
            return await _test_story_impl(story_examples, client, system_prompt_template,
                                         user_prompt_template, model, model_config)
    else:
        return await _test_story_impl(story_examples, client, system_prompt_template,
                                     user_prompt_template, model, model_config)


async def _test_story_impl(story_examples, client, system_prompt_template, user_prompt_template,
                           model, model_config):
    """Implementation of async story testing."""
    system_msg, user_prompt = build_folio_prompt_grouped(story_examples, system_prompt_template,
                                                         user_prompt_template)

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
        answers = parse_folio_multiple_answers(model_response, len(story_examples))

        results = []
        for i, (example, prediction) in enumerate(zip(story_examples, answers)):
            correct = normalize_answer(prediction, answer_format="true_false") == normalize_answer(example["label"], answer_format="true_false")
            results.append({
                'question_num': i + 1,
                'example_id': example.get('example_id'),
                'conclusion': example['conclusion'],
                'ground_truth': example['label'],
                'prediction': prediction,
                'correct': correct
            })

        return {
            'story_id': story_examples[0].get('story_id'),
            'premises': story_examples[0]['premises'],
            'system_prompt': system_msg,
            'user_prompt': user_prompt,
            'model': model,
            'model_config': model_config or {},
            'model_response': model_response,
            'results': results,
            'story_accuracy': sum(r['correct'] for r in results) / len(results)
        }

    except Exception as e:
        return {
            'story_id': story_examples[0].get('story_id'),
            'error': str(e)
        }


async def main_async():
    parser = argparse.ArgumentParser(
        description='Async FOLIO CoT evaluation with concurrent processing'
    )
    parser.add_argument('--api_key', help='API key (uses env var if not provided)')
    parser.add_argument('--base_url', help='Override API base URL (e.g., https://openrouter.ai/api/v1)')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files')
    parser.add_argument('--num_stories', type=int, default=5,
                        help='Number of stories to test (0 or negative = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-4o', help='Model to use')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')
    parser.add_argument('--rerun_errors', action='store_true',
                        help='When used with --resume, only rerun stories that had errors')
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
    system_prompt_template = load_prompt(args.system_prompt)
    user_prompt_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    grouped_data = load_and_group_folio(args.folio_file)

    # Select stories to test
    if args.num_stories > 0:
        story_ids = list(grouped_data.keys())[:args.num_stories]
    else:
        story_ids = list(grouped_data.keys())

    # Handle resume functionality
    processed_story_ids = set()
    error_story_ids = set()
    error_story_indices = {}  # Map story_id -> index in results
    previous_results = []
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            import json as json_module
            with open(all_results_file, 'r') as f:
                previous_results = json_module.load(f)

            # Identify successful and error stories
            for idx, r in enumerate(previous_results):
                story_id = r.get('story_id')
                if story_id:
                    if 'error' in r:
                        error_story_ids.add(story_id)
                        error_story_indices[story_id] = idx
                    else:
                        processed_story_ids.add(story_id)

            if args.rerun_errors:
                # Only rerun stories with errors
                print(f"Found {len(error_story_ids)} stories with errors to rerun")
                print(f"Error story IDs: {sorted(error_story_ids)}")
            else:
                # Normal resume - skip all processed (including errors)
                processed_story_ids.update(error_story_ids)
                print(f"Found {len(processed_story_ids)} already processed stories")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found: {all_results_file}")

    # Filter stories based on mode
    if args.rerun_errors and args.resume:
        # Only rerun error stories
        remaining_story_ids = [sid for sid in story_ids if sid in error_story_ids]
    else:
        # Skip already processed stories
        remaining_story_ids = [sid for sid in story_ids if sid not in processed_story_ids]

    print(f"\nTesting {len(remaining_story_ids)} stories with {args.model}")
    print(f"Concurrency: {args.concurrency} stories in parallel")
    if args.rerun_errors:
        print(f"(Rerunning {len(remaining_story_ids)} stories that had errors)")
    elif processed_story_ids:
        print(f"(Skipping {len(processed_story_ids)} already completed stories)")

    # Initialize saver (will resume if --resume provided)
    if args.resume:
        saver = FOLIOSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = FOLIOSaver(args.output_dir, args.prompt_name)

    # Initialize unified async client (supports OpenAI, OpenRouter, DeepSeek, etc.)
    provider = get_provider(args.model)
    if args.base_url:
        print(f"Using custom base URL: {args.base_url}")
    else:
        print(f"Using provider: {provider}")
    client = create_client(api_key=args.api_key, model=args.model, base_url=args.base_url)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    total_questions = 0
    total_correct = 0

    try:
        # Process stories in batches
        for batch_start in range(0, len(remaining_story_ids), args.concurrency):
            batch_story_ids = remaining_story_ids[batch_start:batch_start + args.concurrency]

            print(f"\nProcessing batch {batch_start // args.concurrency + 1} "
                  f"({len(batch_story_ids)} stories)...")

            # Create tasks for this batch
            tasks = [
                test_story_async(
                    grouped_data[story_id],
                    client,
                    system_prompt_template,
                    user_prompt_template,
                    args.model,
                    model_config,
                    semaphore
                )
                for story_id in batch_story_ids
            ]

            # Run batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Save results with lock
            for i, (story_id, result) in enumerate(zip(batch_story_ids, results)):
                story_index = len(processed_story_ids) + batch_start + i
                total_stories = len(story_ids)

                if isinstance(result, Exception):
                    result = {'story_id': story_id, 'error': str(result)}

                # Thread-safe save with lock
                async with saver._save_lock:
                    if args.rerun_errors and story_id in error_story_indices:
                        # Update existing entry in place
                        original_index = error_story_indices[story_id]
                        saver.update_result(result, original_index)
                    else:
                        # Append new result
                        saver.save_result(result, story_index, total_stories)

                if 'error' in result:
                    print(f"Story {story_index + 1}/{total_stories} (ID: {story_id}): ERROR - {result['error']}")
                    continue

                print(f"Story {story_index + 1}/{total_stories} (ID: {story_id}): "
                      f"{result['story_accuracy']:.2%}")
                for q_result in result['results']:
                    print(f"  Q{q_result['question_num']}: {q_result['ground_truth']} → "
                          f"{q_result['prediction']} {'✓' if q_result['correct'] else '✗'}")
                    total_questions += 1
                    if q_result['correct']:
                        total_correct += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}. Saving results so far...")
        import traceback
        traceback.print_exc()

    saver.finalize(total_questions, total_correct)

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\nOverall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
    else:
        print("\nNo questions were completed successfully.")


if __name__ == "__main__":
    asyncio.run(main_async())
