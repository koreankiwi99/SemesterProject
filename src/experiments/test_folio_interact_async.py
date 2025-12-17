#!/usr/bin/env python3
"""
Async version of FOLIO Lean evaluation with concurrent question processing.
"""

import argparse
import json
import os
import asyncio
from openai import AsyncOpenAI

from utils.prompts import load_prompt
from utils.answer_parsing import parse_folio_answer, normalize_answer
from utils.lean_utils import extract_lean_code, create_lean_server
from utils.savers import FOLIOLeanSaver
from datasets.folio import load_folio, build_folio_prompt_single


async def verify_with_lean_async(lean_code, lean_server, verbose=False):
    """Async version of Lean verification."""
    try:
        if verbose:
            print(f"\nVerifying Lean code:\n{lean_code}\n")

        from lean_interact import Command
        response = await lean_server.async_run(Command(cmd=lean_code))

        messages = response.messages if hasattr(response, 'messages') else []
        errors = [msg for msg in messages if msg.severity == 'error']
        warnings = [msg for msg in messages if msg.severity == 'warning']

        success = len(errors) == 0

        result = {
            'success': success,
            'env': response.env if hasattr(response, 'env') else None,
            'errors': [msg.data for msg in errors],
            'warnings': [msg.data for msg in warnings],
            'all_messages': [{'severity': msg.severity, 'data': msg.data} for msg in messages]
        }

        if verbose:
            print(f"Verification {'succeeded' if success else 'failed'}")
            if errors:
                print(f"Errors: {errors}")

        return result

    except Exception as e:
        return {
            'success': False,
            'env': None,
            'errors': [str(e)],
            'warnings': [],
            'all_messages': []
        }


async def test_question_with_lean_async(example, client, lean_server, system_template, user_template,
                                        model="gpt-4o", model_config=None, max_iterations=3,
                                        verbose=False, semaphore=None):
    """Test a single question with interactive Lean verification (async).

    Args:
        example: FOLIO example dictionary
        client: AsyncOpenAI client
        lean_server: LeanServer instance
        system_template: System prompt template
        user_template: User prompt template
        model: Model name to use
        model_config: Optional model configuration dict
        max_iterations: Maximum number of refinement iterations
        verbose: Whether to print verbose output
        semaphore: Asyncio semaphore for rate limiting

    Returns:
        dict: Result dictionary with iterations and verification info
    """
    if semaphore:
        async with semaphore:
            return await _test_question_impl(example, client, lean_server, system_template,
                                            user_template, model, model_config, max_iterations, verbose)
    else:
        return await _test_question_impl(example, client, lean_server, system_template,
                                        user_template, model, model_config, max_iterations, verbose)


async def _test_question_impl(example, client, lean_server, system_template, user_template,
                              model, model_config, max_iterations, verbose):
    """Implementation of async question testing with Lean."""
    system_msg, user_prompt = build_folio_prompt_single(example, system_template, user_template)

    conversation_history = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ]

    iterations = []
    final_prediction = 'Unknown'
    final_lean_code = None
    final_verification = None

    # Build API call parameters base
    api_params_base = {"model": model}
    if model_config:
        api_params_base.update(model_config)

    try:
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Get LLM response
            api_params = {**api_params_base, "messages": conversation_history}
            response = await client.chat.completions.create(**api_params)

            llm_response = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": llm_response})

            # Extract answer and Lean code
            prediction = parse_folio_answer(llm_response)
            lean_code = extract_lean_code(llm_response)

            iteration_data = {
                'iteration': iteration + 1,
                'llm_response': llm_response,
                'prediction': prediction,
                'lean_code': lean_code,
                'lean_verification': None
            }

            # Verify Lean code if present
            if lean_code:
                lean_verification = await verify_with_lean_async(lean_code, lean_server, verbose)
                iteration_data['lean_verification'] = lean_verification

                if lean_verification['success']:
                    # Success! Use this result
                    if verbose:
                        print(f"✓ Lean verification successful on iteration {iteration + 1}")
                    final_prediction = prediction
                    final_lean_code = lean_code
                    final_verification = lean_verification
                    iterations.append(iteration_data)
                    break
                else:
                    # Failed - provide feedback for next iteration
                    if verbose:
                        print(f"✗ Lean verification failed on iteration {iteration + 1}")

                    if iteration < max_iterations - 1:
                        error_messages = '\n'.join(lean_verification['errors'])
                        feedback = (
                            f"The Lean code has compilation errors:\n\n"
                            f"{error_messages}\n\n"
                            f"Please provide corrected Lean code wrapped in <lean></lean> tags:\n\n"
                            f"<lean>\n"
                            f"[your corrected code here]\n"
                            f"</lean>\n\n"
                            f"Then provide your answer:\n"
                            f"ANSWER: True/False/Unknown"
                        )
                        conversation_history.append({"role": "user", "content": feedback})
                        if verbose:
                            print(f"Sending feedback to LLM...")
            else:
                # No Lean code found
                if verbose:
                    print(f"No Lean code found in iteration {iteration + 1}")

                # Prompt for Lean code if missing
                if iteration < max_iterations - 1:
                    feedback = (
                        f"I didn't find any Lean code in your response. "
                        f"Please provide your Lean translation wrapped in <lean></lean> tags:\n\n"
                        f"<lean>\n"
                        f"[your Lean code here]\n"
                        f"</lean>\n\n"
                        f"Then provide your answer:\n"
                        f"ANSWER: True/False/Unknown"
                    )
                    conversation_history.append({"role": "user", "content": feedback})
                    if verbose:
                        print(f"Prompting for Lean code...")

            iterations.append(iteration_data)

            # If no Lean code or last iteration, use current prediction
            if iteration == max_iterations - 1:
                final_prediction = prediction
                final_lean_code = lean_code
                final_verification = lean_verification if lean_code else None

        correct = normalize_answer(final_prediction, answer_format="true_false") == normalize_answer(example["label"], answer_format="true_false")

        return {
            'example_id': example.get('example_id'),
            'story_id': example.get('story_id'),
            'premises': example['premises'],
            'conclusion': example['conclusion'],
            'ground_truth': example['label'],
            'prediction': final_prediction,
            'correct': correct,
            'model': model,
            'model_config': model_config or {},
            'iterations': iterations,
            'num_iterations': len(iterations),
            'lean_code': final_lean_code,
            'lean_verification': final_verification,
            'conversation_history': conversation_history
        }

    except Exception as e:
        return {
            'example_id': example.get('example_id'),
            'story_id': example.get('story_id'),
            'error': str(e),
            'iterations': iterations
        }


async def main_async():
    parser = argparse.ArgumentParser(
        description='Async FOLIO Lean evaluation with concurrent processing'
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='lean_test', help='Name for output files')
    parser.add_argument('--num_questions', type=int, default=5,
                        help='Number of questions to test (0 or negative = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-4o', help='Model to use')
    parser.add_argument('--lean_version', default=None, help='Lean version (None = latest)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Max iterations for Lean verification')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Number of concurrent questions (default: 5)')

    # Model configuration options
    parser.add_argument('--temperature', type=float, help='Sampling temperature (0.0-2.0)')
    parser.add_argument('--reasoning_effort', type=str, choices=['low', 'medium', 'high'],
                        help='Reasoning effort (for models like o1/o3)')
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

    # Initialize Lean server
    print(f"\nInitializing Lean server...")
    lean_server = create_lean_server(args.lean_version, args.verbose)
    print(f"✓ Lean server initialized")

    # Load FOLIO data
    all_questions = load_folio(args.folio_file)

    # Select questions to test
    if args.num_questions > 0:
        questions_to_test = all_questions[:args.num_questions]
    else:
        questions_to_test = all_questions

    # Handle resume functionality
    processed_questions = set()
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            with open(all_results_file, 'r') as f:
                previous_results = json.load(f)
            # Track by (story_id, example_id) tuple
            processed_questions = {(r['story_id'], r['example_id']) for r in previous_results
                                  if 'story_id' in r and 'example_id' in r}
            print(f"Found {len(processed_questions)} already processed questions")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found: {all_results_file}")

    # Filter out already processed questions
    remaining_questions = [q for q in questions_to_test
                          if (q.get('story_id'), q.get('example_id')) not in processed_questions]

    print(f"\nTesting {len(remaining_questions)} questions with {args.model}")
    print(f"Concurrency: {args.concurrency} questions in parallel")
    if processed_questions:
        print(f"(Skipping {len(processed_questions)} already completed questions)")
    print(f"Max iterations per question: {args.max_iterations}")
    print(f"Interactive Lean verification: ENABLED\n")

    # Initialize saver (will resume if --resume provided)
    if args.resume:
        saver = FOLIOLeanSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = FOLIOLeanSaver(args.output_dir, args.prompt_name)

    # Initialize OpenAI async client
    client = AsyncOpenAI(api_key=args.api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    total_questions = 0
    total_correct = 0
    total_iterations = 0
    lean_stats = {'with_code': 0, 'successful': 0, 'failed': 0, 'avg_iterations': 0}

    try:
        # Process questions in batches
        for batch_start in range(0, len(remaining_questions), args.concurrency):
            batch_questions = remaining_questions[batch_start:batch_start + args.concurrency]

            print(f"\nProcessing batch {batch_start // args.concurrency + 1} "
                  f"({len(batch_questions)} questions)...")

            # Create tasks for this batch
            tasks = [
                test_question_with_lean_async(
                    question,
                    client,
                    lean_server,
                    system_template,
                    user_template,
                    args.model,
                    model_config,
                    args.max_iterations,
                    args.verbose,
                    semaphore
                )
                for question in batch_questions
            ]

            # Run batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Save results with lock
            for i, (question, result) in enumerate(zip(batch_questions, results)):
                question_index = len(processed_questions) + batch_start + i
                total_count = len(questions_to_test)

                if isinstance(result, Exception):
                    result = {
                        'example_id': question.get('example_id'),
                        'story_id': question.get('story_id'),
                        'error': str(result)
                    }

                # Thread-safe save with lock
                async with saver._save_lock:
                    saver.save_result(result, question_index, total_count)

                if 'error' in result:
                    print(f"Question {question_index + 1}/{total_count}: ERROR - {result['error']}")
                    continue

                print(f"Question {question_index + 1}/{total_count} "
                      f"(Story: {result.get('story_id')}, ID: {result.get('example_id')})")
                print(f"Result: {result['ground_truth']} → {result['prediction']} "
                      f"{'✓' if result['correct'] else '✗'}")
                print(f"Iterations: {result['num_iterations']}")

                total_iterations += result['num_iterations']

                # Track Lean verification stats
                if result.get('lean_code'):
                    lean_stats['with_code'] += 1
                    if result.get('lean_verification'):
                        if result['lean_verification']['success']:
                            lean_stats['successful'] += 1
                            print(f"Lean verification: ✓ SUCCESS")
                        else:
                            lean_stats['failed'] += 1
                            print(f"Lean verification: ✗ FAILED")

                total_questions += 1
                if result['correct']:
                    total_correct += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}. Saving results so far...")
        import traceback
        traceback.print_exc()

    if total_questions > 0:
        lean_stats['avg_iterations'] = total_iterations / total_questions

    saver.finalize(total_questions, total_correct, lean_stats)

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\nOverall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
        print(f"Average iterations per question: {lean_stats['avg_iterations']:.2f}")

        if lean_stats['with_code'] > 0:
            print(f"\nLean Verification Summary:")
            print(f"  Questions with Lean code: {lean_stats['with_code']}")
            print(f"  Successful verifications: {lean_stats['successful']}")
            print(f"  Failed verifications: {lean_stats['failed']}")
            verification_rate = lean_stats['successful'] / lean_stats['with_code']
            print(f"  Verification success rate: {verification_rate:.2%}")
    else:
        print("\nNo questions were completed successfully.")


if __name__ == "__main__":
    asyncio.run(main_async())
