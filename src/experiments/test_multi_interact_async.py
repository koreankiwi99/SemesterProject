#!/usr/bin/env python3
"""
Async version of Multi-LogiEval Lean evaluation with concurrent processing.
"""

import argparse
import asyncio
from openai import AsyncOpenAI
from collections import defaultdict

from utils.prompts import load_prompt
from utils.answer_parsing import parse_multilogieval_answer, normalize_answer
from utils.lean_utils import extract_lean_code, create_lean_server
from utils.savers import MultiLogiEvalLeanSaver
from datasets.multilogieval import load_and_sample_multilogieval, build_multilogieval_prompt


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
    """Test a single question with interactive Lean verification (async)."""
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
    system_msg, user_prompt = build_multilogieval_prompt(example, system_template, user_template)

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
            prediction = parse_multilogieval_answer(llm_response)
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
                    if verbose:
                        print(f"✓ Lean verification successful on iteration {iteration + 1}")
                    final_prediction = prediction
                    final_lean_code = lean_code
                    final_verification = lean_verification
                    iterations.append(iteration_data)
                    break
                else:
                    if verbose:
                        print(f"✗ Lean verification failed on iteration {iteration + 1}")

                    if iteration < max_iterations - 1:
                        error_messages = '\n'.join(lean_verification['errors'])
                        feedback = (
                            f"The Lean code has compilation errors:\n\n"
                            f"{error_messages}\n\n"
                            f"Please provide corrected Lean code wrapped in <lean></lean> tags."
                        )
                        conversation_history.append({"role": "user", "content": feedback})
            else:
                if verbose:
                    print(f"No Lean code found in iteration {iteration + 1}")

                if iteration < max_iterations - 1:
                    feedback = "Please provide your Lean translation wrapped in <lean></lean> tags."
                    conversation_history.append({"role": "user", "content": feedback})

            iterations.append(iteration_data)

            if iteration == max_iterations - 1:
                final_prediction = prediction
                final_lean_code = lean_code
                final_verification = lean_verification if lean_code else None

        correct = normalize_answer(final_prediction) == normalize_answer(example["answer"])

        return {
            'logic_type': example['logic_type'],
            'depth': example['depth'],
            'depth_dir': example['depth_dir'],
            'rule': example['rule'],
            'context': example['context'],
            'question': example['question'],
            'ground_truth': example['answer'],
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
            'logic_type': example.get('logic_type'),
            'depth': example.get('depth'),
            'depth_dir': example.get('depth_dir'),
            'rule': example.get('rule'),
            'error': str(e),
            'iterations': iterations
        }


async def main_async():
    parser = argparse.ArgumentParser(
        description='Async Multi-LogiEval Lean evaluation with concurrent processing'
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_dir', required=True, help='Multi-LogiEval data directory')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='lean_test', help='Name for output files')
    parser.add_argument('--num_questions', type=int, default=10,
                        help='Number of questions to test per (logic, depth) combo (0 = all)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-4o', help='Model to use')
    parser.add_argument('--lean_version', default=None, help='Lean version (None = latest)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Max iterations for Lean verification')
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

    # Load Multi-LogiEval data
    all_questions = load_and_sample_multilogieval(args.data_dir, samples_per_combination=args.num_questions)

    print(f"\nTesting {len(all_questions)} questions with {args.model}")
    print(f"Concurrency: {args.concurrency} questions in parallel")
    print(f"Max iterations per question: {args.max_iterations}")
    print(f"Interactive Lean verification: ENABLED\n")

    # Initialize saver
    saver = MultiLogiEvalLeanSaver(args.output_dir, args.prompt_name)

    # Initialize OpenAI async client
    client = AsyncOpenAI(api_key=args.api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    results = []
    results_by_combination = defaultdict(list)

    try:
        # Process questions in batches
        for batch_start in range(0, len(all_questions), args.concurrency):
            batch_questions = all_questions[batch_start:batch_start + args.concurrency]

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
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Save results with lock
            for i, (question, result) in enumerate(zip(batch_questions, batch_results)):
                question_index = batch_start + i

                if isinstance(result, Exception):
                    result = {
                        'logic_type': question.get('logic_type'),
                        'depth': question.get('depth'),
                        'depth_dir': question.get('depth_dir'),
                        'rule': question.get('rule'),
                        'error': str(result)
                    }

                # Thread-safe save with lock
                async with saver._save_lock:
                    saver.save_result(result, question_index, len(all_questions))

                if 'error' in result:
                    print(f"Question {question_index + 1}/{len(all_questions)}: "
                          f"ERROR - {result['error']}")
                else:
                    print(f"Question {question_index + 1}/{len(all_questions)}: "
                          f"{result['logic_type']}/{result['depth_dir']}/{result['rule']} "
                          f"→ {result['prediction']} {'✓' if result['correct'] else '✗'} "
                          f"(Lean: {'✓' if result.get('lean_verification', {}).get('success') else '✗'})")

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

            lean_stats = {
                'with_code': sum(1 for r in results if r.get('lean_code')),
                'successful': sum(1 for r in results if r.get('lean_verification', {}).get('success')),
            }
            if lean_stats['with_code'] > 0:
                print(f"Lean verification rate: {lean_stats['successful']}/{lean_stats['with_code']} "
                      f"({lean_stats['successful']/lean_stats['with_code']:.2%})")
    else:
        print("\nNo questions were completed successfully.")


if __name__ == "__main__":
    asyncio.run(main_async())
