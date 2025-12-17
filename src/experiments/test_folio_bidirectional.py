#!/usr/bin/env python3
"""
Bidirectional Verification for FOLIO dataset (Async version).

Novel approach: Try to prove BOTH the conclusion AND its negation in parallel.
- If only TRUE proof succeeds → Answer is TRUE
- If only FALSE proof succeeds → Answer is FALSE
- If BOTH succeed → Formalization error detected, fall back to CoT
- If NEITHER succeeds → Answer is UNCERTAIN (or fall back to CoT)

This catches the "axiomatizes conclusion" problem automatically.
"""

import argparse
import json
import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from utils.prompts import load_prompt
from utils.answer_parsing import parse_folio_answer, normalize_answer
from utils.lean_utils import extract_lean_code, verify_with_lean_async, create_lean_server
from utils.savers import BidirectionalSaver
from datasets.folio import load_folio

async def get_proof_attempt_async(example, client, system_prompt, user_template,
                                   model, model_config, direction="true",
                                   conversation_history=None):
    """Get a single proof attempt from the LLM (async).

    Args:
        direction: "true" or "false" - which direction we're proving
        conversation_history: Optional list of previous messages for iteration

    Returns:
        dict with keys: response, lean_code, proof_status, messages
    """
    if conversation_history is None:
        user_prompt = user_template.replace("{premises}", example['premises'])
        user_prompt = user_prompt.replace("{conclusion}", example['conclusion'])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = conversation_history

    api_params = {"model": model, "messages": messages}
    if model_config:
        api_params.update(model_config)

    try:
        response = await client.chat.completions.create(**api_params)
        llm_response = response.choices[0].message.content.strip()

        lean_code = extract_lean_code(llm_response)

        # Check if LLM indicated proof failed
        proof_failed_indicator = "PROOF FAILED" in llm_response.upper() or \
                                 "PROOF STATUS: FAILED" in llm_response.upper()

        # Update conversation history
        updated_messages = messages + [{"role": "assistant", "content": llm_response}]

        return {
            'direction': direction,
            'response': llm_response,
            'lean_code': lean_code,
            'proof_failed_indicator': proof_failed_indicator,
            'messages': updated_messages
        }

    except Exception as e:
        return {
            'direction': direction,
            'response': None,
            'lean_code': None,
            'error': str(e),
            'messages': messages
        }


async def get_proof_with_iteration_async(example, client, lean_server, system_prompt,
                                          user_template, model, model_config,
                                          direction="true", max_iterations=3, verbose=False):
    """Get proof attempt with iteration on compilation errors (async).

    Only retries on actual Lean compilation errors, NOT on sorry (incomplete proof).

    Returns:
        dict with final result including all iterations
    """
    iterations = []
    conversation_history = None

    for iteration in range(max_iterations):
        # Get proof attempt
        result = await get_proof_attempt_async(
            example, client, system_prompt, user_template,
            model, model_config, direction, conversation_history
        )

        if result.get('error'):
            # API error - don't retry
            return {
                'direction': direction,
                'response': result.get('response'),
                'lean_code': None,
                'proof_success': False,
                'errors': [result['error']],
                'iterations': iterations,
                'num_iterations': iteration + 1
            }

        lean_code = result.get('lean_code')

        iteration_data = {
            'iteration': iteration + 1,
            'response': result.get('response'),
            'lean_code': lean_code,
            'proof_failed_indicator': result.get('proof_failed_indicator', False)
        }

        # If no Lean code extracted, prompt for it
        if not lean_code:
            iteration_data['verification_success'] = False
            iteration_data['errors'] = ['No Lean code found in response']
            iterations.append(iteration_data)

            if iteration < max_iterations - 1:
                feedback = (
                    "I didn't find any Lean code in your response. "
                    "Please provide your Lean formalization wrapped in <lean></lean> tags:\n\n"
                    "<lean>\n[your Lean code here]\n</lean>\n\n"
                    "Then indicate if the proof succeeded or failed:\n"
                    "PROOF STATUS: SUCCESS or PROOF STATUS: FAILED"
                )
                conversation_history = result['messages'] + [{"role": "user", "content": feedback}]
                continue
            else:
                break

        # If LLM indicated proof failed, don't verify (it gave up)
        if result.get('proof_failed_indicator'):
            iteration_data['verification_success'] = False
            iteration_data['errors'] = []  # Empty = sorry/gave up, not compilation error
            iterations.append(iteration_data)
            break

        # Verify with Lean asynchronously (AutoLeanServer handles restarts automatically)
        success, errors = await verify_lean_code_async(lean_code, lean_server, verbose)
        iteration_data['verification_success'] = success
        iteration_data['errors'] = errors
        iterations.append(iteration_data)

        if success:
            # Proof verified successfully
            if verbose:
                print(f"    {direction.upper()} proof succeeded on iteration {iteration + 1}")
            break

        # Check if it's a compilation error (non-empty errors) or sorry (empty errors)
        if errors and iteration < max_iterations - 1:
            # Compilation error - provide feedback and retry
            if verbose:
                print(f"    {direction.upper()} compilation error on iteration {iteration + 1}, retrying...")

            error_messages = '\n'.join(errors)
            feedback = (
                f"The Lean code has compilation errors:\n\n"
                f"{error_messages}\n\n"
                f"Please provide corrected Lean code wrapped in <lean></lean> tags:\n\n"
                f"<lean>\n[your corrected code here]\n</lean>\n\n"
                f"Then indicate if the proof succeeded or failed:\n"
                f"PROOF STATUS: SUCCESS or PROOF STATUS: FAILED"
            )
            conversation_history = result['messages'] + [{"role": "user", "content": feedback}]
        else:
            # Either sorry (empty errors) or last iteration - stop
            if verbose and not errors:
                print(f"    {direction.upper()} proof incomplete (sorry) on iteration {iteration + 1}")
            break

    # Return final result
    final_iteration = iterations[-1] if iterations else {}
    return {
        'direction': direction,
        'response': final_iteration.get('response'),
        'lean_code': final_iteration.get('lean_code'),
        'proof_success': final_iteration.get('verification_success', False),
        'errors': final_iteration.get('errors', []),
        'iterations': iterations,
        'num_iterations': len(iterations)
    }


async def get_cot_fallback_async(example, client, model, model_config):
    """Get Chain-of-Thought answer as fallback (async)."""
    system_prompt = """You are a logical reasoning expert. Analyze the premises and determine if the conclusion is True, False, or Unknown.

Think step by step, then provide your final answer.

ANSWER FORMAT:
ANSWER: True/False/Unknown"""

    user_prompt = f"""Premises:
{example['premises']}

Conclusion:
{example['conclusion']}

Please analyze step by step and provide your answer."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    api_params = {"model": model, "messages": messages}
    if model_config:
        api_params.update(model_config)

    try:
        response = await client.chat.completions.create(**api_params)
        llm_response = response.choices[0].message.content.strip()
        answer = parse_folio_answer(llm_response)
        return answer, llm_response

    except Exception as e:
        return 'Unknown', str(e)


async def verify_lean_code_async(lean_code, lean_server, verbose=False, timeout=60.0):
    """Verify Lean code asynchronously using AutoLeanServer.async_run().

    This doesn't block the event loop, allowing better concurrency.
    AutoLeanServer handles memory management and restarts automatically.

    Returns:
        tuple: (success, errors)
    """
    if not lean_code:
        return False, []

    verification = await verify_with_lean_async(lean_code, lean_server, verbose, timeout)
    return verification['success'], verification.get('errors', [])


def apply_agreement_logic(true_success, false_success):
    """Apply bidirectional agreement logic.

    Returns:
        tuple: (predicted_answer, agreement_pattern, formalization_error)
    """
    if true_success and not false_success:
        return 'True', 'TRUE_ONLY', False
    elif false_success and not true_success:
        return 'False', 'FALSE_ONLY', False
    elif true_success and false_success:
        # Both succeeded - formalization error!
        return None, 'BOTH_SUCCESS', True
    else:
        # Neither succeeded
        return 'Unknown', 'NEITHER_SUCCESS', False


async def test_question_bidirectional_async(example, client, lean_server,
                                            true_system, false_system, user_template,
                                            model="gpt-4", model_config=None,
                                            max_iterations=3, verbose=False):
    """Test a single question with bidirectional verification (async).

    Both TRUE and FALSE proof attempts run in parallel, with iteration on compilation errors.
    """
    result = {
        'example_id': example.get('example_id'),
        'story_id': example.get('story_id'),
        'premises': example['premises'],
        'conclusion': example['conclusion'],
        'ground_truth': example['label'],
    }

    # Run both proof attempts in parallel (each with its own iteration loop)
    true_task = get_proof_with_iteration_async(
        example, client, lean_server, true_system, user_template,
        model, model_config, direction="true", max_iterations=max_iterations, verbose=verbose
    )
    false_task = get_proof_with_iteration_async(
        example, client, lean_server, false_system, user_template,
        model, model_config, direction="false", max_iterations=max_iterations, verbose=verbose
    )

    true_result, false_result = await asyncio.gather(true_task, false_task)

    # Extract TRUE proof results
    result['true_response'] = true_result.get('response')
    result['true_lean_code'] = true_result.get('lean_code')
    result['true_proof_success'] = true_result.get('proof_success', False)
    result['true_errors'] = true_result.get('errors', [])
    result['true_iterations'] = true_result.get('iterations', [])
    result['true_num_iterations'] = true_result.get('num_iterations', 1)

    # Extract FALSE proof results
    result['false_response'] = false_result.get('response')
    result['false_lean_code'] = false_result.get('lean_code')
    result['false_proof_success'] = false_result.get('proof_success', False)
    result['false_errors'] = false_result.get('errors', [])
    result['false_iterations'] = false_result.get('iterations', [])
    result['false_num_iterations'] = false_result.get('num_iterations', 1)

    # Apply agreement logic
    prediction, agreement_pattern, formalization_error = apply_agreement_logic(
        result['true_proof_success'], result['false_proof_success']
    )

    result['agreement_pattern'] = agreement_pattern
    result['formalization_error'] = formalization_error
    result['used_fallback'] = False

    # If formalization error or uncertain, fall back to CoT
    if formalization_error or prediction is None:
        if verbose:
            print(f"  Formalization error detected, falling back to CoT")
        prediction, cot_response = await get_cot_fallback_async(example, client, model, model_config)
        result['used_fallback'] = True
        result['cot_response'] = cot_response

    result['prediction'] = prediction
    result['correct'] = normalize_answer(prediction, answer_format="true_false") == \
                        normalize_answer(example['label'], answer_format="true_false")

    return result


async def process_batch_async(questions_batch, client, lean_server,
                               true_system, false_system, user_template,
                               model, model_config, semaphore,
                               max_iterations=3, verbose=False):
    """Process a batch of questions concurrently."""

    async def process_one(example):
        async with semaphore:
            return await test_question_bidirectional_async(
                example, client, lean_server,
                true_system, false_system, user_template,
                model, model_config, max_iterations, verbose
            )

    tasks = [process_one(q) for q in questions_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def main_async():
    parser = argparse.ArgumentParser(
        description='Async FOLIO Bidirectional Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_folio_bidirectional.py \\
      --api_key YOUR_KEY \\
      --folio_file data/folio_original/folio-validation.json \\
      --num_questions 50 \\
      --concurrency 5
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files')
    parser.add_argument('--num_questions', type=int, default=10, help='Number of questions (0=all)')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--model', default='gpt-4o', help='Model to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--resume', type=str, help='Resume from existing results directory')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Number of concurrent questions (default: 5)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Max iterations for compilation error correction (default: 3)')

    # Model config
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, help='Max tokens in response')

    args = parser.parse_args()

    # Build model config
    model_config = {}
    if args.temperature is not None:
        model_config['temperature'] = args.temperature
    if args.max_tokens is not None:
        model_config['max_tokens'] = args.max_tokens

    # Load prompts (shared bidirectional prompts)
    print("Loading prompts...")
    true_system = load_prompt('prompts/bidirectional/bidirectional_true_system.txt')
    false_system = load_prompt('prompts/bidirectional/bidirectional_false_system.txt')
    user_template = load_prompt('prompts/bidirectional/bidirectional_user.txt')
    print("✓ Loaded bidirectional prompts")

    # Initialize Lean server (synchronous)
    print("\nInitializing Lean server...")
    lean_server = create_lean_server(verbose=args.verbose)
    print("✓ Lean server initialized")

    # Load FOLIO data
    all_questions = load_folio(args.folio_file)

    if args.num_questions > 0:
        questions_to_test = all_questions[:args.num_questions]
    else:
        questions_to_test = all_questions

    # Handle resume
    processed_questions = set()
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            with open(all_results_file, 'r') as f:
                previous_results = json.load(f)
            processed_questions = {(r['story_id'], r['example_id']) for r in previous_results
                                   if 'story_id' in r and 'example_id' in r}
            print(f"Found {len(processed_questions)} already processed questions")

    remaining_questions = [q for q in questions_to_test
                          if (q.get('story_id'), q.get('example_id')) not in processed_questions]

    print(f"\nTesting {len(remaining_questions)} questions with bidirectional verification")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency} questions in parallel")
    print(f"Max iterations per proof: {args.max_iterations} (for compilation errors only)")
    print(f"(Each question runs TRUE + FALSE proofs in parallel)")

    # Initialize saver
    saver = BidirectionalSaver(args.output_dir, args.prompt_name,
                                resume_dir=args.resume)

    # Initialize async client
    client = AsyncOpenAI(api_key=args.api_key)

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    # Statistics tracking
    stats = {
        'patterns': {},
        'formalization_errors': 0,
        'fallbacks_used': 0,
        'accuracy_by_pattern': {},
        'total_true_iterations': 0,
        'total_false_iterations': 0,
        'compilation_retries': 0  # Questions that needed >1 iteration
    }

    total_questions = 0
    total_correct = 0

    # Create progress bar
    pbar = tqdm(total=len(remaining_questions), desc="Bidirectional Verification",
                unit="q", ncols=100)

    try:
        # Process in batches
        batch_size = args.concurrency
        for batch_start in range(0, len(remaining_questions), batch_size):
            batch = remaining_questions[batch_start:batch_start + batch_size]

            results = await process_batch_async(
                batch, client, lean_server,
                true_system, false_system, user_template,
                args.model, model_config, semaphore,
                args.max_iterations, args.verbose
            )

            # Save results
            for i, result in enumerate(results):
                question_index = len(processed_questions) + batch_start + i
                total_count = len(questions_to_test)

                if isinstance(result, Exception):
                    result = {
                        'example_id': batch[i].get('example_id'),
                        'story_id': batch[i].get('story_id'),
                        'error': str(result)
                    }

                async with saver._save_lock:
                    saver.save_result(result, question_index, total_count)

                if 'error' in result:
                    pbar.set_postfix_str(f"ERROR: {result['error'][:30]}")
                    pbar.update(1)
                    continue

                # Update stats
                pattern = result.get('agreement_pattern', 'Unknown')
                stats['patterns'][pattern] = stats['patterns'].get(pattern, 0) + 1

                if result.get('formalization_error'):
                    stats['formalization_errors'] += 1
                if result.get('used_fallback'):
                    stats['fallbacks_used'] += 1

                # Track iteration stats
                true_iters = result.get('true_num_iterations', 1)
                false_iters = result.get('false_num_iterations', 1)
                stats['total_true_iterations'] += true_iters
                stats['total_false_iterations'] += false_iters
                if true_iters > 1 or false_iters > 1:
                    stats['compilation_retries'] += 1

                if pattern not in stats['accuracy_by_pattern']:
                    stats['accuracy_by_pattern'][pattern] = {'correct': 0, 'total': 0}
                stats['accuracy_by_pattern'][pattern]['total'] += 1
                if result['correct']:
                    stats['accuracy_by_pattern'][pattern]['correct'] += 1

                total_questions += 1
                if result['correct']:
                    total_correct += 1

                # Update progress bar
                acc = total_correct / total_questions if total_questions > 0 else 0
                status = '✓' if result['correct'] else '✗'
                pbar.set_postfix_str(f"{pattern} {status} | Acc: {acc:.1%}")
                pbar.update(1)

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n\nInterrupted by user. Saving results...")
    except Exception as e:
        pbar.close()
        print(f"\n\nError: {e}. Saving results...")
        import traceback
        traceback.print_exc()

    saver.finalize(total_questions, total_correct, stats)

    if total_questions > 0:
        accuracy = total_correct / total_questions
        print(f"\nOverall Accuracy: {total_correct}/{total_questions} ({accuracy:.2%})")

        print(f"\nAgreement Pattern Distribution:")
        for pattern, count in sorted(stats['patterns'].items()):
            print(f"  {pattern}: {count}")

        print(f"\nFormalization Errors Detected: {stats['formalization_errors']}")
        print(f"Fallbacks Used: {stats['fallbacks_used']}")

        # Iteration stats
        if total_questions > 0:
            avg_true_iters = stats['total_true_iterations'] / total_questions
            avg_false_iters = stats['total_false_iterations'] / total_questions
            print(f"\nIteration Statistics:")
            print(f"  Avg TRUE iterations: {avg_true_iters:.2f}")
            print(f"  Avg FALSE iterations: {avg_false_iters:.2f}")
            print(f"  Questions needing retries: {stats['compilation_retries']}")


if __name__ == "__main__":
    asyncio.run(main_async())
