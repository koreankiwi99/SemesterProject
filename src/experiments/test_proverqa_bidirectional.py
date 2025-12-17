#!/usr/bin/env python3
"""
Bidirectional Verification for ProverQA dataset (Async version).

Adapted from test_folio_bidirectional.py for ProverQA format.
"""

import argparse
import json
import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

from utils.prompts import load_prompt
from utils.answer_parsing import parse_folio_answer, normalize_answer
from utils.lean_utils import extract_lean_code, verify_with_lean_async, create_lean_server
from utils.savers import BidirectionalSaver


def load_proverqa(filepath):
    """Load ProverQA dataset and extract premises/conclusion from context/question."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    questions = []
    for item in data:
        # Extract conclusion from question
        question_text = item['question']
        prefix = "Based on the above information, is the following statement true, false, or uncertain? "
        if question_text.startswith(prefix):
            conclusion = question_text[len(prefix):]
        else:
            conclusion = question_text

        questions.append({
            'example_id': item['id'],
            'premises': item['context'],
            'conclusion': conclusion,
            'label': item['answer'],
            'original_question': question_text
        })

    return questions


async def get_proof_attempt_async(example, client, system_prompt, user_template,
                                   model, model_config, direction="true",
                                   conversation_history=None):
    """Get a single proof attempt from the LLM (async)."""
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

        proof_failed_indicator = "PROOF FAILED" in llm_response.upper() or \
                                 "PROOF STATUS: FAILED" in llm_response.upper()

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


async def verify_lean_code_async(lean_code, lean_server, verbose=False, timeout=60.0):
    """Verify Lean code asynchronously."""
    if not lean_code:
        return False, []

    verification = await verify_with_lean_async(lean_code, lean_server, verbose, timeout)
    return verification['success'], verification.get('errors', [])


async def get_proof_with_iteration_async(example, client, lean_server, system_prompt,
                                          user_template, model, model_config,
                                          direction="true", max_iterations=3, verbose=False):
    """Get proof attempt with iteration on compilation errors (async)."""
    iterations = []
    conversation_history = None

    for iteration in range(max_iterations):
        result = await get_proof_attempt_async(
            example, client, system_prompt, user_template,
            model, model_config, direction, conversation_history
        )

        if result.get('error'):
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

        if result.get('proof_failed_indicator'):
            iteration_data['verification_success'] = False
            iteration_data['errors'] = []
            iterations.append(iteration_data)
            break

        success, errors = await verify_lean_code_async(lean_code, lean_server, verbose)
        iteration_data['verification_success'] = success
        iteration_data['errors'] = errors
        iterations.append(iteration_data)

        if success:
            if verbose:
                print(f"    {direction.upper()} proof succeeded on iteration {iteration + 1}")
            break

        if errors and iteration < max_iterations - 1:
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
            if verbose and not errors:
                print(f"    {direction.upper()} proof incomplete (sorry) on iteration {iteration + 1}")
            break

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
    system_prompt = """You are a logical reasoning expert. Analyze the premises and determine if the conclusion is True, False, or Uncertain.

Think step by step, then provide your final answer.

ANSWER FORMAT:
ANSWER: True/False/Uncertain"""

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
        return 'Uncertain', str(e)


def apply_agreement_logic(true_success, false_success):
    """Apply bidirectional agreement logic."""
    if true_success and not false_success:
        return 'True', 'TRUE_ONLY', False
    elif false_success and not true_success:
        return 'False', 'FALSE_ONLY', False
    elif true_success and false_success:
        return None, 'BOTH_SUCCESS', True
    else:
        return 'Uncertain', 'NEITHER_SUCCESS', False


async def test_question_bidirectional_async(example, client, lean_server,
                                            true_system, false_system, user_template,
                                            model="gpt-4", model_config=None,
                                            max_iterations=3, verbose=False):
    """Test a single question with bidirectional verification (async)."""
    result = {
        'example_id': example.get('example_id'),
        'premises': example['premises'],
        'conclusion': example['conclusion'],
        'ground_truth': example['label'],
    }

    # Run both proof attempts in parallel
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
    parser = argparse.ArgumentParser(description='ProverQA Bidirectional Verification')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_file', required=True, help='ProverQA JSON file')
    parser.add_argument('--prompt_name', default='proverqa_bidirectional', help='Name for output')
    parser.add_argument('--num_questions', type=int, default=10, help='Number of questions (0=all)')
    parser.add_argument('--output_dir', default='results/proverqa', help='Output directory')
    parser.add_argument('--model', default='gpt-5-2025-08-07', help='Model to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--resume', type=str, help='Resume from existing results directory')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent questions')
    parser.add_argument('--max_iterations', type=int, default=3, help='Max iterations per proof')
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, help='Max tokens in response')
    parser.add_argument('--restart_lean_every', type=int, default=5, help='Restart Lean server every N questions to avoid memory issues')

    args = parser.parse_args()

    model_config = {}
    if args.temperature is not None:
        model_config['temperature'] = args.temperature
    if args.max_tokens is not None:
        model_config['max_tokens'] = args.max_tokens

    print("Loading prompts...")
    true_system = load_prompt('prompts/proverqa/bidirectional_true_system.txt')
    false_system = load_prompt('prompts/proverqa/bidirectional_false_system.txt')
    user_template = load_prompt('prompts/proverqa/bidirectional_user.txt')
    print("✓ Loaded bidirectional prompts")

    print("\nInitializing Lean server...")
    lean_server = create_lean_server(verbose=args.verbose)
    print("✓ Lean server initialized")

    all_questions = load_proverqa(args.data_file)

    if args.num_questions > 0:
        questions_to_test = all_questions[:args.num_questions]
    else:
        questions_to_test = all_questions

    processed_questions = set()
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            with open(all_results_file, 'r') as f:
                previous_results = json.load(f)
            processed_questions = {r['example_id'] for r in previous_results if 'example_id' in r}
            print(f"Found {len(processed_questions)} already processed questions")

    remaining_questions = [q for q in questions_to_test
                          if q.get('example_id') not in processed_questions]

    print(f"\nTesting {len(remaining_questions)} questions with bidirectional verification")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")

    saver = BidirectionalSaver(args.output_dir, args.prompt_name, resume_dir=args.resume, dataset_name="proverqa")
    client = AsyncOpenAI(api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    stats = {
        'patterns': {},
        'formalization_errors': 0,
        'fallbacks_used': 0,
        'accuracy_by_pattern': {},
        'total_true_iterations': 0,
        'total_false_iterations': 0,
        'compilation_retries': 0
    }

    total_questions = 0
    total_correct = 0

    pbar = tqdm(total=len(remaining_questions), desc="Bidirectional", unit="q", ncols=100)

    questions_since_restart = 0

    try:
        batch_size = args.concurrency
        for batch_start in range(0, len(remaining_questions), batch_size):
            batch = remaining_questions[batch_start:batch_start + batch_size]

            # Restart Lean server periodically to avoid memory issues
            if questions_since_restart >= args.restart_lean_every:
                if args.verbose:
                    print(f"\nRestarting Lean server after {questions_since_restart} questions...")
                lean_server = create_lean_server(verbose=args.verbose)
                questions_since_restart = 0

            results = await process_batch_async(
                batch, client, lean_server,
                true_system, false_system, user_template,
                args.model, model_config, semaphore,
                args.max_iterations, args.verbose
            )

            questions_since_restart += len(batch)

            for i, result in enumerate(results):
                question_index = len(processed_questions) + batch_start + i
                total_count = len(questions_to_test)

                if isinstance(result, Exception):
                    result = {'example_id': batch[i].get('example_id'), 'error': str(result)}

                async with saver._save_lock:
                    saver.save_result(result, question_index, total_count)

                if 'error' in result:
                    pbar.set_postfix_str(f"ERROR: {result['error'][:30]}")
                    pbar.update(1)
                    continue

                pattern = result.get('agreement_pattern', 'Unknown')
                stats['patterns'][pattern] = stats['patterns'].get(pattern, 0) + 1

                if result.get('formalization_error'):
                    stats['formalization_errors'] += 1
                if result.get('used_fallback'):
                    stats['fallbacks_used'] += 1

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

                acc = total_correct / total_questions if total_questions > 0 else 0
                status = '✓' if result['correct'] else '✗'
                pbar.set_postfix_str(f"{pattern} {status} | Acc: {acc:.1%}")
                pbar.update(1)

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n\nInterrupted. Saving results...")
    except Exception as e:
        pbar.close()
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    saver.finalize(total_questions, total_correct, stats)

    if total_questions > 0:
        print(f"\nOverall Accuracy: {total_correct}/{total_questions} ({total_correct/total_questions:.2%})")
        print(f"\nAgreement Patterns: {stats['patterns']}")
        print(f"Formalization Errors: {stats['formalization_errors']}")
        print(f"Fallbacks Used: {stats['fallbacks_used']}")


if __name__ == "__main__":
    asyncio.run(main_async())
