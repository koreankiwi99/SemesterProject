#!/usr/bin/env python3
"""
Test FOLIO dataset with two-stage multi-turn Lean verification (Option B).

Option B: Bidirectional iteration
- Stage 1: Natural language reasoning
- Stage 2: Lean verification with iterative refinement
- If Stage 2 fails after max iterations, go back to Stage 1 and reconsider
- Repeat the whole cycle up to max_stage_cycles times
"""

import argparse
import json
import os
import re

from utils.prompts import load_prompt
from utils.answer_parsing import normalize_answer
from utils.lean_utils import extract_lean_code, verify_with_lean, create_lean_server
from utils.savers import FOLIOLeanSaver
from datasets.folio import load_folio, build_folio_prompt_single


def extract_stage_answer(response, stage=1):
    """Extract stage-specific answer from response.

    Args:
        response: The model response text
        stage: Stage number (1 or 2)

    Returns:
        str: Extracted answer (True/False/Unknown)
    """
    pattern = rf'STAGE\s*{stage}\s*ANSWER:\s*(True|False|Unknown)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1), answer_format="true_false")

    # Fallback: look for ANSWER: without stage number
    if stage == 2:
        pattern = r'ANSWER:\s*(True|False|Unknown)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return normalize_answer(match.group(1), answer_format="true_false")

    return 'Unknown'


def run_stage1(api_key, example, stage1_system, stage1_user, model, model_config,
               revision_feedback=None, verbose=False):
    """Run Stage 1: Natural language reasoning.

    Args:
        api_key: OpenAI API key
        example: FOLIO example
        stage1_system: Stage 1 system prompt
        stage1_user: Stage 1 user prompt template
        model: Model name
        model_config: Model configuration
        revision_feedback: Optional feedback for revision (from Stage 2 failures)
        verbose: Verbose output

    Returns:
        tuple: (stage1_response, stage1_answer)
    """
    import openai
    openai.api_key = api_key

    # Build prompt
    stage1_sys, stage1_usr = build_folio_prompt_single(example, stage1_system, stage1_user)

    messages = [
        {"role": "system", "content": stage1_sys},
        {"role": "user", "content": stage1_usr}
    ]

    # Add revision feedback if this is a revision
    if revision_feedback:
        messages.append({"role": "user", "content": revision_feedback})

    api_params = {"model": model, "messages": messages}
    if model_config:
        api_params.update(model_config)

    response = openai.chat.completions.create(**api_params)
    stage1_response = response.choices[0].message.content.strip()
    stage1_answer = extract_stage_answer(stage1_response, stage=1)

    if verbose:
        if revision_feedback:
            print(f"Stage 1 (REVISED) Answer: {stage1_answer}")
        else:
            print(f"Stage 1 Answer: {stage1_answer}")

    return stage1_response, stage1_answer


def run_stage2(api_key, example, stage1_response, stage2_system, stage2_user,
               model, model_config, lean_server, max_iterations=3, verbose=False):
    """Run Stage 2: Lean verification with iterative refinement.

    Args:
        api_key: OpenAI API key
        example: FOLIO example
        stage1_response: Stage 1 response text
        stage2_system: Stage 2 system prompt
        stage2_user: Stage 2 user prompt template
        model: Model name
        model_config: Model configuration
        lean_server: LeanServer instance
        max_iterations: Max Lean refinement iterations
        verbose: Verbose output

    Returns:
        dict: {
            'success': bool (Lean verified successfully),
            'stage2_answer': str,
            'lean_code': str,
            'verification': dict,
            'iterations': list,
            'conversation_history': list
        }
    """
    import openai
    openai.api_key = api_key

    # Build Stage 2 prompt with Stage 1 context
    stage2_usr = stage2_user.format(
        stage1_response=stage1_response,
        premises=example['premises'],
        conclusion=example['conclusion']
    )

    conversation_history = [
        {"role": "system", "content": stage2_system},
        {"role": "user", "content": stage2_usr}
    ]

    stage2_iterations = []
    final_stage2_answer = 'Unknown'
    final_lean_code = None
    final_verification = None
    lean_success = False

    api_params_base = {"model": model}
    if model_config:
        api_params_base.update(model_config)

    for iteration in range(max_iterations):
        if verbose:
            print(f"  Stage 2 Iteration {iteration + 1}/{max_iterations}")

        api_params = {**api_params_base, "messages": conversation_history}
        response = openai.chat.completions.create(**api_params)

        llm_response = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": llm_response})

        # Extract answer and Lean code
        stage2_answer = extract_stage_answer(llm_response, stage=2)
        lean_code = extract_lean_code(llm_response)

        iteration_data = {
            'iteration': iteration + 1,
            'llm_response': llm_response,
            'stage2_answer': stage2_answer,
            'lean_code': lean_code,
            'lean_verification': None
        }

        # Verify Lean code if present
        if lean_code:
            lean_verification = verify_with_lean(lean_code, lean_server, verbose)
            iteration_data['lean_verification'] = lean_verification

            if lean_verification['success']:
                if verbose:
                    print(f"  ✓ Lean verification successful")
                final_stage2_answer = stage2_answer
                final_lean_code = lean_code
                final_verification = lean_verification
                lean_success = True
                stage2_iterations.append(iteration_data)
                break
            else:
                if verbose:
                    print(f"  ✗ Lean verification failed")

                if iteration < max_iterations - 1:
                    error_messages = '\n'.join(lean_verification['errors'])
                    feedback = (
                        f"The Lean code has compilation errors:\n\n"
                        f"{error_messages}\n\n"
                        f"Please fix the errors and provide corrected Lean code in <lean></lean> tags.\n"
                        f"Then provide your answer:\n"
                        f"STAGE 2 ANSWER: True/False/Unknown"
                    )
                    conversation_history.append({"role": "user", "content": feedback})
        else:
            if verbose:
                print(f"  ✗ No Lean code found")

            if iteration < max_iterations - 1:
                feedback = (
                    f"I didn't find any Lean code in your response. "
                    f"Please provide Lean code wrapped in <lean></lean> tags.\n"
                    f"Then provide your answer:\n"
                    f"STAGE 2 ANSWER: True/False/Unknown"
                )
                conversation_history.append({"role": "user", "content": feedback})

        stage2_iterations.append(iteration_data)

        # Use current answer if last iteration
        if iteration == max_iterations - 1:
            final_stage2_answer = stage2_answer
            final_lean_code = lean_code
            final_verification = lean_verification if lean_code else None

    return {
        'success': lean_success,
        'stage2_answer': final_stage2_answer,
        'lean_code': final_lean_code,
        'verification': final_verification,
        'iterations': stage2_iterations,
        'conversation_history': conversation_history
    }


def test_question_two_stage(example, api_key, lean_server,
                            stage1_system, stage1_user,
                            stage2_system, stage2_user,
                            model="gpt-5", model_config=None,
                            max_lean_iterations=3, max_stage_cycles=2,
                            verbose=False):
    """Test a single question with two-stage bidirectional iteration.

    Args:
        example: FOLIO example dictionary
        api_key: OpenAI API key
        lean_server: LeanServer instance
        stage1_system: Stage 1 system prompt template
        stage1_user: Stage 1 user prompt template
        stage2_system: Stage 2 system prompt template
        stage2_user: Stage 2 user prompt template
        model: Model name to use
        model_config: Optional model configuration dict
        max_lean_iterations: Maximum Lean refinement iterations in Stage 2
        max_stage_cycles: Maximum times to cycle between Stage 1 and Stage 2
        verbose: Whether to print verbose output

    Returns:
        dict: Result dictionary with both stages and all cycles
    """
    all_cycles = []
    final_stage1_answer = 'Unknown'
    final_stage2_answer = 'Unknown'
    final_lean_code = None
    final_verification = None

    try:
        for cycle in range(max_stage_cycles):
            if verbose:
                print(f"\n{'='*60}")
                print(f"CYCLE {cycle + 1}/{max_stage_cycles}")
                print(f"{'='*60}")

            # Prepare revision feedback for Stage 1 (if not first cycle)
            revision_feedback = None
            if cycle > 0:
                revision_feedback = (
                    f"Your previous reasoning led to Lean code that could not be verified "
                    f"after {max_lean_iterations} attempts. Please reconsider your logical reasoning "
                    f"and provide a revised analysis.\n\n"
                    f"Provide your revised reasoning and answer in the format:\n"
                    f"STAGE 1 ANSWER: True/False/Unknown"
                )

            # ===== STAGE 1: Natural Language Reasoning =====
            if verbose:
                print("\n=== STAGE 1: Natural Language Reasoning ===")

            stage1_response, stage1_answer = run_stage1(
                api_key, example, stage1_system, stage1_user,
                model, model_config, revision_feedback, verbose
            )

            # ===== STAGE 2: Lean Verification =====
            if verbose:
                print("\n=== STAGE 2: Lean Verification ===")

            stage2_result = run_stage2(
                api_key, example, stage1_response, stage2_system, stage2_user,
                model, model_config, lean_server, max_lean_iterations, verbose
            )

            # Store this cycle's results
            cycle_data = {
                'cycle': cycle + 1,
                'stage1_response': stage1_response,
                'stage1_answer': stage1_answer,
                'stage2_answer': stage2_result['stage2_answer'],
                'stage2_iterations': stage2_result['iterations'],
                'lean_code': stage2_result['lean_code'],
                'lean_verification': stage2_result['verification'],
                'lean_success': stage2_result['success']
            }
            all_cycles.append(cycle_data)

            # Update finals
            final_stage1_answer = stage1_answer
            final_stage2_answer = stage2_result['stage2_answer']
            final_lean_code = stage2_result['lean_code']
            final_verification = stage2_result['verification']

            # If Lean verification succeeded, stop cycling
            if stage2_result['success']:
                if verbose:
                    print(f"\n✓ Lean verification succeeded in cycle {cycle + 1}. Stopping.")
                break
            else:
                if verbose:
                    print(f"\n✗ Lean verification failed in cycle {cycle + 1}.")
                if cycle < max_stage_cycles - 1:
                    if verbose:
                        print(f"   → Going back to Stage 1 for revision...")
                else:
                    if verbose:
                        print(f"   → Max cycles reached. Stopping.")

        # Determine correctness
        ground_truth = normalize_answer(example['label'], answer_format="true_false")
        stage1_correct = final_stage1_answer == ground_truth
        stage2_correct = final_stage2_answer == ground_truth

        # Calculate total iterations across all cycles
        total_iterations = sum(len(cycle['stage2_iterations']) for cycle in all_cycles)

        return {
            'example_id': example.get('example_id'),
            'story_id': example.get('story_id'),
            'premises': example['premises'],
            'conclusion': example['conclusion'],
            'ground_truth': ground_truth,
            'stage1_answer': final_stage1_answer,
            'stage1_correct': stage1_correct,
            'stage2_answer': final_stage2_answer,
            'stage2_correct': stage2_correct,
            'prediction': final_stage2_answer,  # Use Stage 2 as final
            'correct': stage2_correct,  # Use Stage 2 for overall correctness
            'model': model,
            'model_config': model_config or {},
            'num_cycles': len(all_cycles),
            'all_cycles': all_cycles,
            'total_iterations': total_iterations,
            'lean_code': final_lean_code,
            'lean_verification': final_verification
        }

    except Exception as e:
        import traceback
        return {
            'example_id': example.get('example_id'),
            'story_id': example.get('story_id'),
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test FOLIO with two-stage bidirectional iteration (Option B)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_folio_two_stage.py \\
      --api_key YOUR_KEY \\
      --folio_file data/folio_original/folio-validation.json \\
      --stage1_system prompts/folio/two_stage_lean_system_turn1.txt \\
      --stage1_user prompts/folio/two_stage_lean_user.txt \\
      --stage2_system prompts/folio/two_stage_lean_system_turn2.txt \\
      --stage2_user prompts/folio/two_stage_lean_user_turn2.txt \\
      --prompt_name two_stage_lean \\
      --num_questions 10 \\
      --model gpt-5
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--stage1_system', required=True, help='Stage 1 system prompt')
    parser.add_argument('--stage1_user', required=True, help='Stage 1 user prompt')
    parser.add_argument('--stage2_system', required=True, help='Stage 2 system prompt')
    parser.add_argument('--stage2_user', required=True, help='Stage 2 user prompt')
    parser.add_argument('--prompt_name', default='two_stage_lean', help='Name for output files')
    parser.add_argument('--num_questions', type=int, default=10,
                        help='Number of questions to test (0 = all)')
    parser.add_argument('--max_lean_iterations', type=int, default=3,
                        help='Maximum Lean refinement iterations in Stage 2 (default: 3)')
    parser.add_argument('--max_stage_cycles', type=int, default=2,
                        help='Maximum cycles between Stage 1 and Stage 2 (default: 2)')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    parser.add_argument('--lean_version', default=None, help='Lean version (default: latest)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--resume', type=str, help='Path to existing results directory to resume from')

    # Model configuration options
    parser.add_argument('--temperature', type=float, help='Sampling temperature (0.0-2.0)')
    parser.add_argument('--reasoning_effort', type=str, choices=['low', 'medium', 'high'],
                        help='Reasoning effort (for models like o1/o3)')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens in response')
    parser.add_argument('--top_p', type=float, help='Nucleus sampling parameter')

    args = parser.parse_args()

    # Build model config
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
    print("Loading prompts...")
    stage1_system = load_prompt(args.stage1_system)
    stage1_user = load_prompt(args.stage1_user)
    stage2_system = load_prompt(args.stage2_system)
    stage2_user = load_prompt(args.stage2_user)
    print(f"✓ Loaded Stage 1 prompts")
    print(f"✓ Loaded Stage 2 prompts")

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
            processed_questions = {(r['story_id'], r['example_id']) for r in previous_results
                                  if 'story_id' in r and 'example_id' in r}
            print(f"Found {len(processed_questions)} already processed questions")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found")

    # Filter out already processed questions
    remaining_questions = [q for q in questions_to_test
                          if (q.get('story_id'), q.get('example_id')) not in processed_questions]

    print(f"\nTesting {len(remaining_questions)} questions with {args.model}")
    if processed_questions:
        print(f"(Skipping {len(processed_questions)} already completed questions)")
    print(f"Max Lean iterations per Stage 2: {args.max_lean_iterations}")
    print(f"Max cycles between stages: {args.max_stage_cycles}")
    print(f"Two-stage bidirectional iteration (Option B): ENABLED\n")

    # Initialize saver
    if args.resume:
        saver = FOLIOLeanSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = FOLIOLeanSaver(args.output_dir, args.prompt_name)

    total_questions = 0
    total_correct_stage1 = 0
    total_correct_stage2 = 0
    total_cycles = 0
    total_iterations = 0
    lean_stats = {'with_code': 0, 'successful': 0, 'failed': 0, 'avg_iterations': 0, 'avg_cycles': 0}

    try:
        for i, example in enumerate(remaining_questions):
            total_index = len(processed_questions) + i + 1
            total_count = len(questions_to_test)
            print(f"\n{'='*70}")
            print(f"Question {total_index}/{total_count} "
                  f"(Story: {example.get('story_id')}, ID: {example.get('example_id')})")
            print(f"{'='*70}")

            result = test_question_two_stage(
                example, args.api_key, lean_server,
                stage1_system, stage1_user,
                stage2_system, stage2_user,
                args.model, model_config,
                args.max_lean_iterations, args.max_stage_cycles,
                args.verbose
            )

            saver.save_result(result, total_index - 1, total_count)

            if 'error' in result:
                print(f"\n✗ Error: {result['error']}")
                if 'traceback' in result:
                    print(result['traceback'])
                continue

            print(f"\nGround Truth: {result['ground_truth']}")
            print(f"Stage 1:      {result['stage1_answer']} {'✓' if result['stage1_correct'] else '✗'}")
            print(f"Stage 2:      {result['stage2_answer']} {'✓' if result['stage2_correct'] else '✗'}")
            print(f"Cycles:       {result['num_cycles']}")
            print(f"Iterations:   {result['total_iterations']}")

            if result.get('lean_verification'):
                lean_status = '✓ Success' if result['lean_verification']['success'] else '✗ Failed'
                print(f"Lean:         {lean_status}")
            elif result.get('lean_code') is None:
                print(f"Lean:         No code generated")

            total_questions += 1
            if result['stage1_correct']:
                total_correct_stage1 += 1
            if result['stage2_correct']:
                total_correct_stage2 += 1
            total_cycles += result['num_cycles']
            total_iterations += result['total_iterations']

            # Track Lean stats
            if result.get('lean_code'):
                lean_stats['with_code'] += 1
                if result.get('lean_verification'):
                    if result['lean_verification']['success']:
                        lean_stats['successful'] += 1
                    else:
                        lean_stats['failed'] += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}. Saving results so far...")
        import traceback
        traceback.print_exc()

    if total_questions > 0:
        lean_stats['avg_iterations'] = total_iterations / total_questions
        lean_stats['avg_cycles'] = total_cycles / total_questions

    saver.finalize(total_questions, total_correct_stage2, lean_stats)

    if total_questions > 0:
        stage1_accuracy = total_correct_stage1 / total_questions
        stage2_accuracy = total_correct_stage2 / total_questions

        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"STAGE 1 ACCURACY: {total_correct_stage1}/{total_questions} ({stage1_accuracy:.2%})")
        print(f"STAGE 2 ACCURACY: {total_correct_stage2}/{total_questions} ({stage2_accuracy:.2%})")
        print(f"Average cycles per question: {lean_stats['avg_cycles']:.2f}")
        print(f"Average iterations per question: {lean_stats['avg_iterations']:.2f}")

        if lean_stats['with_code'] > 0:
            print(f"\nLean Verification Summary:")
            print(f"  Questions with Lean code: {lean_stats['with_code']}")
            print(f"  Successful verifications: {lean_stats['successful']}")
            print(f"  Failed verifications: {lean_stats['failed']}")
            verification_rate = lean_stats['successful'] / lean_stats['with_code']
            print(f"  Verification success rate: {verification_rate:.2%}")
        print(f"{'='*70}")
    else:
        print("\nNo questions were completed successfully.")


if __name__ == "__main__":
    main()
