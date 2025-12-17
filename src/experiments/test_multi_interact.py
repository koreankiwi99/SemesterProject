#!/usr/bin/env python3
"""
Test Multi-LogiEval dataset with interactive Lean verification.
Each question is processed individually with iterative Lean refinement.
"""

import argparse
import json
import os
from collections import defaultdict

from utils.prompts import load_prompt
from utils.answer_parsing import parse_multilogieval_answer, parse_two_stage_answers, normalize_answer
from utils.lean_utils import extract_lean_code, verify_with_lean, create_lean_server
from utils.savers import MultiLogiEvalLeanSaver
from datasets.multilogieval import load_and_sample_multilogieval, build_multilogieval_prompt


def test_question_with_lean(sample, api_key, lean_server, system_template, user_template,
                            model="gpt-5", model_config=None, max_iterations=3, verbose=False):
    """Test a single question with interactive Lean verification.

    Args:
        sample: Question sample dictionary
        api_key: OpenAI API key
        lean_server: LeanServer instance
        system_template: System prompt template
        user_template: User prompt template
        model: Model name to use
        model_config: Optional model configuration dict (temperature, reasoning_effort, etc.)
        max_iterations: Maximum number of refinement iterations
        verbose: Whether to print verbose output

    Returns:
        dict: Result dictionary with iterations and verification info
    """
    import openai
    openai.api_key = api_key

    system_msg, user_prompt = build_multilogieval_prompt(sample, system_template, user_template)

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
            response = openai.chat.completions.create(**api_params)

            llm_response = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": llm_response})

            # Extract answer and Lean code
            # Try two-stage parsing first (for two_stage_lean prompt)
            two_stage_answers = parse_two_stage_answers(llm_response)
            if two_stage_answers['stage1_answer'] != 'Unknown' or two_stage_answers['stage2_answer'] != 'Unknown':
                # Two-stage format detected
                prediction = two_stage_answers['stage2_answer']  # Use stage 2 as final
                stage1_answer = two_stage_answers['stage1_answer']
                stage2_answer = two_stage_answers['stage2_answer']
            else:
                # Fallback to single answer parsing
                prediction = parse_multilogieval_answer(llm_response)
                stage1_answer = None
                stage2_answer = None

            lean_code = extract_lean_code(llm_response)

            iteration_data = {
                'iteration': iteration + 1,
                'llm_response': llm_response,
                'prediction': prediction,
                'stage1_answer': stage1_answer,
                'stage2_answer': stage2_answer,
                'lean_code': lean_code,
                'lean_verification': None
            }

            # Verify Lean code if present
            if lean_code:
                lean_verification = verify_with_lean(lean_code, lean_server, verbose)
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
                            f"Please reconsider your reasoning from STAGE 1 in light of these errors. "
                            f"Then provide both stages again:\n\n"
                            f"STAGE 1: [natural language reasoning]\n"
                            f"STAGE 1 ANSWER: Yes/No\n\n"
                            f"STAGE 2: [Lean code in <lean></lean> tags]\n"
                            f"STAGE 2 ANSWER: Yes/No"
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
                        f"Please provide both stages:\n\n"
                        f"STAGE 1: [natural language reasoning]\n"
                        f"STAGE 1 ANSWER: Yes/No\n\n"
                        f"STAGE 2: [Lean code in <lean></lean> tags]\n"
                        f"STAGE 2 ANSWER: Yes/No"
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

        ground_truth = normalize_answer(sample['answer'], answer_format="yes_no")
        correct = final_prediction == ground_truth

        return {
            'question_id': sample.get('id'),
            'logic_type': sample['logic_type'],
            'depth': sample['depth'],
            'depth_dir': sample['depth_dir'],
            'rule': sample['rule'],
            'context': sample['context'],
            'question': sample['question'],
            'ground_truth': ground_truth,
            'prediction': final_prediction,
            'correct': correct,
            'model': model,
            'model_config': model_config or {},
            'iterations': iterations,
            'num_iterations': len(iterations),
            'lean_code': final_lean_code,
            'lean_verification': final_verification,
            'conversation_history': conversation_history,
            'source_file': sample['source_file']
        }

    except Exception as e:
        return {
            'question_id': sample.get('id'),
            'logic_type': sample['logic_type'],
            'depth': sample['depth'],
            'depth_dir': sample['depth_dir'],
            'rule': sample['rule'],
            'error': str(e),
            'iterations': iterations
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test Multi-LogiEval with interactive Lean verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python test_multi_interact.py \\
      --api_key YOUR_KEY \\
      --data_dir data/multi_logi_original/data \\
      --system_prompt prompts/multilogi/lean_system.txt \\
      --user_prompt prompts/multilogi/lean_user.txt \\
      --prompt_name lean_test \\
      --samples_per_combination 5 \\
      --max_iterations 3 \\
      --model gpt-4
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_dir', required=True, help='Path to Multi-LogiEval data directory')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='lean_test', help='Name for output files')
    parser.add_argument('--logic_types', nargs='+',
                        default=['fol', 'nm', 'pl'],
                        help='Logic types to test (default: all)')
    parser.add_argument('--depths', nargs='+',
                        default=['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data'],
                        help='Depths to test (default: all)')
    parser.add_argument('--samples_per_combination', type=int, default=5,
                        help='Number of questions to sample per logic type × depth combination (default: 5)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum Lean revision iterations per question (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-5', help='Model to use (default: gpt-5)')
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
    print("Loading prompts...")
    system_template = load_prompt(args.system_prompt)
    user_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    # Initialize Lean server
    print(f"\nInitializing Lean server...")
    lean_server = create_lean_server(args.lean_version, args.verbose)
    print(f"✓ Lean server initialized")

    # Load and sample dataset
    sampled_questions = load_and_sample_multilogieval(
        args.data_dir,
        args.logic_types,
        args.depths,
        args.samples_per_combination,
        args.seed
    )

    if not sampled_questions:
        print("No questions found! Check your data directory and filters.")
        return

    # Handle resume functionality
    processed_keys = set()
    if args.resume:
        all_results_file = os.path.join(args.resume, 'all_results.json')
        if os.path.exists(all_results_file):
            print(f"\nResuming from: {args.resume}")
            with open(all_results_file, 'r') as f:
                previous_results = json.load(f)
            # Track by (logic_type, depth_dir, rule, context, question) to identify unique questions
            processed_keys = {
                (r['logic_type'], r['depth_dir'], r['rule'], r['context'], r['question'])
                for r in previous_results if 'logic_type' in r
            }
            print(f"Found {len(processed_keys)} already processed questions")
        else:
            print(f"Warning: Resume directory exists but no all_results.json found: {all_results_file}")

    # Filter out already processed questions
    remaining_questions = [
        q for q in sampled_questions
        if (q['logic_type'], q['depth_dir'], q['rule'], q['context'], q['question']) not in processed_keys
    ]

    print(f"\nTotal questions to test: {len(sampled_questions)}")
    if processed_keys:
        print(f"Already completed: {len(processed_keys)}")
        print(f"Remaining: {len(remaining_questions)}")
    print(f"Max iterations per question: {args.max_iterations}")
    print(f"Interactive Lean verification: ENABLED\n")

    if not remaining_questions:
        print("\nAll questions already completed!")
        return

    # Initialize saver (will resume if --resume provided)
    if args.resume:
        saver = MultiLogiEvalLeanSaver(args.output_dir, args.prompt_name, resume_dir=args.resume)
    else:
        saver = MultiLogiEvalLeanSaver(args.output_dir, args.prompt_name)

    results = []
    results_by_combination = defaultdict(list)

    try:
        for i, sample in enumerate(remaining_questions):
            combination_key = (sample['logic_type'], sample['depth_dir'])
            total_index = len(processed_keys) + i + 1
            total_count = len(sampled_questions)

            print(f"\n[{total_index}/{total_count}] {sample['logic_type']}/{sample['depth_dir']} - {sample['rule']}")

            result = test_question_with_lean(sample, args.api_key, lean_server,
                                           system_template, user_template,
                                           args.model, model_config, args.max_iterations, args.verbose)

            results.append(result)
            results_by_combination[combination_key].append(result)

            saver.save_result(result, total_index - 1, total_count)

            if 'error' in result:
                print(f"  ✗ Error: {result['error']}")
            else:
                print(f"  Ground Truth: {result['ground_truth']}")
                print(f"  Prediction:   {result['prediction']}")
                print(f"  Correct:      {'✓' if result['correct'] else '✗'}")
                print(f"  Iterations:   {result['num_iterations']}")

                if result.get('lean_verification'):
                    lean_status = '✓ Success' if result['lean_verification']['success'] else '✗ Failed'
                    print(f"  Lean:         {lean_status}")
                elif result.get('lean_code') is None:
                    print(f"  Lean:         No code generated")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving results...")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    # Finalize and save all results
    # Load all results (including previous ones if resuming)
    with open(saver.all_results_file, 'r') as f:
        all_results = json.load(f)

    # Rebuild results_by_combination from all results
    all_results_by_combination = defaultdict(list)
    for r in all_results:
        key = (r['logic_type'], r['depth_dir'])
        all_results_by_combination[key].append(r)

    saver.finalize(all_results, all_results_by_combination)

    # Print final summary
    total_questions = len([r for r in all_results if 'error' not in r])
    if total_questions > 0:
        total_correct = sum(r['correct'] for r in all_results if 'error' not in r)
        overall_accuracy = total_correct / total_questions

        # Lean stats
        questions_with_code = len([r for r in all_results if 'error' not in r and r.get('lean_code')])
        successful_verifications = len([r for r in all_results if 'error' not in r and r.get('lean_verification', {}).get('success', False)])
        total_iterations = sum(r['num_iterations'] for r in all_results if 'error' not in r)
        avg_iterations = total_iterations / total_questions

        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
        print(f"Average iterations per question: {avg_iterations:.2f}")

        if questions_with_code > 0:
            verification_rate = successful_verifications / questions_with_code
            print(f"Lean verification success rate: {successful_verifications}/{questions_with_code} ({verification_rate:.2%})")

        print(f"{'='*70}")


if __name__ == "__main__":
    main()
