#!/usr/bin/env python3
"""
Test FOLIO dataset with LeanInteract integration
Each question is processed individually with interactive Lean verification
"""

import json
import argparse
import re
import os
from datetime import datetime
from lean_interact import LeanREPLConfig, LeanServer, Command

def load_prompt(prompt_file):
    """Load prompt from a text file"""
    try:
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

def load_folio(file_path):
    """Load FOLIO dataset"""
    print(f"Loading FOLIO from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    
    # Count unique stories for info
    unique_stories = len(set(ex.get('story_id') for ex in data if ex.get('story_id')))
    print(f"Found {unique_stories} unique stories")
    
    return data

def build_prompt(example, system_template, user_template):
    """Build prompt for a single question using templates"""
    system_prompt = system_template
    
    user_prompt = user_template.replace("{premises}", example['premises'])
    user_prompt = user_prompt.replace("{questions}", example['conclusion'])
    
    return system_prompt, user_prompt

def extract_lean_code(llm_response):
    """Extract Lean code from XML-style tags"""
    match = re.search(r'<lean>(.*?)</lean>', llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: try markdown code blocks in case LLM forgets
    code_blocks = re.findall(r'```lean\s*(.*?)```', llm_response, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return '\n\n'.join(block.strip() for block in code_blocks)
    
    return None

def extract_answer(llm_response):
    """Extract the final answer (True/False/Unknown) from LLM response"""
    # Look for ANSWER: format first
    answer_match = re.search(r'ANSWER:\s*(True|False|Unknown)', llm_response, re.IGNORECASE)
    if answer_match:
        return normalize_answer(answer_match.group(1))
    
    # Fallback: look for last occurrence of True/False/Unknown
    all_answers = re.findall(r'\b(True|False|Unknown)\b', llm_response, re.IGNORECASE)
    if all_answers:
        return normalize_answer(all_answers[-1])
    
    return 'Unknown'

def normalize_answer(answer):
    """Normalize answer format"""
    if not answer:
        return 'Unknown'
    low = answer.lower().strip()
    if low in ['true', 't']:
        return 'True'
    elif low in ['false', 'f']:
        return 'False'
    elif low in ['unknown', 'uncertain', 'u']:
        return 'Unknown'
    return answer

def verify_with_lean(lean_code, lean_server, verbose=False):
    """Verify Lean code and return verification results"""
    try:
        if verbose:
            print(f"\nVerifying Lean code:\n{lean_code}\n")
        
        response = lean_server.run(Command(cmd=lean_code))
        
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

def test_question_with_lean(example, api_key, lean_server, system_template, user_template,
                            model="gpt-5", max_iterations=3, verbose=False):
    """Test a single question with interactive Lean verification"""
    import openai
    openai.api_key = api_key

    system_msg, user_prompt = build_prompt(example, system_template, user_template)
    
    conversation_history = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ]
    
    iterations = []
    final_prediction = 'Unknown'
    final_lean_code = None
    final_verification = None

    try:
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Get LLM response
            response = openai.chat.completions.create(
                model=model,
                messages=conversation_history,
            )

            llm_response = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": llm_response})
            
            # Extract answer and Lean code
            prediction = extract_answer(llm_response)
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
                    
                    if iteration < max_iterations - 1:  # Don't give feedback on last iteration
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

        correct = normalize_answer(final_prediction) == normalize_answer(example["label"])

        return {
            'example_id': example.get('example_id'),
            'story_id': example.get('story_id'),
            'premises': example['premises'],
            'conclusion': example['conclusion'],
            'ground_truth': example['label'],
            'prediction': final_prediction,
            'correct': correct,
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

class IncrementalSaver:
    """Handles incremental saving of results"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        self.detailed_file = f"{output_dir}/leaninteract_folio_results_{self.timestamp}.json"
        self.responses_dir = f"{output_dir}/leaninteract_responses_{self.timestamp}"
        self.progress_file = f"{output_dir}/leaninteract_progress_{self.timestamp}.txt"
        
        os.makedirs(self.responses_dir, exist_ok=True)
        self._init_files()
    
    def _init_files(self):
        """Initialize output files"""
        with open(self.detailed_file, 'w') as f:
            json.dump([], f)
        
        with open(self.progress_file, 'w') as f:
            f.write(f"FOLIO + Lean Interactive Testing - Started at {self.timestamp}\n")
            f.write("=" * 50 + "\n")
    
    def save_result(self, result, question_index, total_questions):
        """Save a single result incrementally"""
        self._append_to_json(result)
        
        if 'error' not in result:
            self._save_individual_response(result)
        
        self._update_progress(result, question_index, total_questions)
    
    def _append_to_json(self, result):
        """Append result to JSON file"""
        try:
            with open(self.detailed_file, 'r') as f:
                data = json.load(f)
            data.append(result)
            with open(self.detailed_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")
    
    def _save_individual_response(self, result):
        """Save individual response file"""
        story_id = result['story_id']
        example_id = result['example_id']
        response_file = f"{self.responses_dir}/story_{story_id}_q{example_id}.txt"
        
        try:
            with open(response_file, 'w') as f:
                f.write(f"Story ID: {story_id}\n")
                f.write(f"Example ID: {example_id}\n")
                f.write(f"Premises: {result['premises']}\n")
                f.write(f"Conclusion: {result['conclusion']}\n\n")
                
                # Write all iterations
                for iter_data in result.get('iterations', []):
                    f.write("=" * 50 + "\n")
                    f.write(f"Iteration {iter_data['iteration']}\n")
                    f.write("=" * 50 + "\n")
                    f.write(iter_data['llm_response'])
                    f.write("\n\n")
                    
                    if iter_data.get('lean_code'):
                        f.write("--- Lean Code ---\n")
                        f.write(iter_data['lean_code'])
                        f.write("\n\n")
                    
                    if iter_data.get('lean_verification'):
                        f.write("--- Lean Verification ---\n")
                        verification = iter_data['lean_verification']
                        f.write(f"Success: {verification['success']}\n")
                        if verification['errors']:
                            f.write(f"Errors:\n")
                            for err in verification['errors']:
                                f.write(f"  - {err}\n")
                        if verification['warnings']:
                            f.write(f"Warnings:\n")
                            for warn in verification['warnings']:
                                f.write(f"  - {warn}\n")
                        f.write("\n")
                
                f.write("=" * 50 + "\n")
                f.write("Final Result:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")
                f.write(f"Total Iterations: {result['num_iterations']}\n")
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")
    
    def _update_progress(self, result, question_index, total_questions):
        """Update progress file"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"\nQuestion {question_index+1}/{total_questions}\n")
                f.write(f"Story: {result.get('story_id')}, Example: {result.get('example_id')}\n")
                
                if 'error' in result:
                    f.write(f"ERROR: {result['error']}\n")
                else:
                    f.write(f"Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")
                    f.write(f"Iterations: {result['num_iterations']}\n")
                    
                    if result.get('lean_verification'):
                        lean_success = result['lean_verification']['success']
                        f.write(f"Lean Verification: {'Success' if lean_success else 'Failed'}\n")
                    elif result.get('lean_code') is None:
                        f.write("Lean Code: Not found in response\n")
                
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")
    
    def finalize(self, total_questions, total_correct, lean_stats):
        """Write final summary"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 50 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                
                f.write(f"\nLean Verification Stats:\n")
                f.write(f"  Questions with Lean code: {lean_stats['with_code']}\n")
                f.write(f"  Successful verifications: {lean_stats['successful']}\n")
                f.write(f"  Failed verifications: {lean_stats['failed']}\n")
                f.write(f"  Average iterations: {lean_stats['avg_iterations']:.2f}\n")
                if lean_stats['with_code'] > 0:
                    verification_rate = lean_stats['successful'] / lean_stats['with_code']
                    f.write(f"  Verification success rate: {verification_rate:.2%}\n")
                
                f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize progress: {e}")
        
        print(f"\nAll results saved to: {self.detailed_file}")
        print(f"Individual responses in: {self.responses_dir}/")
        print(f"Progress log: {self.progress_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Test FOLIO with interactive LeanInteract verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_folio_lean.py \\
      --api_key YOUR_KEY \\
      --folio_file data/folio.json \\
      --system_prompt prompts/lean_system.txt \\
      --user_prompt prompts/lean_user.txt \\
      --num_questions 50 \\
      --max_iterations 3 \\
      --verbose
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--folio_file', required=True, help='FOLIO JSON file')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--num_questions', type=int, default=10,
                        help='Number of questions to test (0 = all)')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum Lean revision iterations per question')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    parser.add_argument('--lean_version', default=None, help='Lean version (default: latest)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    # Load prompts
    print(f"Loading prompts...")
    system_template = load_prompt(args.system_prompt)
    user_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")

    # Initialize Lean server
    print(f"\nInitializing Lean server...")
    config_kwargs = {'verbose': args.verbose}
    if args.lean_version:
        config_kwargs['lean_version'] = args.lean_version
    
    config = LeanREPLConfig(**config_kwargs)
    lean_server = LeanServer(config)
    print(f"✓ Lean server initialized")

    # Load FOLIO data
    all_questions = load_folio(args.folio_file)
    
    # Select questions to test
    if args.num_questions > 0:
        questions_to_test = all_questions[:args.num_questions]
    else:
        questions_to_test = all_questions

    print(f"\nTesting {len(questions_to_test)} questions with {args.model}")
    print(f"Max iterations per question: {args.max_iterations}")
    print(f"Interactive Lean verification: ENABLED\n")
    
    saver = IncrementalSaver(args.output_dir)
    
    total_questions = 0
    total_correct = 0
    total_iterations = 0
    lean_stats = {'with_code': 0, 'successful': 0, 'failed': 0, 'avg_iterations': 0}

    try:
        for i, example in enumerate(questions_to_test):
            print(f"\nQuestion {i+1}/{len(questions_to_test)} "
                  f"(Story: {example.get('story_id')}, ID: {example.get('example_id')})")

            result = test_question_with_lean(example, args.api_key, lean_server,
                                            system_template, user_template,
                                            args.model, args.max_iterations, args.verbose)
            
            saver.save_result(result, i, len(questions_to_test))

            if 'error' in result:
                print(f"Error: {result['error']}")
                continue

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
    main()