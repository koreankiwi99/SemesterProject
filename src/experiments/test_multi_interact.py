#!/usr/bin/env python3
"""
Test Multi-LogiEval dataset with interactive Lean verification
Each question is processed individually with iterative Lean refinement
"""

import json
import argparse
import re
from collections import defaultdict
import os
from datetime import datetime
from pathlib import Path
import random
from lean_interact import LeanREPLConfig, LeanServer, Command

def load_prompt(prompt_file):
    """Load prompt from a text file"""
    try:
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

def load_multilogieval_file(file_path, depth_dir):
    """Load a single Multi-LogiEval JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
    
    logic_type = data.get('logic', 'unknown')
    rule = data.get('rule', 'unknown')
    depth = data.get('depth', 'unknown')
    samples = data.get('samples', [])
    
    # Add metadata to each sample
    for sample in samples:
        sample['logic_type'] = logic_type
        sample['rule'] = rule
        sample['depth'] = depth
        sample['depth_dir'] = depth_dir
        sample['source_file'] = str(file_path)
    
    return samples

def load_and_sample_multilogieval(data_dir, logic_types=None, depths=None, samples_per_combination=10, seed=42):
    """Load Multi-LogiEval dataset and sample individual questions"""
    data_path = Path(data_dir)
    
    if logic_types is None:
        logic_types = ['fol', 'nm', 'pl']
    
    if depths is None:
        depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']
    
    print(f"Loading Multi-LogiEval from: {data_dir}")
    print(f"Logic types: {logic_types}")
    print(f"Depths: {depths}")
    print(f"Sampling {samples_per_combination} questions per logic type × depth combination")
    print(f"Random seed: {seed}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Collect all samples organized by combination
    samples_by_combination = defaultdict(list)
    
    for depth_dir in depths:
        depth_path = data_path / depth_dir
        if not depth_path.exists():
            print(f"Warning: {depth_path} does not exist, skipping...")
            continue
        
        for logic_type in logic_types:
            logic_path = depth_path / logic_type
            if not logic_path.exists():
                print(f"Warning: {logic_path} does not exist, skipping...")
                continue
            
            json_files = list(logic_path.glob('*.json'))
            print(f"Found {len(json_files)} files in {depth_dir}/{logic_type}")
            
            # Load all samples from all files for this combination
            combination_key = (logic_type, depth_dir)
            for json_file in json_files:
                samples = load_multilogieval_file(json_file, depth_dir)
                samples_by_combination[combination_key].extend(samples)
    
    # Sample from each combination
    sampled_questions = []
    print(f"\n{'='*70}")
    print("Sampling questions from each combination:")
    print(f"{'='*70}")
    
    for (logic_type, depth_dir), samples in sorted(samples_by_combination.items()):
        num_available = len(samples)
        num_to_sample = min(samples_per_combination, num_available)
        
        sampled = random.sample(samples, num_to_sample)
        sampled_questions.extend(sampled)
        
        print(f"{logic_type}/{depth_dir}: sampled {num_to_sample} from {num_available} available questions")
    
    print(f"\n{'='*70}")
    print(f"Total questions sampled: {len(sampled_questions)}")
    print(f"{'='*70}")
    
    return sampled_questions

def build_prompt(sample, system_template, user_template):
    """Build prompt for a single Multi-LogiEval sample using templates"""
    system_msg = system_template
    
    # Replace placeholders in user prompt
    user_prompt = user_template.replace("{premises}", sample['context'])
    user_prompt = user_prompt.replace("{questions}", sample['question'])
    
    return system_msg, user_prompt

def extract_lean_code(llm_response):
    """Extract Lean code from XML-style tags"""
    match = re.search(r'<lean>(.*?)</lean>', llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: try markdown code blocks
    code_blocks = re.findall(r'```lean\s*(.*?)```', llm_response, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return '\n\n'.join(block.strip() for block in code_blocks)
    
    return None

def parse_multilogieval_answer(response):
    """Extract Yes/No/Unknown answer from model response"""
    # Look for explicit Answer: pattern
    answer_match = re.search(r'ANSWER:\s*(Yes|No|Unknown)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).capitalize()
    
    # Fallback: Look for standalone Yes/No/Unknown
    yes_no_match = re.findall(r'\b(Yes|No|Unknown)\b', response, re.IGNORECASE)
    if yes_no_match:
        return yes_no_match[-1].capitalize()
    
    return 'Unknown'

def normalize_answer(answer):
    """Normalize answer format to Yes/No/Unknown"""
    if not answer:
        return 'Unknown'
    
    low = answer.lower().strip()
    
    if low in ['yes', 'y', 'true', 't', '1']:
        return 'Yes'
    elif low in ['no', 'n', 'false', 'f', '0']:
        return 'No'
    
    return 'Unknown'

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

def test_question_with_lean(sample, api_key, lean_server, system_template, user_template,
                            model="gpt-4", max_iterations=3, verbose=False):
    """Test a single question with interactive Lean verification"""
    import openai
    openai.api_key = api_key

    system_msg, user_prompt = build_prompt(sample, system_template, user_template)
    
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
                            f"ANSWER: Yes/No/Unknown"
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
                        f"ANSWER: Yes/No/Unknown"
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

        ground_truth = normalize_answer(sample['answer'])
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

class IncrementalSaver:
    """Handles incremental saving of results"""
    
    def __init__(self, output_dir="results", prompt_name="lean_test"):
        self.output_dir = output_dir
        self.prompt_name = prompt_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output directories
        self.base_dir = f"{output_dir}/multilogieval_lean_{prompt_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.responses_dir = f"{self.base_dir}/responses"
        os.makedirs(self.responses_dir, exist_ok=True)
        
        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.accuracy_table_file = f"{self.base_dir}/accuracy_table.txt"
        self.lean_stats_file = f"{self.base_dir}/lean_verification_stats.txt"
        self.progress_file = f"{self.base_dir}/progress.txt"
        
        self._init_files()
    
    def _init_files(self):
        """Initialize output files"""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)
        
        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval + Lean Interactive Testing - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")
    
    def save_result(self, result, question_index, total_questions):
        """Save a single result incrementally"""
        self._append_to_json(result)
        
        if 'error' not in result:
            self._save_individual_response(result, question_index)
        
        self._update_progress(result, question_index, total_questions)
    
    def _append_to_json(self, result):
        """Append result to JSON file"""
        try:
            with open(self.all_results_file, 'r') as f:
                data = json.load(f)
            data.append(result)
            with open(self.all_results_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")
    
    def _save_individual_response(self, result, question_index):
        """Save individual response file"""
        filename = f"q{question_index + 1:03d}_{result['logic_type']}_{result['depth_dir']}_{result['rule']}.txt"
        response_file = f"{self.responses_dir}/{filename}"
        
        try:
            with open(response_file, 'w') as f:
                f.write("Multi-LogiEval Question with Lean Verification\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Logic Type: {result['logic_type']}\n")
                f.write(f"Depth: {result['depth']} ({result['depth_dir']})\n")
                f.write(f"Rule: {result['rule']}\n")
                f.write(f"Source: {result['source_file']}\n\n")
                
                f.write("Context:\n")
                f.write(result['context'] + "\n\n")
                
                f.write("Question:\n")
                f.write(result['question'] + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("Iterations:\n")
                f.write("=" * 80 + "\n\n")
                
                for iter_data in result.get('iterations', []):
                    f.write(f"--- Iteration {iter_data['iteration']} ---\n\n")
                    f.write("LLM Response:\n")
                    f.write(iter_data['llm_response'] + "\n\n")
                    
                    if iter_data.get('lean_code'):
                        f.write("Extracted Lean Code:\n")
                        f.write("-" * 40 + "\n")
                        f.write(iter_data['lean_code'] + "\n")
                        f.write("-" * 40 + "\n\n")
                        
                        if iter_data.get('lean_verification'):
                            verification = iter_data['lean_verification']
                            f.write("Lean Verification:\n")
                            f.write(f"  Success: {verification['success']}\n")
                            if verification['errors']:
                                f.write("  Errors:\n")
                                for err in verification['errors']:
                                    f.write(f"    - {err}\n")
                            if verification['warnings']:
                                f.write("  Warnings:\n")
                                for warn in verification['warnings']:
                                    f.write(f"    - {warn}\n")
                    else:
                        f.write("No Lean code found in this iteration.\n")
                    
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("Final Result:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'✓ Yes' if result['correct'] else '✗ No'}\n")
                f.write(f"Total Iterations: {result['num_iterations']}\n")
                
                if result.get('lean_verification'):
                    f.write(f"Final Lean Verification: {'✓ Success' if result['lean_verification']['success'] else '✗ Failed'}\n")
                elif result.get('lean_code') is None:
                    f.write("Final Lean Verification: No Lean code generated\n")
                
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")
    
    def _update_progress(self, result, question_index, total_questions):
        """Update progress file"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Logic: {result.get('logic_type')}, Depth: {result.get('depth_dir')}, Rule: {result.get('rule')}\n")
                
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")
                    f.write(f"  Iterations: {result['num_iterations']}\n")
                    
                    if result.get('lean_verification'):
                        lean_success = result['lean_verification']['success']
                        f.write(f"  Lean Verification: {'Success' if lean_success else 'Failed'}\n")
                    elif result.get('lean_code') is None:
                        f.write("  Lean Code: Not found in response\n")
                
                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")
    
    def finalize(self, results, results_by_combination):
        """Generate final summaries and statistics"""
        self._save_summary(results)
        self._save_accuracy_table(results_by_combination)
        self._save_lean_stats(results)
        
        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results (JSON): {self.all_results_file}")
        print(f"Summary:            {self.summary_file}")
        print(f"Accuracy table:     {self.accuracy_table_file}")
        print(f"Lean stats:         {self.lean_stats_file}")
        print(f"Progress log:       {self.progress_file}")
        print(f"Individual responses: {self.responses_dir}/")
        print(f"{'='*70}")
    
    def _save_summary(self, results):
        """Save overall summary"""
        with open(self.summary_file, 'w') as f:
            f.write("Multi-LogiEval with Lean Verification - Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_correct = sum(r['correct'] for r in results if 'error' not in r)
            total_questions = len([r for r in results if 'error' not in r])
            total_errors = len([r for r in results if 'error' in r])
            
            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
            else:
                f.write("No questions completed successfully.\n")
            
            if total_errors > 0:
                f.write(f"Errors: {total_errors}\n")
            
            # Logic type breakdown
            f.write("\nAccuracy by Logic Type:\n")
            f.write("-" * 70 + "\n")
            logic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            for r in results:
                if 'error' not in r:
                    logic_stats[r['logic_type']]['total'] += 1
                    if r['correct']:
                        logic_stats[r['logic_type']]['correct'] += 1
            
            for logic_type in sorted(logic_stats.keys()):
                stats = logic_stats[logic_type]
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    f.write(f"{logic_type.upper():<10} {stats['correct']:>3}/{stats['total']:<3} ({acc:.2%})\n")
    
    def _save_accuracy_table(self, results_by_combination):
        """Save accuracy table in paper format"""
        with open(self.accuracy_table_file, 'w') as f:
            f.write("Accuracy Table (Paper's Table 6 Format)\n")
            f.write("=" * 70 + "\n\n")
            
            logic_types = ['pl', 'fol', 'nm']
            depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']
            
            # Calculate accuracies
            accuracies = {}
            for logic_type in logic_types:
                accuracies[logic_type] = {}
                for depth in depths:
                    key = (logic_type, depth)
                    if key in results_by_combination:
                        results = results_by_combination[key]
                        results = [r for r in results if 'error' not in r]
                        if results:
                            correct = sum(r['correct'] for r in results)
                            total = len(results)
                            accuracies[logic_type][depth] = correct / total * 100
                        else:
                            accuracies[logic_type][depth] = None
                    else:
                        accuracies[logic_type][depth] = None
            
            # Print table
            f.write(f"{'Logic':<15}")
            for depth in depths:
                f.write(f"{depth.replace('_Data', ''):>10}")
            f.write(f"{'Average':>12}\n")
            f.write("-" * 70 + "\n")
            
            # Print rows
            for logic_type in logic_types:
                f.write(f"{logic_type.upper():<15}")
                valid_accs = []
                for depth in depths:
                    acc = accuracies[logic_type][depth]
                    if acc is not None:
                        f.write(f"{acc:>9.2f}%")
                        valid_accs.append(acc)
                    else:
                        f.write(f"{'N/A':>10}")
                
                if valid_accs:
                    avg = sum(valid_accs) / len(valid_accs)
                    f.write(f"{avg:>11.2f}%")
                else:
                    f.write(f"{'N/A':>12}")
                f.write("\n")
    
    def _save_lean_stats(self, results):
        """Save Lean verification statistics"""
        with open(self.lean_stats_file, 'w') as f:
            f.write("Lean Verification Statistics\n")
            f.write("=" * 70 + "\n\n")
            
            # Calculate stats
            total_questions = 0
            questions_with_code = 0
            successful_verifications = 0
            failed_verifications = 0
            total_iterations = 0
            
            iteration_counts = defaultdict(int)
            
            for r in results:
                if 'error' not in r:
                    total_questions += 1
                    total_iterations += r['num_iterations']
                    iteration_counts[r['num_iterations']] += 1
                    
                    if r.get('lean_code'):
                        questions_with_code += 1
                        if r.get('lean_verification'):
                            if r['lean_verification']['success']:
                                successful_verifications += 1
                            else:
                                failed_verifications += 1
            
            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Questions with Lean code: {questions_with_code}\n")
            f.write(f"Successful verifications: {successful_verifications}\n")
            f.write(f"Failed verifications: {failed_verifications}\n")
            
            if total_questions > 0:
                avg_iterations = total_iterations / total_questions
                f.write(f"\nAverage iterations: {avg_iterations:.2f}\n")
            
            if questions_with_code > 0:
                verification_rate = successful_verifications / questions_with_code
                f.write(f"Verification success rate: {verification_rate:.2%}\n")
            
            f.write("\nIteration Distribution:\n")
            for num_iter in sorted(iteration_counts.keys()):
                count = iteration_counts[num_iter]
                f.write(f"  {num_iter} iteration(s): {count} questions\n")

def main():
    parser = argparse.ArgumentParser(
        description='Test Multi-LogiEval with interactive Lean verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python test_multilogieval_lean.py \\
      --api_key YOUR_KEY \\
      --data_dir data/MultiLogicEval \\
      --system_prompt prompts/multilogieval/lean_system.txt \\
      --user_prompt prompts/multilogieval/lean_user.txt \\
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
    
    args = parser.parse_args()
    
    # Load prompts
    print("Loading prompts...")
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
    
    print(f"\nTesting {len(sampled_questions)} questions with {args.model}")
    print(f"Max iterations per question: {args.max_iterations}")
    print(f"Interactive Lean verification: ENABLED\n")
    
    saver = IncrementalSaver(args.output_dir, args.prompt_name)
    
    results = []
    results_by_combination = defaultdict(list)
    
    try:
        for i, sample in enumerate(sampled_questions):
            combination_key = (sample['logic_type'], sample['depth_dir'])
            
            print(f"\n[{i+1}/{len(sampled_questions)}] {sample['logic_type']}/{sample['depth_dir']} - {sample['rule']}")
            
            result = test_question_with_lean(sample, args.api_key, lean_server,
                                           system_template, user_template,
                                           args.model, args.max_iterations, args.verbose)
            
            results.append(result)
            results_by_combination[combination_key].append(result)
            
            saver.save_result(result, i, len(sampled_questions))
            
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
    saver.finalize(results, results_by_combination)
    
    # Print final summary
    total_questions = len([r for r in results if 'error' not in r])
    if total_questions > 0:
        total_correct = sum(r['correct'] for r in results if 'error' not in r)
        overall_accuracy = total_correct / total_questions
        
        # Lean stats
        questions_with_code = len([r for r in results if 'error' not in r and r.get('lean_code')])
        successful_verifications = len([r for r in results if 'error' not in r and r.get('lean_verification', {}).get('success', False)])
        total_iterations = sum(r['num_iterations'] for r in results if 'error' not in r)
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