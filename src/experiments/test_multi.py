#!/usr/bin/env python3
"""
Test models on Multi-LogiEval dataset using the paper's evaluation approach
Modified to sample individual questions rather than files
"""

import json
import argparse
import re
from collections import defaultdict
import os
from datetime import datetime
from pathlib import Path
import random

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
    
    # Add metadata to each sample, including the depth directory name
    for sample in samples:
        sample['logic_type'] = logic_type
        sample['rule'] = rule
        sample['depth'] = depth
        sample['depth_dir'] = depth_dir  # Add the directory name (e.g., 'd1_Data')
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
    
    # Print distribution
    distribution = defaultdict(lambda: defaultdict(int))
    for sample in sampled_questions:
        distribution[sample['logic_type']][sample['depth']] += 1
    
    print("\nDistribution of sampled questions:")
    print(f"{'='*70}")
    for logic_type in sorted(distribution.keys()):
        print(f"\n{logic_type}:")
        for depth in sorted(distribution[logic_type].keys()):
            print(f"  {depth}: {distribution[logic_type][depth]} questions")
    
    return sampled_questions

def build_prompt_single_sample(sample, system_prompt_template, user_prompt_template):
    """Build prompt for a single Multi-LogiEval sample using templates"""
    
    system_msg = system_prompt_template
    
    # Replace placeholders in user prompt
    user_prompt = user_prompt_template.replace("{premises}", sample['context'])
    user_prompt = user_prompt.replace("{questions}", sample['question'])
    
    return system_msg, user_prompt

def parse_multilogieval_answer(response):
    """Extract Yes/No answer from model response following paper's approach"""
    # Look for explicit Answer: pattern
    answer_match = re.search(r'Answer:\s*(Yes|No)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).capitalize()
    
    # Fallback: Look for standalone Yes/No
    yes_no_match = re.findall(r'\b(Yes|No)\b', response, re.IGNORECASE)
    if yes_no_match:
        # Return the last occurrence
        return yes_no_match[-1].capitalize()
    
    return 'Unknown'

def normalize_answer(answer):
    """Normalize answer format to Yes/No"""
    if not answer:
        return 'Unknown'
    
    low = answer.lower().strip()
    
    # Multi-LogiEval uses Yes/No format
    if low in ['yes', 'y', 'true', 't', '1']:
        return 'Yes'
    elif low in ['no', 'n', 'false', 'f', '0']:
        return 'No'
    
    return 'Unknown'

def test_model_on_samples(samples, api_key, system_prompt_template, user_prompt_template, model="gpt-4"):
    """Test model on sampled questions"""
    import openai
    openai.api_key = api_key
    
    results = []
    results_by_combination = defaultdict(list)
    
    print(f"\nTesting {len(samples)} questions with {model}...")
    print("=" * 70)
    
    for i, sample in enumerate(samples, 1):
        # Use depth_dir for the combination key to match the sampling
        combination_key = (sample['logic_type'], sample['depth_dir'])
        print(f"\n[{i}/{len(samples)}] {sample['logic_type']}/{sample['depth_dir']} - {sample['rule']}")
        
        system_msg, user_prompt = build_prompt_single_sample(sample, system_prompt_template, user_prompt_template)
        
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            # Parse the answer
            prediction = parse_multilogieval_answer(gpt_response)
            ground_truth = normalize_answer(sample['answer'])
            
            correct = prediction == ground_truth
            
            result = {
                'question_num': i,
                'logic_type': sample['logic_type'],
                'depth': sample['depth'],
                'depth_dir': sample['depth_dir'],
                'rule': sample['rule'],
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': correct,
                'full_response': gpt_response,
                'source_file': sample['source_file']
            }
            
            results.append(result)
            results_by_combination[combination_key].append(result)
            
            print(f"  Prediction: {prediction} | Ground Truth: {ground_truth} | {'✓' if correct else '✗'}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            result = {
                'question_num': i,
                'logic_type': sample['logic_type'],
                'depth': sample['depth'],
                'depth_dir': sample['depth_dir'],
                'rule': sample['rule'],
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': normalize_answer(sample['answer']),
                'prediction': 'Error',
                'correct': False,
                'error': str(e),
                'source_file': sample['source_file']
            }
            results.append(result)
            results_by_combination[combination_key].append(result)
    
    return results, results_by_combination

class ResultsSaver:
    """Handles saving of results in organized structure"""
    
    def __init__(self, output_dir="results", prompt_name="test"):
        self.output_dir = output_dir
        self.prompt_name = prompt_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output directories
        self.base_dir = f"{output_dir}/multilogieval_{prompt_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.responses_file = f"{self.base_dir}/all_responses.json"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.detailed_file = f"{self.base_dir}/detailed_results.txt"
        self.accuracy_table_file = f"{self.base_dir}/accuracy_table.txt"
    
    def save_results(self, results, results_by_combination):
        """Save all results in various formats"""
        # Save JSON results
        with open(self.responses_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed results with reasoning chains
        self._save_detailed_results(results)
        
        # Save summary
        self._save_summary(results, results_by_combination)
        
        # Save accuracy table like paper's Table 6
        self._save_accuracy_table(results_by_combination)
        
        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All responses (JSON): {self.responses_file}")
        print(f"Detailed results:     {self.detailed_file}")
        print(f"Summary:              {self.summary_file}")
        print(f"Accuracy table:       {self.accuracy_table_file}")
        print(f"{'='*70}")
    
    def _save_detailed_results(self, results):
        """Save detailed results with full reasoning chains"""
        with open(self.detailed_file, 'w') as f:
            f.write("Multi-LogiEval Detailed Results\n")
            f.write("=" * 100 + "\n\n")
            
            for result in results:
                f.write(f"Question {result['question_num']}/{len(results)}\n")
                f.write("-" * 100 + "\n")
                f.write(f"Logic Type: {result['logic_type']}\n")
                f.write(f"Depth: {result['depth']}\n")
                f.write(f"Rule: {result['rule']}\n")
                f.write(f"Source: {result['source_file']}\n\n")
                
                f.write("Context:\n")
                f.write(result['context'] + "\n\n")
                
                f.write("Question:\n")
                f.write(result['question'] + "\n\n")
                
                if 'full_response' in result:
                    f.write("Model Response:\n")
                    f.write("-" * 50 + "\n")
                    f.write(result['full_response'])
                    f.write("\n" + "-" * 50 + "\n\n")
                
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction:   {result['prediction']}\n")
                f.write(f"Correct:      {'✓ Yes' if result['correct'] else '✗ No'}\n")
                
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                
                f.write("\n" + "=" * 100 + "\n\n")
    
    def _save_summary(self, results, results_by_combination):
        """Save overall summary"""
        with open(self.summary_file, 'w') as f:
            f.write("Multi-LogiEval Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_correct = sum(r['correct'] for r in results)
            total_questions = len(results)
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            
            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Total Correct: {total_correct}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
            
            # Accuracy by logic type
            f.write("Accuracy by Logic Type:\n")
            f.write("-" * 70 + "\n")
            logic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            for r in results:
                logic_stats[r['logic_type']]['total'] += 1
                if r['correct']:
                    logic_stats[r['logic_type']]['correct'] += 1
            
            for logic_type in sorted(logic_stats.keys()):
                stats = logic_stats[logic_type]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"{logic_type.upper():<10} {stats['correct']:>3}/{stats['total']:<3} ({acc:.2%})\n")
            
            # Accuracy by depth
            f.write("\nAccuracy by Depth:\n")
            f.write("-" * 70 + "\n")
            depth_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            for r in results:
                depth_stats[r['depth']]['total'] += 1
                if r['correct']:
                    depth_stats[r['depth']]['correct'] += 1
            
            for depth in sorted(depth_stats.keys()):
                stats = depth_stats[depth]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                depth_label = depth.replace('_Data', '')
                f.write(f"{depth_label:<10} {stats['correct']:>3}/{stats['total']:<3} ({acc:.2%})\n")
    
    def _save_accuracy_table(self, results_by_combination):
        """Save accuracy table in paper's Table 6 format"""
        with open(self.accuracy_table_file, 'w') as f:
            f.write("Accuracy Table (Format similar to paper's Table 6)\n")
            f.write("=" * 70 + "\n\n")
            
            # Calculate accuracies
            accuracies = {}
            logic_types = ['pl', 'fol', 'nm']
            depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']
            
            for logic_type in logic_types:
                accuracies[logic_type] = {}
                for depth in depths:
                    key = (logic_type, depth)
                    if key in results_by_combination:
                        results = results_by_combination[key]
                        correct = sum(r['correct'] for r in results)
                        total = len(results)
                        if total > 0:
                            accuracies[logic_type][depth] = correct / total * 100
                        else:
                            accuracies[logic_type][depth] = None
                    else:
                        accuracies[logic_type][depth] = None
            
            # Print table header
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
                
                # Calculate average
                if valid_accs:
                    avg = sum(valid_accs) / len(valid_accs)
                    f.write(f"{avg:>11.2f}%")
                else:
                    f.write(f"{'N/A':>12}")
                f.write("\n")
            
            # Overall average row
            f.write("-" * 70 + "\n")
            f.write(f"{'Average':<15}")
            for depth in depths:
                depth_accs = []
                for logic_type in logic_types:
                    if accuracies[logic_type][depth] is not None:
                        depth_accs.append(accuracies[logic_type][depth])
                if depth_accs:
                    avg = sum(depth_accs) / len(depth_accs)
                    f.write(f"{avg:>9.2f}%")
                else:
                    f.write(f"{'N/A':>10}")
            
            # Overall average
            all_accs = []
            for logic_type in logic_types:
                for depth in depths:
                    if accuracies[logic_type][depth] is not None:
                        all_accs.append(accuracies[logic_type][depth])
            if all_accs:
                overall_avg = sum(all_accs) / len(all_accs)
                f.write(f"{overall_avg:>11.2f}%")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description='Test models on Multi-LogiEval dataset (sampling individual questions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Zero-shot CoT (paper's main approach) - 10 questions per combination
  python test_multilogieval_sampling.py \\
      --api_key YOUR_KEY \\
      --data_dir data/MultiLogicEval \\
      --system_prompt prompts/multilogieval/zero_shot_cot_system.txt \\
      --user_prompt prompts/multilogieval/zero_shot_cot_user.txt \\
      --prompt_name zero_shot_cot \\
      --samples_per_combination 10 \\
      --model gpt-4
      
  # Test with 50 questions per combination
  python test_multilogieval_sampling.py \\
      --api_key YOUR_KEY \\
      --data_dir data/MultiLogicEval \\
      --system_prompt prompts/multilogieval/zero_shot_cot_system.txt \\
      --user_prompt prompts/multilogieval/zero_shot_cot_user.txt \\
      --prompt_name zero_shot_cot_large \\
      --samples_per_combination 50 \\
      --model gpt-4
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_dir', required=True, help='Path to Multi-LogiEval data directory')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files')
    parser.add_argument('--logic_types', nargs='+', 
                        default=['fol', 'nm', 'pl'],
                        help='Logic types to test (default: all)')
    parser.add_argument('--depths', nargs='+',
                        default=['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data'],
                        help='Depths to test (default: all)')
    parser.add_argument('--samples_per_combination', type=int, default=10,
                        help='Number of questions to sample per logic type × depth combination (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    args = parser.parse_args()
    
    # Load prompts
    print("Loading prompts...")
    system_prompt_template = load_prompt(args.system_prompt)
    user_prompt_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")
    
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
    
    # Test model on sampled questions
    try:
        results, results_by_combination = test_model_on_samples(
            sampled_questions,
            args.api_key,
            system_prompt_template,
            user_prompt_template,
            args.model
        )
        
        # Save results
        saver = ResultsSaver(args.output_dir, args.prompt_name)
        saver.save_results(results, results_by_combination)
        
        # Print final summary
        total_correct = sum(r['correct'] for r in results)
        total_questions = len(results)
        if total_questions > 0:
            overall_accuracy = total_correct / total_questions
            print(f"\n{'='*70}")
            print(f"OVERALL ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
            print(f"{'='*70}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()