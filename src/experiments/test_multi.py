#!/usr/bin/env python3
"""
Test GPT-5 on Multi-LogiEval dataset with customizable prompts and incremental saving
Samples 10 files from each logic type × depth combination
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

def load_multilogieval_file(file_path):
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
    
    converted_samples = []
    for sample in samples:
        converted_samples.append({
            'id': sample.get('id'),
            'context': sample.get('context', ''),
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'logic_type': logic_type,
            'rule': rule,
            'depth': depth,
            'file_path': str(file_path)
        })
    
    return {
        'logic_type': logic_type,
        'rule': rule,
        'depth': depth,
        'samples': converted_samples,
        'file_path': str(file_path)
    }

def load_multilogieval_dataset(data_dir, logic_types=None, depths=None, sample_per_combination=10, seed=42):
    """Load Multi-LogiEval dataset from directory structure with sampling"""
    data_path = Path(data_dir)
    
    if logic_types is None:
        logic_types = ['fol', 'nm', 'pl']
    
    if depths is None:
        depths = ['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data']
    
    print(f"Loading Multi-LogiEval from: {data_dir}")
    print(f"Logic types: {logic_types}")
    print(f"Depths: {depths}")
    print(f"Sampling {sample_per_combination} files per logic type × depth combination")
    print(f"Random seed: {seed}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Organize files by logic type and depth
    files_by_combination = defaultdict(list)
    
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
            
            # Store files by combination
            combination_key = (logic_type, depth_dir)
            files_by_combination[combination_key].extend(json_files)
    
    # Sample from each combination
    sampled_files = []
    print(f"\n{'='*70}")
    print("Sampling files from each combination:")
    print(f"{'='*70}")
    
    for (logic_type, depth_dir), files in sorted(files_by_combination.items()):
        num_available = len(files)
        num_to_sample = min(sample_per_combination, num_available)
        
        sampled = random.sample(files, num_to_sample)
        sampled_files.extend(sampled)
        
        print(f"{logic_type}/{depth_dir}: sampled {num_to_sample} from {num_available} available files")
    
    # Load the sampled files
    print(f"\n{'='*70}")
    print("Loading sampled files...")
    print(f"{'='*70}")
    
    all_files = []
    for json_file in sampled_files:
        file_data = load_multilogieval_file(json_file)
        all_files.append(file_data)
    
    print(f"\nTotal files loaded: {len(all_files)}")
    total_samples = sum(len(f['samples']) for f in all_files)
    print(f"Total samples: {total_samples}")
    
    # Print distribution
    print(f"\n{'='*70}")
    print("Distribution of sampled files:")
    print(f"{'='*70}")
    distribution = defaultdict(lambda: defaultdict(int))
    for file_data in all_files:
        distribution[file_data['logic_type']][file_data['depth']] += 1
    
    for logic_type in sorted(distribution.keys()):
        print(f"\n{logic_type}:")
        for depth in sorted(distribution[logic_type].keys()):
            print(f"  {depth}: {distribution[logic_type][depth]} files")
    
    return all_files

def build_prompt_single_sample(sample, system_prompt_template, user_prompt_template):
    """Build prompt for a single Multi-LogiEval sample using templates"""
    
    system_msg = system_prompt_template
    
    user_prompt = user_prompt_template.replace("{premises}", sample['context'])
    user_prompt = user_prompt.replace("{questions}", sample['question'])
    
    return system_msg, user_prompt

def test_gpt5_multilogieval(file_data, api_key, system_prompt_template, user_prompt_template, model="gpt-5"):
    """Test GPT-5 on Multi-LogiEval file - processing each sample individually"""
    import openai
    openai.api_key = api_key
    
    samples = file_data['samples']
    
    if not samples:
        return {'error': 'No samples in file'}
    
    results = []
    all_responses = []
    
    for i, sample in enumerate(samples, 1):
        print(f"  Processing sample {i}/{len(samples)}...", end=' ')
        
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
            all_responses.append({
                'sample_id': sample.get('id'),
                'response': gpt_response
            })
            
            prediction = parse_single_answer(gpt_response)
            ground_truth = normalize_answer(sample['answer'])
            prediction_normalized = normalize_answer(prediction)
            
            correct = prediction_normalized == ground_truth
            
            results.append({
                'question_num': i,
                'sample_id': sample.get('id'),
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': ground_truth,
                'prediction': prediction_normalized,
                'correct': correct
            })
            
            print(f"{'✓' if correct else '✗'}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                'question_num': i,
                'sample_id': sample.get('id'),
                'context': sample['context'],
                'question': sample['question'],
                'ground_truth': normalize_answer(sample['answer']),
                'prediction': 'Error',
                'correct': False,
                'error': str(e)
            })
    
    return {
        'logic_type': file_data['logic_type'],
        'rule': file_data['rule'],
        'depth': file_data['depth'],
        'file_path': file_data['file_path'],
        'all_responses': all_responses,
        'results': results,
        'file_accuracy': sum(r['correct'] for r in results) / len(results) if results else 0
    }

def parse_single_answer(response):
    """Extract single answer from GPT response"""
    answer_match = re.search(r'ANSWER:\s*(True|False|Unknown|Yes|No)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    
    final_answer = re.findall(r'\b(True|False|Unknown|Yes|No)\b', response, re.IGNORECASE)
    if final_answer:
        return final_answer[-1]
    
    return 'Unknown'

def normalize_answer(answer):
    """Normalize answer format"""
    if not answer:
        return 'Unknown'
    low = answer.lower().strip()
    if low in ['true', 't', 'yes', 'y']:
        return 'True'
    elif low in ['false', 'f', 'no', 'n']:
        return 'False'
    elif low in ['unknown', 'uncertain', 'u']:
        return 'Unknown'
    return answer

class IncrementalSaver:
    """Handles incremental saving of results with organized structure"""
    
    def __init__(self, output_dir="results", prompt_name="test"):
        self.output_dir = output_dir
        self.prompt_name = prompt_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        self.detailed_file = f"{output_dir}/multilogieval_{prompt_name}_responses_{self.timestamp}.json"
        self.progress_file = f"{output_dir}/multilogieval_{prompt_name}_progress_{self.timestamp}.txt"
        
        self.responses_dir = f"{output_dir}/multilogieval_{prompt_name}_responses_{self.timestamp}"
        self.by_logic_dir = f"{output_dir}/multilogieval_{prompt_name}_by_logic_{self.timestamp}"
        self.by_depth_dir = f"{output_dir}/multilogieval_{prompt_name}_by_depth_{self.timestamp}"
        self.summary_dir = f"{output_dir}/multilogieval_{prompt_name}_summaries_{self.timestamp}"
        
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.by_logic_dir, exist_ok=True)
        os.makedirs(self.by_depth_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        
        self.results_by_logic = defaultdict(list)
        self.results_by_depth = defaultdict(list)
        self.results_by_logic_depth = defaultdict(lambda: defaultdict(list))
        
        self._init_files()
    
    def _init_files(self):
        """Initialize output files"""
        with open(self.detailed_file, 'w') as f:
            json.dump([], f)
        
        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval Testing Progress - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")
    
    def save_result(self, result, file_index, total_files):
        """Save a single result incrementally"""
        self._append_to_json(result)
        
        if 'error' not in result:
            self._save_individual_response(result)
            self._track_for_aggregation(result)
        
        self._update_progress(result, file_index, total_files)
        
        print(f"✓ Saved result for {result.get('logic_type', 'unknown')}/{result.get('rule', 'unknown')}")
    
    def _append_to_json(self, result):
        """Append result to main JSON file"""
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
        logic_type = result['logic_type']
        rule = result['rule']
        depth = result['depth']
        
        safe_rule = re.sub(r'[^\w\-_]', '_', rule)
        response_file = f"{self.responses_dir}/{logic_type}_{depth}_{safe_rule}.txt"
        
        try:
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"Logic Type: {logic_type}\n")
                f.write(f"Rule: {rule}\n")
                f.write(f"Depth: {depth}\n")
                f.write(f"File: {result['file_path']}\n")
                f.write(f"Accuracy: {result['file_accuracy']:.2%}\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("=" * 70 + "\n")
                f.write("Individual Sample Results:\n")
                f.write("=" * 70 + "\n\n")
                
                for q_result in result['results']:
                    f.write(f"Sample {q_result['question_num']} (ID: {q_result.get('sample_id', 'N/A')}):\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Context: {q_result.get('context', 'N/A')}\n\n")
                    f.write(f"Question: {q_result['question']}\n\n")
                    
                    if 'all_responses' in result:
                        matching_response = next(
                            (r for r in result['all_responses'] if r['sample_id'] == q_result.get('sample_id')),
                            None
                        )
                        if matching_response:
                            f.write(f"GPT Response:\n{matching_response['response']}\n\n")
                    
                    f.write(f"Ground Truth: {q_result['ground_truth']}\n")
                    f.write(f"Prediction:   {q_result['prediction']}\n")
                    f.write(f"Correct:      {'✓ Yes' if q_result['correct'] else '✗ No'}\n")
                    
                    if 'error' in q_result:
                        f.write(f"Error: {q_result['error']}\n")
                    
                    f.write("\n" + "=" * 70 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")
    
    def _track_for_aggregation(self, result):
        """Track results for later aggregation"""
        logic_type = result['logic_type']
        depth = result['depth']
        
        self.results_by_logic[logic_type].append(result)
        self.results_by_depth[depth].append(result)
        self.results_by_logic_depth[logic_type][depth].append(result)
    
    def _update_progress(self, result, file_index, total_files):
        """Update progress file"""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"[{file_index+1}/{total_files}] {result.get('logic_type')}/{result.get('depth')}/{result.get('rule')}\n")
                
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    accuracy = result['file_accuracy']
                    num_questions = len(result['results'])
                    correct_count = sum(r['correct'] for r in result['results'])
                    f.write(f"  Accuracy: {correct_count}/{num_questions} ({accuracy:.2%})\n")
                
                f.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")
    
    def _save_aggregated_results(self):
        """Save aggregated results by logic type and depth"""
        print("\nGenerating aggregated summaries...")
        
        self._save_logic_summaries()
        self._save_depth_summaries()
        self._save_logic_depth_summaries()
        self._save_overall_summary()
    
    def _save_logic_summaries(self):
        """Save summaries organized by logic type"""
        for logic_type, results in self.results_by_logic.items():
            safe_logic = re.sub(r'[^\w\-_]', '_', logic_type)
            
            json_file = f"{self.by_logic_dir}/{safe_logic}_results.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            summary_file = f"{self.by_logic_dir}/{safe_logic}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Summary for Logic Type: {logic_type}\n")
                f.write("=" * 70 + "\n\n")
                
                total_questions = sum(len(r['results']) for r in results)
                total_correct = sum(sum(q['correct'] for q in r['results']) for r in results)
                
                f.write(f"Total Files: {len(results)}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {total_correct/total_questions:.2%}\n\n")
                
                f.write("Breakdown by Depth:\n")
                f.write("-" * 70 + "\n")
                depth_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                for r in results:
                    depth = r['depth']
                    for q in r['results']:
                        depth_stats[depth]['total'] += 1
                        if q['correct']:
                            depth_stats[depth]['correct'] += 1
                
                for depth in sorted(depth_stats.keys()):
                    stats = depth_stats[depth]
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    f.write(f"  {depth}: {stats['correct']}/{stats['total']} ({acc:.2%})\n")
    
    def _save_depth_summaries(self):
        """Save summaries organized by depth"""
        for depth, results in self.results_by_depth.items():
            safe_depth = re.sub(r'[^\w\-_]', '_', depth)
            
            json_file = f"{self.by_depth_dir}/{safe_depth}_results.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            summary_file = f"{self.by_depth_dir}/{safe_depth}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Summary for Depth: {depth}\n")
                f.write("=" * 70 + "\n\n")
                
                total_questions = sum(len(r['results']) for r in results)
                total_correct = sum(sum(q['correct'] for q in r['results']) for r in results)
                
                f.write(f"Total Files: {len(results)}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {total_correct/total_questions:.2%}\n\n")
                
                f.write("Breakdown by Logic Type:\n")
                f.write("-" * 70 + "\n")
                logic_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                for r in results:
                    logic = r['logic_type']
                    for q in r['results']:
                        logic_stats[logic]['total'] += 1
                        if q['correct']:
                            logic_stats[logic]['correct'] += 1
                
                for logic in sorted(logic_stats.keys()):
                    stats = logic_stats[logic]
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    f.write(f"  {logic}: {stats['correct']}/{stats['total']} ({acc:.2%})\n")
    
    def _save_logic_depth_summaries(self):
        """Save detailed breakdown by logic type AND depth"""
        summary_file = f"{self.summary_dir}/logic_depth_matrix.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Accuracy Matrix: Logic Type × Depth\n")
            f.write("=" * 70 + "\n\n")
            
            all_logics = sorted(self.results_by_logic_depth.keys())
            all_depths = set()
            for logic_results in self.results_by_logic_depth.values():
                all_depths.update(logic_results.keys())
            all_depths = sorted(all_depths)
            
            f.write(f"{'Logic Type':<30}")
            for depth in all_depths:
                f.write(f"{depth:>12}")
            f.write("\n")
            f.write("-" * 70 + "\n")
            
            for logic in all_logics:
                f.write(f"{logic:<30}")
                for depth in all_depths:
                    results = self.results_by_logic_depth[logic].get(depth, [])
                    if results:
                        total = sum(len(r['results']) for r in results)
                        correct = sum(sum(q['correct'] for q in r['results']) for r in results)
                        acc = correct / total if total > 0 else 0
                        f.write(f"{acc:>11.1%} ")
                    else:
                        f.write(f"{'N/A':>12}")
                f.write("\n")
    
    def _save_overall_summary(self):
        """Save overall summary statistics"""
        summary_file = f"{self.summary_dir}/overall_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Overall Multi-LogiEval Test Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            with open(self.detailed_file, 'r') as df:
                all_results = json.load(df)
            
            total_files = len(all_results)
            total_questions = sum(len(r.get('results', [])) for r in all_results)
            total_correct = sum(sum(q['correct'] for q in r.get('results', [])) for r in all_results)
            
            f.write(f"Total Files Tested: {total_files}\n")
            f.write(f"Total Questions: {total_questions}\n")
            f.write(f"Total Correct: {total_correct}\n")
            if total_questions > 0:
                f.write(f"Overall Accuracy: {total_correct/total_questions:.2%}\n\n")
            
            f.write("\nAccuracy by Logic Type:\n")
            f.write("-" * 70 + "\n")
            for logic_type in sorted(self.results_by_logic.keys()):
                results = self.results_by_logic[logic_type]
                total_q = sum(len(r['results']) for r in results)
                correct_q = sum(sum(q['correct'] for q in r['results']) for r in results)
                acc = correct_q / total_q if total_q > 0 else 0
                f.write(f"{logic_type:<30} {correct_q:>4}/{total_q:<4} ({acc:.2%})\n")
            
            f.write("\nAccuracy by Depth:\n")
            f.write("-" * 70 + "\n")
            for depth in sorted(self.results_by_depth.keys()):
                results = self.results_by_depth[depth]
                total_q = sum(len(r['results']) for r in results)
                correct_q = sum(sum(q['correct'] for q in r['results']) for r in results)
                acc = correct_q / total_q if total_q > 0 else 0
                f.write(f"{depth:<30} {correct_q:>4}/{total_q:<4} ({acc:.2%})\n")
    
    def finalize(self, total_questions, total_correct):
        """Write final summary and save all aggregated results"""
        try:
            self._save_aggregated_results()
            
            with open(self.progress_file, 'a') as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize results: {e}")
        
        print(f"\n{'='*70}")
        print("All results saved!")
        print(f"{'='*70}")
        print(f"Main results:        {self.detailed_file}")
        print(f"Progress log:        {self.progress_file}")
        print(f"Individual responses: {self.responses_dir}/")
        print(f"By logic type:       {self.by_logic_dir}/")
        print(f"By depth:            {self.by_depth_dir}/")
        print(f"Summaries:           {self.summary_dir}/")
        print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description='Test GPT-5 on Multi-LogiEval dataset with customizable prompts (samples 10 from each logic×depth)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test_multilogieval.py \\
      --api_key YOUR_KEY \\
      --data_dir data/MultiLogicEval \\
      --system_prompt prompts/multilogieval/lean_system.txt \\
      --user_prompt prompts/multilogieval/lean_user.txt \\
      --prompt_name lean \\
      --sample_per_combination 10
        """
    )
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--data_dir', required=True, help='Path to Multi-LogiEval data directory')
    parser.add_argument('--system_prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--user_prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--prompt_name', default='test', help='Name for output files (e.g., "lean", "cot")')
    parser.add_argument('--logic_types', nargs='+', 
                        default=['fol', 'nm', 'pl'],
                        help='Logic types to test')
    parser.add_argument('--depths', nargs='+',
                        default=['d1_Data', 'd2_Data', 'd3_Data', 'd4_Data', 'd5_Data'],
                        help='Depths to test')
    parser.add_argument('--sample_per_combination', type=int, default=10,
                        help='Number of files to sample per logic type × depth combination (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--output_dir', default='results', help='Directory to save responses')
    parser.add_argument('--model', default='gpt-5', help='Model to use')
    args = parser.parse_args()
    
    # Load prompts
    print(f"Loading prompts...")
    system_prompt_template = load_prompt(args.system_prompt)
    user_prompt_template = load_prompt(args.user_prompt)
    print(f"✓ Loaded system prompt from {args.system_prompt}")
    print(f"✓ Loaded user prompt from {args.user_prompt}")
    
    # Load dataset with sampling
    all_files = load_multilogieval_dataset(
        args.data_dir, 
        args.logic_types, 
        args.depths,
        args.sample_per_combination,
        args.seed
    )
    
    if not all_files:
        print("No files found! Check your data directory and filters.")
        return
    
    print(f"\nTesting {len(all_files)} files with {args.model}")
    print(f"Results will be saved to: {args.output_dir}")
    
    saver = IncrementalSaver(args.output_dir, args.prompt_name)
    
    total_questions = 0
    total_correct = 0
    
    try:
        for i, file_data in enumerate(all_files):
            print(f"\n{'='*70}")
            print(f"File {i+1}/{len(all_files)}")
            print(f"Logic: {file_data['logic_type']}, Rule: {file_data['rule']}, Depth: {file_data['depth']}")
            print(f"Samples: {len(file_data['samples'])}")
            print(f"{'='*70}")
            
            result = test_gpt5_multilogieval(file_data, args.api_key, 
                                            system_prompt_template, user_prompt_template,
                                            args.model)
            
            saver.save_result(result, i, len(all_files))
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print(f"\n✅ File accuracy: {result['file_accuracy']:.2%}")
            for q_result in result['results']:
                status = '✓' if q_result['correct'] else '✗'
                print(f"  Q{q_result['question_num']}: {q_result['ground_truth']} → {q_result['prediction']} {status}")
                total_questions += 1
                if q_result['correct']:
                    total_correct += 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving final results...")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}. Saving results so far...")
    
    print("\n" + "="*70)
    print("Finalizing results and generating summaries...")
    print("="*70)
    saver.finalize(total_questions, total_correct)
    
    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
        print(f"{'='*70}")
    else:
        print("\n⚠️  No questions were completed successfully.")

if __name__ == "__main__":
    main()