"""Result saving utilities."""

import json
import os
import asyncio
from datetime import datetime
from collections import defaultdict


class BaseSaver:
    """Base class for saving experiment results."""

    def __init__(self, output_dir="results", experiment_name="test"):
        """Initialize saver with output directory and experiment name.

        Args:
            output_dir: Base output directory
            experiment_name: Name of the experiment for file naming
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        self._save_lock = asyncio.Lock()

    def _append_to_json(self, file_path, result):
        """Append a result to a JSON file.

        Args:
            file_path: Path to the JSON file
            result: Result dictionary to append
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            data.append(result)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")

    def _update_json_at_index(self, file_path, result, index):
        """Update a specific entry in a JSON file.

        Args:
            file_path: Path to the JSON file
            result: Result dictionary to update
            index: Index of the entry to update
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 0 <= index < len(data):
                data[index] = result
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                print(f"Warning: Index {index} out of range for JSON file")
        except Exception as e:
            print(f"Warning: Could not update JSON file at index {index}: {e}")


class MultiLogiEvalSaver(BaseSaver):
    """Saver for Multi-LogiEval experiment results."""

    def __init__(self, output_dir="results", prompt_name="test", resume_dir=None):
        super().__init__(output_dir, prompt_name)

        if resume_dir:
            # Resume from existing directory
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            # Create new directory
            self.base_dir = f"{output_dir}/multilogieval_{prompt_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.accuracy_table_file = f"{self.base_dir}/accuracy_table.txt"
        self.progress_file = f"{self.base_dir}/progress.txt"

        if not resume_dir:
            self._init_files()
        else:
            # Append resume marker to progress file
            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval Testing - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single result incrementally."""
        self._append_to_json(self.all_results_file, result)
        self._update_progress(result, question_index, total_questions)

    def update_result(self, result, index):
        """Update an existing result in place (for rerunning errors)."""
        self._update_json_at_index(self.all_results_file, result, index)

        # Update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"[RERUN] Updated result at index {index}\n")
                f.write(f"  Logic: {result.get('logic_type')}, Depth: {result.get('depth_dir')}, Rule: {result.get('rule')}\n")
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")
                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

        print(f"✓ Updated result at index {index}")

    def _update_progress(self, result, question_index, total_questions):
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Logic: {result.get('logic_type')}, Depth: {result.get('depth_dir')}, Rule: {result.get('rule')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, results, results_by_combination):
        """Generate final summaries."""
        self._save_summary(results)
        self._save_accuracy_table(results_by_combination)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Accuracy table: {self.accuracy_table_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"{'='*70}")

    def _save_summary(self, results):
        """Save overall summary."""
        with open(self.summary_file, 'w') as f:
            f.write("Multi-LogiEval Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_correct = sum(r['correct'] for r in results if 'error' not in r)
            total_questions = len([r for r in results if 'error' not in r])

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
            else:
                f.write("No questions completed successfully.\n\n")

            # Accuracy by logic type
            f.write("Accuracy by Logic Type:\n")
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
        """Save accuracy table in paper's Table 6 format."""
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
                        results = [r for r in results_by_combination[key] if 'error' not in r]
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


class MultiLogiEvalLeanSaver(MultiLogiEvalSaver):
    """Saver for Multi-LogiEval experiments with Lean verification."""

    def __init__(self, output_dir="results", prompt_name="lean_test", resume_dir=None):
        # Initialize base class but don't create parent directories/files
        BaseSaver.__init__(self, output_dir, prompt_name)

        if resume_dir:
            # Resume from existing directory
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            # Create Lean-specific directory structure
            self.base_dir = f"{output_dir}/multilogieval_lean_{prompt_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.responses_dir = f"{self.base_dir}/responses"
        os.makedirs(self.responses_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.accuracy_table_file = f"{self.base_dir}/accuracy_table.txt"
        self.lean_stats_file = f"{self.base_dir}/lean_verification_stats.txt"
        self.progress_file = f"{self.base_dir}/progress.txt"

        if not resume_dir:
            self._init_files()
        else:
            # Append resume marker to progress file
            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"Multi-LogiEval + Lean - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single result incrementally."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result, question_index)

        self._update_progress_lean(result, question_index, total_questions)

    def _save_individual_response(self, result, question_index):
        """Save individual response file."""
        filename = f"q{question_index + 1:03d}_{result['logic_type']}_{result['depth_dir']}_{result['rule']}.txt"
        response_file = f"{self.responses_dir}/{filename}"

        try:
            with open(response_file, 'w') as f:
                f.write("Multi-LogiEval Question with Lean Verification\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Logic Type: {result['logic_type']}\n")
                f.write(f"Depth: {result['depth']} ({result['depth_dir']})\n")
                f.write(f"Rule: {result['rule']}\n\n")

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
                    else:
                        f.write("No Lean code found.\n")

                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("Final Result:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'✓ Yes' if result['correct'] else '✗ No'}\n")
                f.write(f"Total Iterations: {result['num_iterations']}\n")

                if result.get('lean_verification'):
                    f.write(f"Final Lean: {'✓ Success' if result['lean_verification']['success'] else '✗ Failed'}\n")

        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")

    def _update_progress_lean(self, result, question_index, total_questions):
        """Update progress file with Lean info."""
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
                        f.write(f"  Lean: {'Success' if lean_success else 'Failed'}\n")
                    elif result.get('lean_code') is None:
                        f.write("  Lean: No code generated\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, results, results_by_combination):
        """Generate final summaries including Lean stats."""
        self._save_summary(results)
        self._save_accuracy_table(results_by_combination)
        self._save_lean_stats(results)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Accuracy table: {self.accuracy_table_file}")
        print(f"Lean stats:     {self.lean_stats_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

    def _save_lean_stats(self, results):
        """Save Lean verification statistics."""
        with open(self.lean_stats_file, 'w') as f:
            f.write("Lean Verification Statistics\n")
            f.write("=" * 70 + "\n\n")

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


class FOLIOSaver(BaseSaver):
    """Saver for FOLIO experiment results."""

    def __init__(self, output_dir="results", prompt_name="test", resume_dir=None):
        super().__init__(output_dir, prompt_name)

        if resume_dir:
            # Resume from existing directory
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            # Create new directory
            self.base_dir = f"{output_dir}/folio_{prompt_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.responses_dir = f"{self.base_dir}/responses"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.txt"

        os.makedirs(self.responses_dir, exist_ok=True)

        if not resume_dir:
            self._init_files()
        else:
            # Append resume marker to progress file
            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"FOLIO Testing - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, story_index, total_stories):
        """Save a single story result."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result)

        self._update_progress(result, story_index, total_stories)
        print(f"✓ Saved result for story {result.get('story_id', 'unknown')}")

    def update_result(self, result, index):
        """Update an existing result in place (for rerunning errors)."""
        self._update_json_at_index(self.all_results_file, result, index)

        if 'error' not in result:
            self._save_individual_response(result)

        # Update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"[RERUN] Updated result at index {index}\n")
                f.write(f"  Story: {result.get('story_id', 'unknown')}\n")
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    accuracy = result.get('story_accuracy', 0)
                    f.write(f"  Accuracy: {accuracy:.2%}\n")
                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

        print(f"✓ Updated result at index {index} for story {result.get('story_id', 'unknown')}")

    def _save_individual_response(self, result):
        """Save individual response file."""
        story_id = result['story_id']
        response_file = f"{self.responses_dir}/story_{story_id}_response.txt"

        try:
            with open(response_file, 'w') as f:
                f.write(f"Story ID: {story_id}\n")
                f.write(f"Premises: {result['premises']}\n\n")
                f.write("=" * 50 + "\n")
                f.write("Model Response:\n")
                f.write("=" * 50 + "\n")
                f.write(result['model_response'])
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("Questions and Results:\n")
                f.write("=" * 50 + "\n")
                for q_result in result['results']:
                    f.write(f"Q{q_result['question_num']}: {q_result['conclusion']}\n")
                    f.write(f"Ground Truth: {q_result['ground_truth']}\n")
                    f.write(f"Prediction: {q_result['prediction']}\n")
                    f.write(f"Correct: {'Yes' if q_result['correct'] else 'No'}\n\n")
        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")

    def _update_progress(self, result, story_index, total_stories):
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Story {story_index+1}/{total_stories}: {result.get('story_id', 'unknown')}\n")
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    accuracy = result['story_accuracy']
                    num_questions = len(result['results'])
                    correct_count = sum(r['correct'] for r in result['results'])
                    f.write(f"  Accuracy: {correct_count}/{num_questions} ({accuracy:.2%})\n")
                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, total_questions, total_correct):
        """Write final summary."""
        # Save to summary file
        with open(self.summary_file, 'w') as f:
            f.write("FOLIO Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
            else:
                f.write("No questions completed successfully.\n")

        # Also update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write("=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize progress: {e}")

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")


class FOLIOLeanSaver(BaseSaver):
    """Saver for FOLIO experiments with Lean verification."""

    def __init__(self, output_dir="results", prompt_name="lean_test", resume_dir=None):
        super().__init__(output_dir, prompt_name)

        if resume_dir:
            # Resume from existing directory
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            # Create new directory
            self.base_dir = f"{output_dir}/folio_lean_{prompt_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.responses_dir = f"{self.base_dir}/responses"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.lean_stats_file = f"{self.base_dir}/lean_verification_stats.txt"

        os.makedirs(self.responses_dir, exist_ok=True)

        if not resume_dir:
            self._init_files()
        else:
            # Append resume marker to progress file
            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"FOLIO + Lean - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single question result."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result)

        self._update_progress(result, question_index, total_questions)

    def _save_individual_response(self, result):
        """Save individual response file."""
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
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Story: {result.get('story_id')}, Example: {result.get('example_id')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")
                    f.write(f"  Iterations: {result['num_iterations']}\n")

                    if result.get('lean_verification'):
                        lean_success = result['lean_verification']['success']
                        f.write(f"  Lean: {'Success' if lean_success else 'Failed'}\n")
                    elif result.get('lean_code') is None:
                        f.write("  Lean: No code found\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, total_questions, total_correct, lean_stats):
        """Write final summary."""
        # Save to summary file
        with open(self.summary_file, 'w') as f:
            f.write("FOLIO + Lean Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
            else:
                f.write("No questions completed successfully.\n\n")

            f.write("Lean Verification Stats:\n")
            f.write(f"  Questions with Lean code: {lean_stats['with_code']}\n")
            f.write(f"  Successful verifications: {lean_stats['successful']}\n")
            f.write(f"  Failed verifications: {lean_stats['failed']}\n")
            f.write(f"  Average iterations: {lean_stats['avg_iterations']:.2f}\n")
            if lean_stats['with_code'] > 0:
                verification_rate = lean_stats['successful'] / lean_stats['with_code']
                f.write(f"  Verification success rate: {verification_rate:.2%}\n")

        # Also update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write("=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
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

        # Save detailed Lean stats
        self._save_lean_stats(lean_stats)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Lean stats:     {self.lean_stats_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

    def _save_lean_stats(self, lean_stats):
        """Save detailed Lean statistics."""
        with open(self.lean_stats_file, 'w') as f:
            f.write("Lean Verification Statistics\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Questions with Lean code: {lean_stats['with_code']}\n")
            f.write(f"Successful verifications: {lean_stats['successful']}\n")
            f.write(f"Failed verifications: {lean_stats['failed']}\n")
            f.write(f"Average iterations: {lean_stats['avg_iterations']:.2f}\n")

            if lean_stats['with_code'] > 0:
                verification_rate = lean_stats['successful'] / lean_stats['with_code']
                f.write(f"Verification success rate: {verification_rate:.2%}\n")


class BidirectionalSaver(BaseSaver):
    """Saver for bidirectional verification results."""

    def __init__(self, output_dir="results", prompt_name="bidirectional", resume_dir=None, dataset_name="bidirectional"):
        super().__init__(output_dir, prompt_name)
        self.dataset_name = dataset_name

        if resume_dir:
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            self.base_dir = f"{output_dir}/{dataset_name}_{prompt_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.responses_dir = f"{self.base_dir}/responses"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.txt"
        self.stats_file = f"{self.base_dir}/bidirectional_stats.txt"

        os.makedirs(self.responses_dir, exist_ok=True)

        if not resume_dir:
            self._init_files()

    def _init_files(self):
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)
        with open(self.progress_file, 'w') as f:
            f.write(f"{self.dataset_name} Bidirectional Verification - Started at {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        self._append_to_json(self.all_results_file, result)
        if 'error' not in result:
            self._save_individual_response(result)
        self._update_progress(result, question_index, total_questions)

    def _save_individual_response(self, result):
        # Support both FOLIO (story_id/example_id) and Multi-LogiEval (question_num/sample_id)
        story_id = result.get('story_id', result.get('question_num', 0))
        example_id = result.get('example_id', result.get('sample_id', 0))
        response_file = f"{self.responses_dir}/story_{story_id}_q{example_id}.txt"

        try:
            with open(response_file, 'w') as f:
                f.write(f"Story ID: {story_id}\n")
                f.write(f"Example ID: {example_id}\n")
                # Support both FOLIO (premises/conclusion) and Multi-LogiEval (context/question)
                premises = result.get('premises', result.get('context', ''))
                conclusion = result.get('conclusion', result.get('question', ''))
                f.write(f"Premises: {premises}\n")
                f.write(f"Conclusion: {conclusion}\n\n")

                f.write("=" * 70 + "\n")
                f.write("PROVE TRUE ATTEMPT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Lean Code:\n{result.get('true_lean_code', 'None')}\n\n")
                f.write(f"Verification Success: {result.get('true_proof_success', False)}\n")
                if result.get('true_errors'):
                    f.write(f"Errors: {result['true_errors']}\n")
                f.write("\n")

                f.write("=" * 70 + "\n")
                f.write("PROVE FALSE ATTEMPT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Lean Code:\n{result.get('false_lean_code', 'None')}\n\n")
                f.write(f"Verification Success: {result.get('false_proof_success', False)}\n")
                if result.get('false_errors'):
                    f.write(f"Errors: {result['false_errors']}\n")
                f.write("\n")

                f.write("=" * 70 + "\n")
                f.write("AGREEMENT ANALYSIS\n")
                f.write("=" * 70 + "\n")
                f.write(f"Agreement Pattern: {result.get('agreement_pattern', 'Unknown')}\n")
                f.write(f"Formalization Error Detected: {result.get('formalization_error', False)}\n")
                f.write(f"Used Fallback: {result.get('used_fallback', False)}\n\n")

                f.write("=" * 70 + "\n")
                f.write("FINAL RESULT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")

        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")

    def _update_progress(self, result, question_index, total_questions):
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  Story: {result.get('story_id')}, Example: {result.get('example_id')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  True proof: {'✓' if result.get('true_proof_success') else '✗'}, "
                           f"False proof: {'✓' if result.get('false_proof_success') else '✗'}\n")
                    f.write(f"  Pattern: {result.get('agreement_pattern', 'Unknown')}\n")
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, total_questions, total_correct, stats):
        """Write final summary."""
        with open(self.summary_file, 'w') as f:
            f.write("FOLIO Bidirectional Verification Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")

            f.write("Agreement Pattern Distribution:\n")
            for pattern, count in sorted(stats['patterns'].items()):
                f.write(f"  {pattern}: {count}\n")

            f.write(f"\nFormalization Errors Detected: {stats['formalization_errors']}\n")
            f.write(f"Fallback to CoT Used: {stats['fallbacks_used']}\n")

        self._save_stats(stats)

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Stats:          {self.stats_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

    def _save_stats(self, stats):
        with open(self.stats_file, 'w') as f:
            f.write("Bidirectional Verification Statistics\n")
            f.write("=" * 70 + "\n\n")

            f.write("Agreement Patterns:\n")
            f.write("-" * 40 + "\n")
            for pattern, count in sorted(stats['patterns'].items()):
                f.write(f"  {pattern}: {count}\n")

            f.write(f"\nFormalization Errors Detected: {stats['formalization_errors']}\n")
            f.write(f"(Both TRUE and FALSE proofs succeeded - contradiction)\n\n")

            f.write(f"Fallbacks to CoT: {stats['fallbacks_used']}\n")

            f.write("\nAccuracy by Pattern:\n")
            f.write("-" * 40 + "\n")
            for pattern, data in sorted(stats['accuracy_by_pattern'].items()):
                if data['total'] > 0:
                    acc = data['correct'] / data['total']
                    f.write(f"  {pattern}: {data['correct']}/{data['total']} ({acc:.2%})\n")


class ConditionsSaver(BaseSaver):
    """Saver for conditions experiment results with incremental saving."""

    def __init__(self, output_dir="results/conditions_experiment", dataset="folio",
                 conditions=None, resume_dir=None, full_dataset=False, model="gpt-5"):
        super().__init__(output_dir, dataset)
        self.dataset = dataset
        self.conditions = conditions or []
        self.model = model

        if resume_dir:
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            # Sanitize model name for directory (replace / with -)
            model_name = model.replace("/", "-").replace(":", "-")
            suffix = "_full" if full_dataset else ""
            cond_suffix = "_" + "_".join(self.conditions) if self.conditions else ""
            self.base_dir = f"{output_dir}/{model_name}_{dataset}{suffix}{cond_suffix}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.json"
        self.responses_dir = f"{self.base_dir}/responses"

        os.makedirs(self.responses_dir, exist_ok=True)

        # Results organized by condition
        self.results = {cond: [] for cond in self.conditions}
        self.completed = set()  # (case_idx, condition) tuples

        if not resume_dir:
            self._init_files()
        else:
            self._load_existing()

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        with open(self.progress_file, 'w') as f:
            f.write(f"Conditions Experiment - {self.dataset.upper()} - Started at {self.timestamp}\n")
            f.write(f"Conditions: {', '.join(self.conditions)}\n")
            f.write("=" * 70 + "\n\n")

    def _load_existing(self):
        """Load existing results for resume support."""
        if os.path.exists(self.all_results_file):
            with open(self.all_results_file, 'r') as f:
                self.results = json.load(f)

            for cond, results_list in self.results.items():
                for r in results_list:
                    case_idx = r.get('case_idx')
                    if case_idx is not None:
                        self.completed.add((case_idx, cond))

            print(f"Loaded {len(self.completed)} existing results for resume")

            # Append resume marker
            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def is_completed(self, case_idx, condition):
        """Check if a case/condition pair has been completed."""
        return (case_idx, condition) in self.completed

    async def save_result(self, result, case_idx, condition):
        """Save a single result incrementally (async-safe)."""
        async with self._save_lock:
            self.results[condition].append(result)
            self.completed.add((case_idx, condition))

            # Save full results
            with open(self.all_results_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            # Save individual response
            self._save_individual_response(result, case_idx, condition)

            # Update progress
            total_saved = sum(len(v) for v in self.results.values())
            with open(self.progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                       f"Case {case_idx}, {condition}: "
                       f"{result.get('ground_truth')} → {result.get('prediction')} "
                       f"{'✓' if result.get('correct') else '✗'} "
                       f"(total: {total_saved})\n")

    def _save_individual_response(self, result, case_idx, condition):
        """Save individual response file with full traces."""
        # Save JSON with full data
        json_file = f"{self.responses_dir}/case_{case_idx}_{condition}.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save JSON response: {e}")

        # Save human-readable TXT
        txt_file = f"{self.responses_dir}/case_{case_idx}_{condition}.txt"
        try:
            with open(txt_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write(f"CASE {case_idx} - CONDITION: {condition.upper()}\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"Story ID: {result.get('story_id', 'N/A')}\n")
                f.write(f"Example ID: {result.get('example_id', 'N/A')}\n")
                f.write(f"Ground Truth: {result.get('ground_truth')}\n")
                f.write(f"Final Prediction: {result.get('prediction')}\n")
                f.write(f"Correct: {'Yes' if result.get('correct') else 'No'}\n")
                f.write(f"Parse Status: {result.get('parse_status')}\n")
                f.write(f"Num Iterations: {result.get('num_iterations', 0)}\n")
                f.write(f"Total Retries: {result.get('total_retries', 0)}\n")

                lean_ver = result.get('lean_verification') or {}
                f.write(f"Lean Pass: {lean_ver.get('success', False)}\n")
                if lean_ver.get('errors'):
                    f.write(f"Lean Errors: {lean_ver.get('errors')}\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("ITERATIONS\n")
                f.write("=" * 70 + "\n")

                for iter_data in result.get('iterations', []):
                    f.write(f"\n{'─' * 50}\n")
                    f.write(f"ITERATION {iter_data.get('iteration', '?')}\n")
                    f.write(f"{'─' * 50}\n")
                    f.write(f"Prediction: {iter_data.get('prediction')}\n")
                    f.write(f"Parse Status: {iter_data.get('parse_status')}\n")

                    lean_v = iter_data.get('lean_verification') or {}
                    f.write(f"Lean Pass: {lean_v.get('success', 'N/A')}\n")
                    if lean_v.get('errors'):
                        f.write("Lean Errors:\n")
                        for err in lean_v.get('errors', [])[:5]:
                            f.write(f"  - {err[:300]}\n")

                    # Reasoning content (for models like DeepSeek-R1)
                    if iter_data.get('reasoning_content'):
                        f.write(f"\n--- REASONING TRACE ---\n")
                        f.write(iter_data.get('reasoning_content') + "\n")

                    f.write(f"\n--- LLM RESPONSE ---\n")
                    f.write(iter_data.get('llm_response', '') + "\n")

                    if iter_data.get('lean_code'):
                        f.write(f"\n--- LEAN CODE ---\n")
                        f.write(iter_data.get('lean_code') + "\n")

        except Exception as e:
            print(f"Warning: Could not save TXT response: {e}")

    def finalize(self):
        """Generate final summary."""
        summary = {}

        for condition in self.conditions:
            cond_results = self.results.get(condition, [])
            n_correct = sum(1 for r in cond_results if r.get('correct', False))
            n_lean_pass = sum(1 for r in cond_results
                            if r.get('lean_verification') and r['lean_verification'].get('success', False))
            n_total = len(cond_results)
            total_retries = sum(r.get('total_retries', 0) for r in cond_results)

            # Count predictions and parse statuses
            predictions = {}
            parse_statuses = {}
            for r in cond_results:
                pred = r.get('prediction') or 'None'
                predictions[pred] = predictions.get(pred, 0) + 1
                status = r.get('parse_status') or 'None'
                parse_statuses[status] = parse_statuses.get(status, 0) + 1

            summary[condition] = {
                'accuracy': n_correct / n_total if n_total > 0 else 0,
                'lean_pass_rate': n_lean_pass / n_total if n_total > 0 else 0,
                'n_correct': n_correct,
                'n_lean_pass': n_lean_pass,
                'n_total': n_total,
                'total_retries': total_retries,
                'predictions': predictions,
                'parse_statuses': parse_statuses
            }

        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Update progress file
        with open(self.progress_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"{'Condition':<15} {'Accuracy':<12} {'Lean Pass':<12} {'Retries'}\n")
            f.write("-" * 50 + "\n")
            for cond in self.conditions:
                s = summary[cond]
                f.write(f"{cond.upper():<15} {s['accuracy']*100:>5.1f}%      "
                       f"{s['lean_pass_rate']*100:>5.1f}%      {s['total_retries']}\n")

            f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

        return summary

    def get_results(self):
        return self.results


class MultiLogiEvalSaver(BaseSaver):
    """Saver for MultiLogiEval experiments with incremental JSONL saving."""

    def __init__(self, output_dir: str, model: str, depths: list, resume_dir=None):
        super().__init__(output_dir, "multilogieval")
        self.model = model
        self.depths = depths

        if resume_dir and os.path.exists(resume_dir):
            self.base_dir = resume_dir
            self._load_existing()
        else:
            model_name = model.replace("/", "-").replace(":", "-")
            self.base_dir = f"{output_dir}/{model_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)
            self._init_files()

        self.responses_dir = f"{self.base_dir}/responses"
        os.makedirs(self.responses_dir, exist_ok=True)

        # Files
        self.jsonl_file = f"{self.base_dir}/results.jsonl"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.summary_file = f"{self.base_dir}/summary.json"

    def _init_files(self):
        """Initialize output files."""
        with open(f"{self.base_dir}/progress.txt", 'w') as f:
            f.write(f"MultiLogiEval Experiment - Started at {self.timestamp}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Depths: {self.depths}\n")
            f.write("=" * 70 + "\n\n")

    def _load_existing(self):
        """Load existing results for resume support."""
        jsonl_path = f"{self.base_dir}/results.jsonl"
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        case_idx = result.get('case_idx')
                        if case_idx is not None:
                            self.completed.add(case_idx)
            print(f"Loaded {len(self.completed)} existing results for resume")

            with open(f"{self.base_dir}/progress.txt", 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def is_completed(self, case_idx: int) -> bool:
        """Check if a case has been completed."""
        return case_idx in self.completed

    async def save_result(self, result: dict, case_idx: int, case: dict):
        """Save a single result incrementally (async-safe)."""
        async with self._save_lock:
            self.completed.add(case_idx)

            # Append to JSONL (crash-safe incremental saving)
            with open(self.jsonl_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            # Save detailed individual response
            self._save_individual_response(result, case_idx, case)

            # Update progress
            lean_status = "PASS" if result.get('lean_verification', {}).get('success') else "FAIL"
            with open(self.progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                       f"Case {case_idx} ({case.get('logic_type')}/{case.get('depth')}): "
                       f"GT={result.get('ground_truth')} → Pred={result.get('prediction')} "
                       f"{'✓' if result.get('correct') else '✗'} "
                       f"Lean={lean_status} "
                       f"(iters={result.get('num_iterations')})\n")

    def _save_individual_response(self, result: dict, case_idx: int, case: dict):
        """Save detailed individual response file with full traces."""
        filename = f"case_{case_idx}_{case.get('logic_type')}_{case.get('depth')}"

        # Save full JSON with all details
        json_filepath = f"{self.responses_dir}/{filename}.json"
        with open(json_filepath, 'w') as f:
            json.dump(result, f, indent=2)

        # Also save human-readable txt
        txt_filepath = f"{self.responses_dir}/{filename}.txt"
        with open(txt_filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"CASE {case_idx}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Logic Type: {case.get('logic_type')}\n")
            f.write(f"Depth: {case.get('depth')}\n")
            f.write(f"Rule: {case.get('rule')}\n")
            f.write(f"Ground Truth: {result.get('ground_truth')}\n")
            f.write(f"Final Prediction: {result.get('prediction')}\n")
            f.write(f"Correct: {result.get('correct')}\n")
            f.write(f"Parse Status: {result.get('parse_status')}\n")
            f.write(f"Num Iterations: {result.get('num_iterations')}\n")

            lean_ver = result.get('lean_verification') or {}
            f.write(f"Lean Pass: {lean_ver.get('success', False)}\n")
            if lean_ver.get('errors'):
                f.write(f"Lean Errors: {lean_ver.get('errors')}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("CONTEXT\n")
            f.write("=" * 70 + "\n")
            f.write(case.get('context', '') + "\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("QUESTION\n")
            f.write("=" * 70 + "\n")
            f.write(case.get('question', '') + "\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("ITERATIONS\n")
            f.write("=" * 70 + "\n")

            for it in result.get('iterations', []):
                f.write(f"\n{'─' * 50}\n")
                f.write(f"ITERATION {it.get('iteration')}\n")
                f.write(f"{'─' * 50}\n")
                f.write(f"Prediction: {it.get('prediction')}\n")
                f.write(f"Parse Status: {it.get('parse_status')}\n")

                lean_v = it.get('lean_verification') or {}
                f.write(f"Lean Pass: {lean_v.get('success', 'N/A')}\n")
                if lean_v.get('errors'):
                    f.write("Lean Errors:\n")
                    for err in lean_v.get('errors', [])[:5]:
                        f.write(f"  - {err[:300]}\n")

                # Reasoning content (for models like DeepSeek-R1)
                if it.get('reasoning_content'):
                    f.write(f"\n--- REASONING TRACE ---\n")
                    f.write(it.get('reasoning_content') + "\n")

                f.write(f"\n--- LLM RESPONSE ---\n")
                f.write(it.get('llm_response', '') + "\n")

                if it.get('lean_code'):
                    f.write(f"\n--- LEAN CODE ---\n")
                    f.write(it.get('lean_code') + "\n")

    def finalize(self) -> dict:
        """Generate final summary from JSONL."""
        results = []
        if os.path.exists(self.jsonl_file):
            with open(self.jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

        # Save all_results.json
        with open(self.all_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate stats
        total = len(results)
        correct = sum(1 for r in results if r.get('correct'))
        lean_pass = sum(1 for r in results if r.get('lean_verification', {}).get('success'))

        # By depth
        by_depth = {}
        for r in results:
            d = r.get('depth', 'unknown')
            if d not in by_depth:
                by_depth[d] = {'total': 0, 'correct': 0, 'lean_pass': 0}
            by_depth[d]['total'] += 1
            if r.get('correct'):
                by_depth[d]['correct'] += 1
            if r.get('lean_verification', {}).get('success'):
                by_depth[d]['lean_pass'] += 1

        # By logic type
        by_logic = {}
        for r in results:
            lt = r.get('logic_type', 'unknown')
            if lt not in by_logic:
                by_logic[lt] = {'total': 0, 'correct': 0, 'lean_pass': 0}
            by_logic[lt]['total'] += 1
            if r.get('correct'):
                by_logic[lt]['correct'] += 1
            if r.get('lean_verification', {}).get('success'):
                by_logic[lt]['lean_pass'] += 1

        summary = {
            'model': self.model,
            'depths': self.depths,
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'lean_pass': lean_pass,
            'lean_pass_rate': lean_pass / total if total > 0 else 0,
            'by_depth': by_depth,
            'by_logic': by_logic
        }

        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save human-readable summary
        with open(f"{self.base_dir}/summary.txt", 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MULTILOGIEVAL RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Model: {self.model}\n")
            f.write(f"Depths: {self.depths}\n\n")

            if total > 0:
                f.write(f"Overall: {correct}/{total} ({100*correct/total:.1f}%)\n")
                f.write(f"Lean Pass: {lean_pass}/{total} ({100*lean_pass/total:.1f}%)\n\n")

                f.write("By Depth:\n")
                for d in sorted(by_depth.keys()):
                    s = by_depth[d]
                    acc = 100*s['correct']/s['total'] if s['total'] > 0 else 0
                    lp = 100*s['lean_pass']/s['total'] if s['total'] > 0 else 0
                    f.write(f"  {d}: {s['correct']}/{s['total']} ({acc:.1f}%) | Lean: {lp:.1f}%\n")

                f.write("\nBy Logic Type:\n")
                for lt in sorted(by_logic.keys()):
                    s = by_logic[lt]
                    acc = 100*s['correct']/s['total'] if s['total'] > 0 else 0
                    lp = 100*s['lean_pass']/s['total'] if s['total'] > 0 else 0
                    f.write(f"  {lt}: {s['correct']}/{s['total']} ({acc:.1f}%) | Lean: {lp:.1f}%\n")

            f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n{'='*60}")
        print("Results saved:")
        print(f"{'='*60}")
        print(f"JSONL:       {self.jsonl_file}")
        print(f"All results: {self.all_results_file}")
        print(f"Summary:     {self.summary_file}")
        print(f"Responses:   {self.responses_dir}/")
        print(f"{'='*60}")

        return summary


class MemorizationSaver(BaseSaver):
    """Saver for memorization test results."""

    def __init__(self, output_dir="results", experiment_name="memorization"):
        super().__init__(output_dir, experiment_name)
        self.base_dir = f"{output_dir}/{experiment_name}_{self.timestamp}"
        os.makedirs(self.base_dir, exist_ok=True)
        self.all_results_file = f"{self.base_dir}/all_results.json"

    def save_all(self, results, config):
        """Save all results at once."""
        output = {
            'config': config,
            'timestamp': self.timestamp,
            'results': results
        }

        # Calculate summaries
        for dataset in ['folio', 'multilogieval']:
            if results.get(dataset):
                valid = [r for r in results[dataset] if 'error' not in r]
                correct = sum(1 for r in valid if r['correct'])
                output[f'{dataset}_accuracy'] = correct / len(valid) if valid else 0
                output[f'{dataset}_total'] = len(valid)
                output[f'{dataset}_correct'] = correct

        with open(self.all_results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {self.all_results_file}")
        print(f"{'='*70}")


class ProverQASaver(BaseSaver):
    """Saver for ProverQA experiment results."""

    def __init__(self, output_dir="results/proverqa", difficulty="hard", model_name="gpt-5", resume_dir=None):
        super().__init__(output_dir, f"{difficulty}_{model_name}")

        if resume_dir:
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            self.base_dir = f"{output_dir}/{difficulty}/{model_name}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.difficulty = difficulty
        self.model_name = model_name

        self.all_results_file = f"{self.base_dir}/all_results.json"
        self.responses_dir = f"{self.base_dir}/responses"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.txt"

        os.makedirs(self.responses_dir, exist_ok=True)

        if not resume_dir:
            self._init_files()
        else:
            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def _init_files(self):
        """Initialize output files."""
        with open(self.all_results_file, 'w') as f:
            json.dump([], f)

        with open(self.progress_file, 'w') as f:
            f.write(f"ProverQA {self.difficulty.upper()} Testing - Started at {self.timestamp}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("=" * 70 + "\n\n")

    def save_result(self, result, question_index, total_questions):
        """Save a single question result."""
        self._append_to_json(self.all_results_file, result)

        if 'error' not in result:
            self._save_individual_response(result, question_index)

        self._update_progress(result, question_index, total_questions)

    def _save_individual_response(self, result, question_index):
        """Save individual response file."""
        response_file = f"{self.responses_dir}/q{question_index:03d}.txt"

        try:
            with open(response_file, 'w') as f:
                f.write(f"Question ID: {result['id']}\n")
                f.write(f"Difficulty: {self.difficulty}\n\n")

                f.write("Context:\n")
                f.write(result['context'] + "\n\n")

                f.write("Question:\n")
                f.write(result['question'] + "\n\n")

                f.write("Options:\n")
                for opt in result['options']:
                    f.write(f"  {opt}\n")
                f.write("\n")

                f.write("=" * 50 + "\n")
                f.write("Model Response:\n")
                f.write("=" * 50 + "\n")
                f.write(result.get('model_response', 'No response') + "\n\n")

                f.write("=" * 50 + "\n")
                f.write("Result:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")

        except Exception as e:
            print(f"Warning: Could not save individual response: {e}")

    def _update_progress(self, result, question_index, total_questions):
        """Update progress file."""
        try:
            with open(self.progress_file, 'a') as f:
                f.write(f"Question {question_index+1}/{total_questions}\n")
                f.write(f"  ID: {result.get('id', 'unknown')}\n")

                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n")
                else:
                    f.write(f"  Result: {result['ground_truth']} → {result['prediction']} "
                           f"{'✓' if result['correct'] else '✗'}\n")

                f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

    def finalize(self, total_questions, total_correct):
        """Write final summary."""
        with open(self.summary_file, 'w') as f:
            f.write(f"ProverQA {self.difficulty.upper()} Evaluation Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Difficulty: {self.difficulty}\n\n")

            if total_questions > 0:
                overall_accuracy = total_correct / total_questions
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
            else:
                f.write("No questions completed successfully.\n")

        # Also update progress file
        try:
            with open(self.progress_file, 'a') as f:
                f.write("=" * 70 + "\n")
                f.write("FINAL RESULTS\n")
                f.write("=" * 70 + "\n")
                if total_questions > 0:
                    overall_accuracy = total_correct / total_questions
                    f.write(f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})\n")
                else:
                    f.write("No questions completed successfully.\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Warning: Could not finalize progress: {e}")

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"All results:    {self.all_results_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")
