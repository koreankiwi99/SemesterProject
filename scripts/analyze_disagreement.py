import json
from collections import defaultdict

# Load results
with open('results/multilogieval/all_depths/cot/all_results.json') as f:
    cot = json.load(f)
with open('results/multilogieval/all_depths/bidirectional/all_results.json') as f:
    bidir = json.load(f)

# Create lookup by question ID
cot_lookup = {r.get('question', r.get('id', '')): r for r in cot}
bidir_lookup = {r.get('question', r.get('id', '')): r for r in bidir}

# Find disagreements by depth
disagreements = defaultdict(list)

for q, c_result in cot_lookup.items():
    if q in bidir_lookup:
        b_result = bidir_lookup[q]
        c_pred = c_result.get('prediction', '').lower()
        b_pred = b_result.get('final_answer', b_result.get('prediction', '')).lower()
        c_correct = c_result.get('correct', False)
        b_correct = b_result.get('correct', False)
        depth = c_result.get('depth', 'unknown')

        if c_pred != b_pred:
            disagreements[depth].append({
                'cot_pred': c_pred, 'cot_correct': c_correct,
                'bidir_pred': b_pred, 'bidir_correct': b_correct,
                'ground_truth': c_result.get('answer', 'unknown')
            })

print('=== Disagreement Analysis by Depth ===')
print(f"{'Depth':<6} | {'#Disagree':>9} | {'Bidir Wins':>11} | {'CoT Wins':>9} | {'Both Wrong':>10}")
print('-' * 60)

for d in sorted(disagreements.keys()):
    total = len(disagreements[d])
    bidir_wins = sum(1 for x in disagreements[d] if x['bidir_correct'] and not x['cot_correct'])
    cot_wins = sum(1 for x in disagreements[d] if x['cot_correct'] and not x['bidir_correct'])
    both_wrong = sum(1 for x in disagreements[d] if not x['cot_correct'] and not x['bidir_correct'])
    print(f"{d:<6} | {total:>9} | {bidir_wins:>11} | {cot_wins:>9} | {both_wrong:>10}")

# Summary
all_disagree = sum(len(v) for v in disagreements.values())
all_bidir_wins = sum(sum(1 for x in v if x['bidir_correct'] and not x['cot_correct']) for v in disagreements.values())
all_cot_wins = sum(sum(1 for x in v if x['cot_correct'] and not x['bidir_correct']) for v in disagreements.values())
all_both_wrong = sum(sum(1 for x in v if not x['cot_correct'] and not x['bidir_correct']) for v in disagreements.values())
print('-' * 60)
print(f"{'Total':<6} | {all_disagree:>9} | {all_bidir_wins:>11} | {all_cot_wins:>9} | {all_both_wrong:>10}")

print()
print('=== d5 Disagreements Detail ===')
for i, x in enumerate(disagreements.get('d5', [])[:10], 1):
    c_mark = "V" if x["cot_correct"] else "X"
    b_mark = "V" if x["bidir_correct"] else "X"
    print(f'{i}. GT={x["ground_truth"]:>4} | CoT={x["cot_pred"]:>4} ({c_mark}) | Bidir={x["bidir_pred"]:>4} ({b_mark})')
