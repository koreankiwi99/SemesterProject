import json
from collections import defaultdict

# Load d5_only results
with open('results/multilogieval/d5_only/cot/all_results.json') as f:
    cot = json.load(f)
with open('results/multilogieval/d5_only/bidirectional/all_results.json') as f:
    bidir = json.load(f)

print('=== d5_only Dataset (n=110) ===')
print(f'CoT: {sum(1 for r in cot if r.get("correct"))}/{len(cot)} ({sum(1 for r in cot if r.get("correct"))/len(cot)*100:.2f}%)')
print(f'Bidir: {sum(1 for r in bidir if r.get("correct"))}/{len(bidir)} ({sum(1 for r in bidir if r.get("correct"))/len(bidir)*100:.2f}%)')

# Create lookup by question
cot_lookup = {r.get('question', r.get('id', str(i))): r for i, r in enumerate(cot)}
bidir_lookup = {r.get('question', r.get('id', str(i))): r for i, r in enumerate(bidir)}

# Find disagreements
disagreements = []
for q, c_result in cot_lookup.items():
    if q in bidir_lookup:
        b_result = bidir_lookup[q]
        c_pred = c_result.get('prediction', '').lower()
        b_pred = b_result.get('final_answer', b_result.get('prediction', '')).lower()
        c_correct = c_result.get('correct', False)
        b_correct = b_result.get('correct', False)

        if c_pred != b_pred:
            disagreements.append({
                'cot_pred': c_pred, 'cot_correct': c_correct,
                'bidir_pred': b_pred, 'bidir_correct': b_correct,
                'ground_truth': c_result.get('ground_truth', 'unknown'),
                'logic': c_result.get('logic_type', c_result.get('logic', 'unknown'))
            })

print()
print('=== Disagreement Analysis (d5_only) ===')
total = len(disagreements)
if total == 0:
    print('No disagreements found!')
else:
    bidir_wins = sum(1 for x in disagreements if x['bidir_correct'] and not x['cot_correct'])
    cot_wins = sum(1 for x in disagreements if x['cot_correct'] and not x['bidir_correct'])
    both_wrong = sum(1 for x in disagreements if not x['cot_correct'] and not x['bidir_correct'])
    both_right = sum(1 for x in disagreements if x['cot_correct'] and x['bidir_correct'])

    print(f'Total disagreements: {total}')
    print(f'Bidir wins: {bidir_wins} ({bidir_wins/total*100:.1f}%)')
    print(f'CoT wins: {cot_wins} ({cot_wins/total*100:.1f}%)')
    print(f'Both wrong: {both_wrong}')
    print(f'Both right: {both_right}')

    # By logic type
    print()
    print('=== Disagreements by Logic Type ===')
    by_logic = defaultdict(list)
    for d in disagreements:
        by_logic[d['logic']].append(d)

    for logic, items in sorted(by_logic.items()):
        bw = sum(1 for x in items if x['bidir_correct'] and not x['cot_correct'])
        cw = sum(1 for x in items if x['cot_correct'] and not x['bidir_correct'])
        print(f'{logic}: {len(items)} disagreements | Bidir wins: {bw} | CoT wins: {cw}')

    print()
    print('=== Sample Disagreements ===')
    for i, x in enumerate(disagreements[:15], 1):
        c_mark = 'V' if x['cot_correct'] else 'X'
        b_mark = 'V' if x['bidir_correct'] else 'X'
        print(f'{i}. [{x["logic"]}] GT={x["ground_truth"]:>3} | CoT={x["cot_pred"]:>3} ({c_mark}) | Bidir={x["bidir_pred"]:>3} ({b_mark})')
