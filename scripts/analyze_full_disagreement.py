import json
from collections import defaultdict

# Load all_depths results
with open('results/multilogieval/all_depths/cot/all_results.json') as f:
    cot = json.load(f)
with open('results/multilogieval/all_depths/bidirectional/all_results.json') as f:
    bidir = json.load(f)

print('=== Accuracy by Depth ===')
cot_by_depth = defaultdict(lambda: {'correct': 0, 'total': 0})
bidir_by_depth = defaultdict(lambda: {'correct': 0, 'total': 0})

for r in cot:
    d = r.get('depth', 'unknown')
    cot_by_depth[d]['total'] += 1
    if r.get('correct', False):
        cot_by_depth[d]['correct'] += 1

for r in bidir:
    d = r.get('depth', 'unknown')
    bidir_by_depth[d]['total'] += 1
    if r.get('correct', False):
        bidir_by_depth[d]['correct'] += 1

print(f"{'Depth':<6} | {'CoT':>15} | {'Bidir':>15} | {'Gap':>8}")
print('-' * 50)
gaps = []
for d in sorted(cot_by_depth.keys()):
    cot_acc = cot_by_depth[d]['correct'] / cot_by_depth[d]['total'] * 100
    bidir_acc = bidir_by_depth[d]['correct'] / bidir_by_depth[d]['total'] * 100
    gap = bidir_acc - cot_acc
    gaps.append((d, gap))
    print(f"{d:<6} | {cot_by_depth[d]['correct']}/{cot_by_depth[d]['total']} ({cot_acc:.1f}%) | {bidir_by_depth[d]['correct']}/{bidir_by_depth[d]['total']} ({bidir_acc:.1f}%) | {gap:+.1f}%")

print()
print('=== Gap Divergence Analysis ===')
for i, (d, gap) in enumerate(gaps):
    if i > 0:
        prev_gap = gaps[i-1][1]
        change = gap - prev_gap
        print(f"{gaps[i-1][0]}→{d}: Gap changed {prev_gap:+.1f}% → {gap:+.1f}% (Δ={change:+.1f}%)")

print()
print('=== Accuracy by Logic Type (All Depths) ===')
cot_by_logic = defaultdict(lambda: {'correct': 0, 'total': 0})
bidir_by_logic = defaultdict(lambda: {'correct': 0, 'total': 0})

for r in cot:
    logic = r.get('logic_type', r.get('logic', 'unknown'))
    cot_by_logic[logic]['total'] += 1
    if r.get('correct', False):
        cot_by_logic[logic]['correct'] += 1

for r in bidir:
    logic = r.get('logic_type', r.get('logic', 'unknown'))
    bidir_by_logic[logic]['total'] += 1
    if r.get('correct', False):
        bidir_by_logic[logic]['correct'] += 1

print(f"{'Logic':<6} | {'CoT':>15} | {'Bidir':>15} | {'Gap':>8}")
print('-' * 50)
for logic in sorted(cot_by_logic.keys()):
    cot_acc = cot_by_logic[logic]['correct'] / cot_by_logic[logic]['total'] * 100
    bidir_acc = bidir_by_logic[logic]['correct'] / bidir_by_logic[logic]['total'] * 100
    gap = bidir_acc - cot_acc
    print(f"{logic:<6} | {cot_by_logic[logic]['correct']}/{cot_by_logic[logic]['total']} ({cot_acc:.1f}%) | {bidir_by_logic[logic]['correct']}/{bidir_by_logic[logic]['total']} ({bidir_acc:.1f}%) | {gap:+.1f}%")

print()
print('=== Accuracy by Logic Type × Depth ===')
cot_by_ld = defaultdict(lambda: {'correct': 0, 'total': 0})
bidir_by_ld = defaultdict(lambda: {'correct': 0, 'total': 0})

for r in cot:
    logic = r.get('logic_type', r.get('logic', 'unknown'))
    depth = r.get('depth', 'unknown')
    key = (logic, depth)
    cot_by_ld[key]['total'] += 1
    if r.get('correct', False):
        cot_by_ld[key]['correct'] += 1

for r in bidir:
    logic = r.get('logic_type', r.get('logic', 'unknown'))
    depth = r.get('depth', 'unknown')
    key = (logic, depth)
    bidir_by_ld[key]['total'] += 1
    if r.get('correct', False):
        bidir_by_ld[key]['correct'] += 1

logics = sorted(set(k[0] for k in cot_by_ld.keys()))
depths = sorted(set(k[1] for k in cot_by_ld.keys()))

for logic in logics:
    print(f"\n{logic.upper()}:")
    print(f"{'Depth':<6} | {'CoT':>12} | {'Bidir':>12} | {'Gap':>8}")
    print('-' * 45)
    for depth in depths:
        key = (logic, depth)
        if key in cot_by_ld:
            cot_acc = cot_by_ld[key]['correct'] / cot_by_ld[key]['total'] * 100
            bidir_acc = bidir_by_ld[key]['correct'] / bidir_by_ld[key]['total'] * 100
            gap = bidir_acc - cot_acc
            print(f"{depth:<6} | {cot_by_ld[key]['correct']}/{cot_by_ld[key]['total']} ({cot_acc:.0f}%) | {bidir_by_ld[key]['correct']}/{bidir_by_ld[key]['total']} ({bidir_acc:.0f}%) | {gap:+.0f}%")

print()
print('=== Disagreement Analysis by Depth ===')
cot_lookup = {r.get('question', str(i)): r for i, r in enumerate(cot)}
bidir_lookup = {r.get('question', str(i)): r for i, r in enumerate(bidir)}

disagree_by_depth = defaultdict(lambda: {'total': 0, 'bidir_wins': 0, 'cot_wins': 0})

for q, c_result in cot_lookup.items():
    if q in bidir_lookup:
        b_result = bidir_lookup[q]
        c_pred = c_result.get('prediction', '').lower()
        b_pred = b_result.get('final_answer', b_result.get('prediction', '')).lower()
        c_correct = c_result.get('correct', False)
        b_correct = b_result.get('correct', False)
        depth = c_result.get('depth', 'unknown')

        if c_pred != b_pred:
            disagree_by_depth[depth]['total'] += 1
            if b_correct and not c_correct:
                disagree_by_depth[depth]['bidir_wins'] += 1
            elif c_correct and not b_correct:
                disagree_by_depth[depth]['cot_wins'] += 1

print(f"{'Depth':<6} | {'Disagree':>8} | {'Bidir Wins':>12} | {'CoT Wins':>10} | {'Bidir Win%':>10}")
print('-' * 60)
for d in sorted(disagree_by_depth.keys()):
    data = disagree_by_depth[d]
    win_pct = data['bidir_wins'] / data['total'] * 100 if data['total'] > 0 else 0
    print(f"{d:<6} | {data['total']:>8} | {data['bidir_wins']:>12} | {data['cot_wins']:>10} | {win_pct:>9.1f}%")
