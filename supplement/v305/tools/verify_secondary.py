#!/usr/bin/env python3
"""Internal-consistency checks of the published secondary aggregates (standard library only).

Checks that task-level and summary tables agree (affected tasks, winner changes), that the typed
non-supported cells decompose exactly into 4,648 undefined-D2 and 664 D1-HGB validation records plus
166 removed SAMME.R records, and that the E5 table covers all 13 consecutive edges on 48 tasks.
"""
import csv, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1] / 'results'

def rows(name):
    with open(ROOT / 'secondary' / name, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    task = rows('modern_D0_task_level_conclusions.csv')
    summ = {(r['edge'], r['frame'], r['preprocessing']): r for r in rows('all_33_registered_D0_summaries.csv') if r['edge'] in ('E_LINEAR_DUAL', 'E_ADA_SAMME')}
    for key, s in summ.items():
        t = [r for r in task if (r['edge'], r['frame'], r['preprocessing']) == key]
        assert len(t) == int(s['selected_tasks']), key
        assert sum(r['affected_M_gt_0p5'] == 'True' for r in t) == int(s['affected_M_gt_0p5']), key
        assert sum(r['affected_any_pair_gt_0p5'] == 'True' for r in t) == int(s['affected_any_pair_gt_0p5']), key
        assert sum(r['winner_changed'] == 'True' for r in t) == int(s['winner_changes']), key
        assert abs(sum(float(r['M_pp']) for r in t) / len(t) - float(s['M_pp'])) < 1e-9, key
    reasons = rows('unsupported_and_removed_reasons.csv')
    unsup = sum(int(r['records']) for r in reasons if r['status'] == 'UNSUPPORTED')
    d2 = sum(int(r['records']) for r in reasons if r['status'] == 'UNSUPPORTED' and r['policy'] == 'D2')
    hgb = sum(int(r['records']) for r in reasons if r['status'] == 'UNSUPPORTED' and r['policy'] == 'D1' and r['algo'] == 'HGB')
    removed = sum(int(r['records']) for r in reasons if r['status'] == 'REMOVED_BEHAVIOR')
    assert (unsup, d2, hgb, removed) == (5312, 4648, 664, 166), (unsup, d2, hgb, removed)
    e5 = rows('E5_all_13_historical_edges.csv')
    assert len(e5) == 13 and all(r['selected_tasks'] == r['complete_tasks'] == '48' and r['estimators'] == '6' for r in e5)
    chain = [e5[0]['release_old']] + [r['release_new'] for r in sorted(e5, key=lambda r: r['edge'])]
    assert all(a['release_new'] == b['release_old'] for a, b in zip(sorted(e5, key=lambda r: r['edge']), sorted(e5, key=lambda r: r['edge'])[1:]))
    assert chain[0] == '0.21.3' and chain[-1] == '1.9.1'
    d1 = [r for r in rows('modern_D1_shared_support_contrasts.csv') if r['frame'] == 'evaluation' and r['preprocessing'] == 'P1']
    assert len(d1) == 2 and all(r['shared_panel_size'] == '7' and r['shared_pairs'] == '21' and r['D1_missing_original_estimator'] == 'HGB' for r in d1)
    print('SECONDARY_AGGREGATES_CONSISTENT', len(summ), 'modern groups,', len(e5), 'E5 edges')

if __name__ == '__main__':
    main()
