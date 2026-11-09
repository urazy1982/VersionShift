"""Independent structural validation + project freeze dry-run for ORACLE_FRAME_REVIEWED.csv.

The project's oracle_frame.py is executed UNMODIFIED (its own REQUIRED/CATEGORIES/key rules); the freeze output is
written to a scratch directory and is NOT part of the deliverables (freezing remains the author's action).
"""
import csv, hashlib, json, re, subprocess, sys, tempfile, shutil
from pathlib import Path

OUT = Path(sys.argv[1]); PROJ = Path(sys.argv[2])
frame_path = OUT / 'ORACLE_FRAME_REVIEWED.csv'
ev_path = OUT / 'ORACLE_EVIDENCE_78.csv'
working_path = Path('/mnt/user-data/uploads/ff804fd2-ORACLE_FRAME_WORKING.csv')
prov_path = Path('/mnt/user-data/uploads/d0281c58-ORACLE_FRAME_WORKING.csv.PROVENANCE.json')

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RELEASES = ['0.21.3', '0.22.2.post1', '0.23.2', '0.24.2', '1.0.2', '1.1.3', '1.2.2', '1.3.2', '1.4.2', '1.5.2', '1.6.1', '1.7.2', '1.8.0', '1.9.1']
ESTIMATORS = ['LR', 'SVC', 'kNN', 'RF', 'GB', 'HGB']
EXPECTED = {('%s->%s' % (o, n), e) for o, n in zip(RELEASES, RELEASES[1:]) for e in ESTIMATORS}
REQUIRED = ['predicted_impact_category', 'trigger_condition', 'applicable_data_types', 'estimated_compatibility',
            'release_note_reference', 'coding_uncertainty', 'coder', 'coded_on']
CATEGORIES = ['no_change_expected', 'fp_level_change_possible', 'change_possible', 'change_expected']

# ---- independent checks --------------------------------------------------------------------------------------
rows = list(csv.DictReader(open(frame_path, newline='')))
working = list(csv.DictReader(open(working_path, newline='')))
keys = [(r['edge'], r['estimator']) for r in rows]
dups = sorted({k for k in keys if keys.count(k) > 1})
missing = sorted(EXPECTED - set(keys)); unexpected = sorted(set(keys) - EXPECTED)
schema_ok = list(rows[0].keys()) == list(working[0].keys())
order_ok = keys == [(r['edge'], r['estimator']) for r in working]
missing_fields = {c: [r['edge'] + '|' + r['estimator'] for r in rows if r[c] == ''] for c in REQUIRED}
invalid_cat = [r['edge'] + '|' + r['estimator'] for r in rows if r['predicted_impact_category'] not in CATEGORIES]
draft_coder = [r['edge'] + '|' + r['estimator'] for r in rows if 'draft' in r['coder']]
status_counts = {}
for r in rows:
    status_counts[r['coding_status']] = status_counts.get(r['coding_status'], 0) + 1
cat_counts = {c: sum(r['predicted_impact_category'] == c for r in rows) for c in CATEGORIES}
unc_counts = {}
for r in rows:
    unc_counts[r['coding_uncertainty']] = unc_counts.get(r['coding_uncertainty'], 0) + 1
date_ok = all(re.fullmatch(r'\d{4}-\d{2}-\d{2}', r['coded_on']) for r in rows)
url_ok = all(all(u.strip().startswith('https://') for u in r['release_note_reference'].split(';')) for r in rows)
urls = sorted({u.strip() for r in rows for u in r['release_note_reference'].split(';')})
pipe_ok = len({r['pipeline_conditions'] for r in rows}) == 1 and rows[0]['pipeline_conditions'] == working[0]['pipeline_conditions']

# evidence file
ev = list(csv.DictReader(open(ev_path, newline='')))
ev_keys = [(r['edge'], r['estimator']) for r in ev]
ev_ok = ev_keys == keys and all(r['reviewed_predicted_impact_category'] == f['predicted_impact_category'] for r, f in zip(ev, rows))
ev_measured_blank = all(r['measured_effect'].startswith('NOT EXAMINED') for r in ev)

# draft comparison
wmap = {(r['edge'], r['estimator']): r for r in working}
changed = []
for r in rows:
    w = wmap[(r['edge'], r['estimator'])]
    if w['predicted_impact_category'] and w['predicted_impact_category'] != r['predicted_impact_category']:
        changed.append(dict(edge=r['edge'], estimator=r['estimator'], draft=w['predicted_impact_category'], reviewed=r['predicted_impact_category']))
newly = sum(1 for r in rows if not wmap[(r['edge'], r['estimator'])]['predicted_impact_category'])
compat_changed = [dict(edge=r['edge'], estimator=r['estimator'], draft=wmap[(r['edge'], r['estimator'])]['estimated_compatibility'],
                       reviewed=r['estimated_compatibility'].split(' (')[0]) for r in rows
                  if wmap[(r['edge'], r['estimator'])]['estimated_compatibility'].split(':')[0] != r['estimated_compatibility'].split(':')[0].split(' (')[0]]

# provenance checks (original files untouched)
prov = json.loads(prov_path.read_text())
prov_checks = dict(
    working_frame_sha256_matches_built_frame_sha256=(sha(working_path) == prov['built_frame_sha256']),
    builder_sha256_matches_zip_oracle_frame_py=(sha(PROJ / 'code/oracle_frame.py') == prov['builder_sha256']),
    changelog_csv_sha256_matches_zip=(sha(PROJ / 'code/changelog_oracle.csv') == prov['changelog_csv_sha256']),
    releases_match=(prov['releases'] == RELEASES), estimators_match=(prov['estimators'] == ESTIMATORS),
    expected_opportunities_78=(prov['expected_opportunities'] == 78),
    introspect_json_present_in_zip=False,  # introspect_sha256 is recorded but the JSON is not shipped
    introspect_sha256_recorded=prov['introspect_sha256'])

# ---- project freeze dry-run (unmodified oracle_frame.py; scratch output) --------------------------------------
scratch = Path(tempfile.mkdtemp(prefix='oracle_freeze_dryrun_'))
shutil.copy(frame_path, scratch / 'ORACLE_FRAME_REVIEWED.csv')
shutil.copy(prov_path, scratch / 'ORACLE_FRAME_REVIEWED.csv.PROVENANCE.json')
cmd = [sys.executable, '-I', str(PROJ / 'code/oracle_frame.py'), 'freeze', '--frame', str(scratch / 'ORACLE_FRAME_REVIEWED.csv'),
       '--out', str(scratch / 'FROZEN_DRYRUN_NOT_A_DELIVERABLE.json')]
p = subprocess.run(cmd, capture_output=True, text=True)
dry = dict(command=' '.join(c.replace(str(scratch), '<scratch>') for c in cmd), returncode=p.returncode,
           stdout=p.stdout.strip()[:800], stderr=p.stderr.strip()[:800],
           passed=(p.returncode == 0 and (scratch / 'FROZEN_DRYRUN_NOT_A_DELIVERABLE.json').exists()),
           note=('Dry run only: executed with the UNMODIFIED project script (sha256 %s) and the ORIGINAL provenance file copied beside the '
                 'reviewed frame in a scratch directory; the resulting FROZEN json was discarded. The oracle is NOT frozen, '
                 'NOT externally timestamped and NOT author-confirmed.' % sha(PROJ / 'code/oracle_frame.py')))
# also: the same freeze on the working frame, to document the baseline failure
shutil.copy(working_path, scratch / 'ORACLE_FRAME_WORKING.csv')
shutil.copy(prov_path, scratch / 'ORACLE_FRAME_WORKING.csv.PROVENANCE.json')
p0 = subprocess.run([sys.executable, '-I', str(PROJ / 'code/oracle_frame.py'), 'freeze', '--frame', str(scratch / 'ORACLE_FRAME_WORKING.csv'),
                     '--out', str(scratch / 'FROZEN_WORKING_DRYRUN.json')], capture_output=True, text=True)
baseline = dict(returncode=p0.returncode, message=(p0.stderr.strip() or p0.stdout.strip()).replace(str(scratch), '<scratch>')[:400],
                note='baseline: the supplied working frame, same unmodified script and original provenance file')
dry['stdout'] = dry['stdout'].replace(str(scratch), '<scratch>')
shutil.rmtree(scratch, ignore_errors=True)

structural_ok = (len(rows) == 78 and not dups and not missing and not unexpected and schema_ok and order_ok and
                 not any(missing_fields.values()) and not invalid_cat and not draft_coder and date_ok and url_ok and
                 pipe_ok and ev_ok and ev_measured_blank and status_counts.get('coded_complete') == 78)

high_unc = [r['edge'] + '|' + r['estimator'] for r in rows if r['coding_uncertainty'] == 'high']
disputed = [dict(edge=r['edge'], estimator=r['estimator'], note=r['disputed_or_changed_vs_draft']) for r in ev
            if r['disputed_or_changed_vs_draft'].startswith('DISPUTED') or 'Author decision' in r['disputed_or_changed_vs_draft']]

decision = 'READY_FOR_AUTHOR_CONFIRMATION' if structural_ok and all(prov_checks[k] for k in list(prov_checks)[:6]) else 'NEEDS_EVIDENCE'
report = dict(
    generated_on='2026-11-06', reviewer='Claude (Anthropic AI model claude-fable-5-1), independent blinded reviewer',
    frame_file='ORACLE_FRAME_REVIEWED.csv', frame_sha256=sha(frame_path), evidence_file='ORACLE_EVIDENCE_78.csv', evidence_sha256=sha(ev_path),
    source_working_frame_sha256=sha(working_path), source_provenance_sha256=sha(prov_path),
    row_count=len(rows), expected_rows=78, duplicates=dups, missing_keys=missing, unexpected_keys=unexpected,
    schema_identical_to_working=schema_ok, row_order_identical_to_working=order_ok, pipeline_conditions_unchanged=pipe_ok,
    coding_status_counts=status_counts, missing_required_fields={k: v for k, v in missing_fields.items() if v},
    missing_required_fields_total=sum(len(v) for v in missing_fields.values()),
    invalid_categories=invalid_cat, draft_coder_rows=draft_coder, coded_on_all_iso_dates=date_ok, all_references_https=url_ok,
    distinct_reference_urls=len(urls), predicted_impact_counts=cat_counts, coding_uncertainty_counts=unc_counts,
    high_uncertainty_rows=high_unc, draft_classifications_changed=changed, draft_rows_newly_coded=newly,
    compatibility_estimates_changed_vs_draft=len(compat_changed), compatibility_changes=compat_changed,
    evidence_file_aligned=ev_ok, measured_effects_examined=False, evidence_measured_effect_column_blank=ev_measured_blank,
    provenance_checks=prov_checks, project_freeze_dry_run=dry, project_freeze_on_working_frame=baseline,
    structural_validation_passed=structural_ok, disputed_classifications_for_author=disputed,
    author_decisions_required=[
        'Confirm or overrule each reviewed classification (78 rows); the coder field records the reviewer, not the author.',
        'Decide whether later-release notes may document earlier regressions (1.5.2->1.6.1 RF/GB coded from the 1.8.0 fix note #32259).',
        "Decide SVC 0.22.2.post1->0.23.2: official 'Changed models' listing vs documented probability=True-only RNG mechanism.",
        'Decide kNN 1.8.0->1.9.1: tie-order-only change coded no_change_expected; conservative alternative is fp_level_change_possible.',
        'Decide whether dependency-stack (SciPy/NumPy/BLAS) drift should be coded into the oracle or kept as a separate construct (reviewer kept it separate).',
        'Run controller.py introspect on all 14 releases and reconcile auto_default_diff / estimated_compatibility (reviewer values are doc-derived).',
        'Freeze (oracle_frame.py freeze) and externally timestamp AFTER confirmation; this review did not freeze.'],
    final_decision=decision,
    final_decision_meaning={'READY_FOR_AUTHOR_CONFIRMATION': 'all 78 classifications carry independently justified evidence and pass structural checks; author confirmation, freeze and external timestamp remain to be done'}.get(decision, ''))
(OUT / 'ORACLE_VALIDATION.json').write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ['row_count', 'duplicates', 'missing_keys', 'coding_status_counts', 'missing_required_fields_total',
                                          'invalid_categories', 'draft_coder_rows', 'predicted_impact_counts', 'structural_validation_passed',
                                          'final_decision']}, indent=1))
print('freeze dry run:', dry['passed'], dry['stdout'][:200], dry['stderr'][:300])
print('baseline working frame freeze:', baseline)
print('changed vs draft:', len(changed), 'newly coded:', newly, 'compat changed:', len(compat_changed))
