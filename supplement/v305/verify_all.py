#!/usr/bin/env python3
"""Run every public check: frozen release, companion manifest, aggregate summaries,
secondary-table consistency and (if numpy/scipy are installed) primary-inference reproduction."""
import pathlib, subprocess, sys, importlib.util
HERE = pathlib.Path(__file__).resolve().parent
steps = [['tools/verify_frozen_release.py'], ['tools/verify_manifest.py'], ['tools/verify_summary.py', '--root', '.'],
         ['tools/verify_secondary.py']]
if importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'):
    steps.append(['tools/reproduce_primary_inference.py'])
else:
    print('numpy/scipy not installed: skipping primary-inference reproduction (pip install -r analysis_code/ANALYSIS_REQUIREMENTS.txt)')
bad = 0
for s in steps:
    print('==>', ' '.join(s), flush=True)
    bad |= subprocess.run([sys.executable, '-I', *s], cwd=HERE).returncode != 0
print('ALL_PUBLIC_CHECKS_PASS' if not bad else 'SOME_CHECKS_FAILED'); sys.exit(1 if bad else 0)
