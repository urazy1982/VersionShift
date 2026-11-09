#!/usr/bin/env python3
"""VersionShift add-on: complete the kit's 'analyze' without editing the frozen kit.

Bug in versionshift_cpu_finish.py combine() (lines 236-240): when a merged CSV already exists, the
kit writes a gzip temp file named '<name>.csv.gz.tmp' and reads it back with pd.read_csv, which
infers compression from the '.tmp' suffix (none) and fails with UnicodeDecodeError 0x8b. It only
triggers when 'analyze' is re-run. This add-on loads the kit module unchanged, replaces combine()
with a version that performs the same checks with explicit gzip compression, removes stale
'*.csv.gz.tmp' leftovers, and then calls the kit's own assess() (which skips existing outputs).
"""
import argparse, importlib.util, sys, tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='/mnt/c/VersionShift_v2_2_5')
    ap.add_argument('--data', default=str(Path.home() / 'versionshift_data'))
    ap.add_argument('--combine-only', action='store_true', help='test: only run the corrected combine check')
    ap.add_argument('--skip-emulation', action='store_true', help='kit option: do not compute G5 (recorded as NOT_PERFORMED)')
    a = ap.parse_args()
    kit_path = Path(a.project) / 'versionshift_cpu_finish.py'
    spec = importlib.util.spec_from_file_location('vs_kit', kit_path)
    K = importlib.util.module_from_spec(spec)
    sys.argv = [str(kit_path)]
    spec.loader.exec_module(K)

    def safe_combine(c):
        import pandas as pd
        merge_dir = c['out'] / 'merged'
        merge_dir.mkdir(parents=True, exist_ok=True)
        for stale in merge_dir.glob('*.csv.gz.tmp'):
            stale.unlink()
            print('removed stale temp file', stale.name)
        result = {}
        for code, old, new, bname in K.EDGES:
            for reverse in (False, True):
                prev = bname + ('_REVERSE' if reverse else '')
                later = 'STAGE_C47_' + code + ('_REVERSE' if reverse else '')
                items = []
                for rn in (prev, later):
                    df = pd.read_csv(c['runs'] / rn / 'CROSS_RELEASE_FOLDS.csv.gz')
                    if set(df['dataset'].astype(str)) != (c['b_ids'] if rn == prev else set(c['remain'])):
                        K.fail('dataset coverage mismatch in ' + rn)
                    items.append(df)
                frame = pd.concat(items, ignore_index=True)
                if set(frame.dataset.astype(str)) != c['all_ids']:
                    K.fail('combined missing datasets')
                if frame[frame.status.eq('ok')].duplicated(['dataset', 'release', 'algo', 'policy', 'rep', 'fold']).any():
                    K.fail('duplicate fold identity in combined frame ' + code)
                if frame.status.eq('ok').sum() == 0:
                    K.fail('no good fold rows')
                dest = merge_dir / (code + ('_REVERSE' if reverse else '') + '.csv.gz')
                if dest.exists():
                    with tempfile.TemporaryDirectory() as td:
                        tmp = Path(td) / 'check.csv.gz'
                        frame.to_csv(tmp, index=False, compression='gzip')
                        same = pd.read_csv(tmp, compression='gzip').equals(pd.read_csv(dest, compression='gzip'))
                    if not same:
                        K.fail('derived CSV changed; preserve existing: ' + str(dest))
                    print('merged file verified unchanged:', dest.name)
                else:
                    frame.to_csv(dest, index=False, compression='gzip')
                    print('merged file written:', dest.name)
                result[code + ('_REVERSE' if reverse else '')] = dest
        return result

    K.combine = safe_combine
    args = argparse.Namespace(command='analyze', project=a.project, data=a.data, workers=10,
                              skip_aa=False, skip_emulation=a.skip_emulation)
    if a.combine_only:
        safe_combine(K.context(args))
        print('COMBINE CHECK OK')
        return
    K.assess(args)
    print('ANALYZE COMPLETE')


if __name__ == '__main__':
    main()
