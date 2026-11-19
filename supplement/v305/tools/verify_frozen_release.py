#!/usr/bin/env python3
"""Run the frozen v3.0.5 verifier (repository-root verify_release.py) unchanged.

verify_release.py intentionally fails if any file outside its frozen manifest is present. This wrapper
copies exactly the files listed in the root MANIFEST_SHA256.json into a temporary directory and runs the
frozen verifier there, so post-results additions (this supplement, .github/README.md, CITATION.cff)
never alter or mask the frozen check.
"""
import json, pathlib, shutil, subprocess, sys, tempfile
REPO = pathlib.Path(__file__).resolve().parents[3]

def main():
    manifest = json.loads((REPO / 'MANIFEST_SHA256.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for rel in list(manifest) + ['MANIFEST_SHA256.json']:
            dst = tmp / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)
        r = subprocess.run([sys.executable, '-I', str(tmp / 'verify_release.py')], cwd=tmp)
    sys.exit(r.returncode)

if __name__ == '__main__':
    main()
