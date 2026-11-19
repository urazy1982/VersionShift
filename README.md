# VersionShift v3.0.5 prospective CPU study release

This is a reproducibility source package, not completed v3 study outcomes. The protocol has been locally frozen. External publication/timestamp evidence must be independently authenticated before separately authorized study execution. No acceptance or publication outcome is promised.

## Exact frozen protocol

`runs/plan/FROZEN_PROTOCOL.json` is byte-identical to the locally frozen artifact. SHA-256: `7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55`.
The local `created_utc` is not an external server timestamp. Personal approval/signature records are intentionally absent; the immutable protocol contains their hash commitments only.

## Contents and limits

Included: current scientific Python source, tests, public configuration views, 83-task allocation (15 pilot, 48 evaluation, 20 external), all 83 manifest views containing 4,980 matrix commitments, 14 historical environment lock/provenance views, modern environment records, protocol/amendment, hash verifier, test dependency lock and usage instructions.

Referenced and NOT bundled: source datasets, raw row labels/keys, 4,980 matrix NPZ payloads, 77,613 historical runtime files, runtime binaries, private author authentication documents and historical study archives. No unpublished study-task outcomes are included. Metadata/integrity verification is possible offline; full controlled-host scientific replay requires the omitted licensed inputs and authentic author records. There is no claim that package-only validation replays those bytes.

Personal absolute paths were removed from derived public views; `PUBLIC_SOURCE_MAP.json` records the original SHA, public SHA and exact/view status. Exact historical E2 parameters are retained, while its private `source_path` metadata is redacted; the resulting public JSON is NOT byte-identical E2. Its original SHA is preserved in the immutable protocol. Historical lock redaction changes only provenance comments; package pins and artifact hashes remain unchanged. Source views that change any bytes are explicitly marked. The published private-host gate cannot pass using public views, and this release does not disable that gate. Restore licensed/private originals by their frozen SHA on a controlled host for execution.

## Verify and test

```bash
python3 verify_release.py
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements-test.lock
.venv/bin/python -m pytest -q
```

The package verifier uses only included files. Tests are synthetic engineering tests; the private E2/signature gate tests are retained but skipped in the privacy-filtered public package. `requirements-test.lock` is a separate public engineering-test stack, not a replacement for frozen study stacks. Install with `--require-hashes`; never substitute a historical package version silently. All model workers must have OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS and BLIS_NUM_THREADS set to 1.

See `RUN_INSTRUCTIONS.md` for controlled-host restoration and execution gates. See `THIRD_PARTY_AND_DATA_LICENSES.md` before obtaining source datasets. Preserve negative results and task-level uncertainty; folds are not independent units.

Supplemental `environments/modern/requirements-<version>.hashlock` files resolve hashes for exactly the recorded modern pins; they are release packaging aids, not replacements for original frozen file commitments.

`evidence/v26/G3E_worker.py` is the exact MIT-licensed archived six-estimator E2 source (SHA-256 `ee39adb1d9cda6030e62245dc6e10576e9b38056de94403c2355e3086431d90c`). The public verifier parses its EXPLICIT dictionary without executing the historical worker and compares all six parameter maps with the public E2 view. No original data or outcome archive is redistributed.
