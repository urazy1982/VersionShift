# VersionShift v2.2.3 — From Exposure to Impact (design frozen; v2.2.2 audit + v2.2.3 integrity amendment)

**Date:** 2026-11-06. Audit decision adopted: the design is accepted with amendments; Stage B setup and oracle coding are GO; Stage B inference uses the v2.2.1 code only; Stage C is not authorised until gates G1–G7 are evaluated. See `docs/FINAL_DESIGN_V22.md` (§8 lists the v2.2.1 corrections), `docs/STAGE_B_PROTOCOL.md` and `RESPONSE_TO_REVIEW.md` (rounds 1–3).

## Tests
```bash
VS_FAST_TESTS=1 python3 -I -m unittest discover -s tests -p 'test_*.py'   # 51 run + 3 skipped, ~15 s
python3 -I -m unittest discover -s tests -p 'test_*.py'                    # all 54, ~2.5 min on 2 CPUs
```

## Frozen analysis implementation
`ANALYSIS_FREEZE_V221.json` lists SHA-256 hashes of the decision, analysis, attribution, oracle, Stage-B, controller, worker and environment-build code. Stage B inference must use exactly these files.

## Next actions (in order)
1. Code all 78 oracle opportunities (30 rows are pre-filled drafts you must confirm), then run `code/oracle_frame.py freeze`.
2. Run `STAGE=B bash crossversion/build_envs.sh` on Ubuntu.
3. Follow `docs/STAGE_B_PROTOCOL.md`.

## Version chain
v2.2.1 (author) → v2.2.2 (independent audit: selection enforcement, job-universe check, fail-closed stratification; `docs/AMENDMENT_V222.md`, `SCIENTIFIC_DECISION_V222.md`) → v2.2.3 (plan coverage in mitigation and attribution, G2 failure ledger; `docs/AMENDMENT_V223.md`). The current hashes are in `AMENDMENT_MANIFEST_V223.json`.
