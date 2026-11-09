# VersionShift v2.2.5: From Exposure to Impact (design frozen; hybrid validation amendment)

**Date:** 2026-11-06. **Status:**
- CPU continuation done; G2 passes at 191/192 D jobs, with 15/16 datasets retained.
- Historical-binary Stage B is **pending on Ubuntu**.
- Stage C is not authorised; GPU work has not started.

v2.2.5 leaves the primary design unchanged. It adds:
- pre-registered, calibrated **secondary lenses** B (prediction intervals), C (FDR discovery) and D (anytime-valid continuation);
- a **policy S** arm (SonarSource rule python:S6973);
- Experiment **E-MIT**.

See `docs/AMENDMENT_V225.md` and `docs/LITERATURE_UPDATE_V225.md`.

## Tests
```bash
VS_FAST_TESTS=1 python3 -I -m unittest discover -s tests -p 'test_*.py'   # 81 run + 3 skipped, ~30 s
python3 -I -m unittest discover -s tests -p 'test_*.py'                    # all 84, ~2.5 min on 2 CPUs
```

## Next actions (in order)
1. Run `STAGE=B bash crossversion/build_envs.sh` on Ubuntu, with `VS_PY37` set if 0.21.3 needs Python 3.7.
2. Code and freeze the 78-case oracle, if not done already.
3. Follow `docs/STAGE_B_PROTOCOL.md`. Step 5 now runs policies `D,E,P,S`, and step 6b adds the secondary lenses.
4. Evaluate gates G1–G7 before any Stage C decision.

## Hashes
`SHA256_MANIFEST_V225.json` covers every file. Since v2.2.4, the changed frozen-scope files are `crossversion/worker.py` and `crossversion/controller.py`: policy S was added, and D/E/P/PT behaviour is unchanged (tests in `tests/test_v225.py`). The primary analysis files are byte-identical to v2.2.4.

## Version chain
v2.2.1 → v2.2.2 (audit) → v2.2.3 (plan coverage) → v2.2.4 (job-level G2, CPU handoff) → v2.2.5 (secondary lenses, policy S, E-MIT). Earlier READMEs are kept as `README_V224.md` and `docs/README_V223_archived.md`.
