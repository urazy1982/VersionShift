# VersionShift v2.2.4: CPU-stage integrity handoff

This archive inherits v2.2.3's protocol and retains all code except the G2 validator adjustment. See `docs/AMENDMENT_V224.md` and `docs/STAGE_B_PROTOCOL.md`.

Fast targeted validation: `VS_FAST_TESTS=1 python -m pytest -q tests/test_v222.py tests/test_v223.py tests/test_v224.py`.

The broader test suite exceeded the execution time limit in this runtime and is NOT certified by this archive. Previous full 66/66 is v2.2.3 evidence, not a v2.2.4 claim.

No historical-binary release pair was executed here. To run Stage B, follow `docs/STAGE_B_PROTOCOL.md` in networked Ubuntu/WSL, externally timestamp the completed oracle and preserve all raw plans, predictions, run locks, and failure logs.
