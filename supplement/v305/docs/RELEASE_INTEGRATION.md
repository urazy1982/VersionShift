# How this supplement is integrated into the VersionShift repository

This supplement is an **additional post-results disclosure**. It never modifies the prospectively frozen release.

- The frozen preregistration is the GitHub release `v3.0.5` at commit `2330e78171fad7bda3d8bc69a3698204ad35a545` (protocol SHA-256 `7a62fb456f89dc58d2797e0e5ce83a456b178b4e740de93e6e9eacfeeca56f55`, release published 2026-11-17 16:40:51 UTC). Do not move, delete or re-create that tag, and do not force-push or rewrite history before it.
- Every file of the frozen release stays byte-identical at the repository root. This supplement lives only in `supplement/v305/`, plus the post-results files `.github/README.md` (repository landing page), `CITATION.cff` and `RELEASE_NOTES_v3.0.5-results.md`.
- The frozen `verify_release.py` rejects any file outside its manifest by design. Run it through `supplement/v305/tools/verify_frozen_release.py`, which copies exactly the frozen file set to a temporary directory, or check out the tag `v3.0.5`.
- Publish this supplement as a normal new commit on `main` and a new release tag `v3.0.5-results`.
- Raw predictions, labels, feature matrices, the private 1.6 GB validated archive (SHA-256 `f0e37943808537f757edde212ad53dafa833fe1d77393873aea1b7d59e3d9953`) and environment binaries are not published.
