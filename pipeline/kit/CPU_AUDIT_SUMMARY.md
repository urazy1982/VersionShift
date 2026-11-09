# CPU audit and execution status

## Independently executed in this session

- `pytest` targeted v2.2.3–v2.2.5 tests: **24 passed**, 1 expected scikit-learn deprecation warning, 18.40 s. The broad test invocation exceeded the execution limit; it is **not** claimed as passed in this session.
- CPU-completion workflow structural tests: **4 passed**, covering immutable 47/16 complement, checksum binding, immutable selection rejection, 63-dataset merge/no duplicate job identities, and A/A mismatch detection.
- Real on-container `sklearn 1.8.0` A/A execution of the patched scheduler on bundled iris, 1 repetition × 5 folds: **12/12 model jobs successful**, controller collect **12/12**, prediction replay **60/60 fold scores**. This is a *small scheduler smoke check*, **not** the user's actual historical A/A controls or a scientific result.
- Rechecked user-reported Stage B summary arithmetic: early M=3.8116037 pp, one-sided 95% lower bound=2.3077071, unadjusted p=0.00077067, Holm across two edges p=0.00154134; later M=0.0383746 pp, upper bound=0.0555387; no-event 0/16 exact one-sided 95% upper prevalence=0.17075 (the posted stratified/Kish approximation was 0.17243). The explicit-policy E descriptive reduction relative to D is 87.9814%. These **are re-computations of reported summaries, not independent replay of the user's raw historical predictions**.
- `python -m py_compile` and `bash -n` succeeded on all packaged runnable files.

## Not executed here / requires WSL or externally supplied inputs

- Remaining 47-dataset Stage C model fittings (four forward/reverse runs), four full 16-dataset historical A/A controls, complete 63-dataset analyses, same-plan emulation and secondary lenses.
- Ten additional historical scikit-learn environments for the 14-release adjacent-edge longitudinal experiment; subsequent 47-dataset RQ3/RQ4 scan and numeric fixed-stack control.
- Independently authenticated external timestamp and explicit human coding confirmation for the frozen oracle.
- Delayed 25% blinded intra-rater recoding at least 14 days after 6 November 2026.
- Stage D external validation of 5–8 real artifacts: the prespecified scripts/notebooks and baseline environments have not been provided.

**The study does not need a GPU for any of the listed CPU experiments.** This package cannot execute the user's private WSL environment from the chat. The run scripts execute the work **on their machine**, producing independently auditable outputs; do not describe those runs as complete until terminal logs and artifacts exist.
