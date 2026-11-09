"""Operating characteristics of the frozen v2.2 decision rule on the v2.1 known-truth generator.
Not a new calibration cycle: same scenarios, labels tallied per K (pre-data)."""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_v21 import SCENARIOS, generate
from estimands_v21 import dataset_level
from decision_v22 import decide


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sims', type=int, default=300); ap.add_argument('--Ks', default='16,29,48,72')
    ap.add_argument('--out', required=True); a = ap.parse_args()
    rng = np.random.default_rng(20261107); out = {}
    for name, sc in SCENARIOS.items():
        big = dataset_level(generate(rng, 20000, sc))
        truth = dict(M=float(big['M'].mean()), pi=float(np.mean(big['M'] > 0.5)),
                     max_abs_pair_mean=float(np.abs(big['pair_shift'].mean(0)).max()))
        rows = {}
        for K in [int(x) for x in a.Ks.split(',')]:
            c = Counter()
            for _ in range(a.sims):
                for lab in decide(dataset_level(generate(rng, K, sc))['pair_shift'], 0.5)['labels']:
                    c[lab] += 1
            rows[K] = {k: v / a.sims for k, v in sorted(c.items())}
        out[name] = dict(truth=truth, label_rates=rows)
        print(name, truth, rows, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
