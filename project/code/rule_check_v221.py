"""Pre-data properties of the v2.2.1 sensitivity outcome pi_max and of the one-strong-pair failure mode."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_v21 import SCENARIOS, generate
from estimands_v21 import dataset_level
import decision_v221 as DV

rng = np.random.default_rng(20261108); out = {}
scen = dict(SCENARIOS); scen['C1_one_strong_pair'] = dict(SCENARIOS['N2_rng_stream_changed'])
for name, sc in scen.items():
    S = generate(rng, 20000, sc)
    if name == 'C1_one_strong_pair':
        S[:, 1, 4] += 0.04          # +4 pp on GB only -> changes the 5 GB pairs... use one pair below
    D = dataset_level(S)['pair_shift']
    if name == 'C1_one_strong_pair':
        D = dataset_level(generate(rng, 20000, sc))['pair_shift']; D[:, 0] += 4.0     # exactly one of 15 pairs +4 pp
    Mk = np.abs(D).mean(1); mx = np.abs(D).max(1)
    out[name] = dict(pi_avg_truth=float(np.mean(Mk > 0.5)), pi_max_truth=float(np.mean(mx > 0.5)),
                     pi_max_truth_at_1pp=float(np.mean(mx > 1.0)), mean_M=float(Mk.mean()))
    print(name, {k: round(v, 4) for k, v in out[name].items()})
Path('results/RULE_CHECK_V221_PIMAX.json').write_text(json.dumps(out, indent=1))
