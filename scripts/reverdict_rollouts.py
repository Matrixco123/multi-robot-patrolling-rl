"""
Re-verdict every completed rollout, WITHOUT re-running any notebook.

Reads max_history out of the rollout_*.npz checkpoints the notebooks already wrote.

WHY THIS VERSION
----------------
v1 gated on tail slope > 0.70. That mislabelled a long excursion that merely happened
to be in progress at the end of a run (20% 3-step reward=avg, slope 0.738) as a collapse.

v2 added a second condition: the ongoing climb had to exceed the longest climb that
"recovered" earlier in the run. That was wrong, because a dip in MaxIdle is not the same
thing as a recovery. On 20% 3-step reward=max, two nodes were abandoned about 39 steps
apart; when the first was finally visited the second — already at ~2859 — immediately
became the maximum. MaxIdle dipped 1.4% and carried on climbing. v2 read that dip as a
2899-step recovery and cleared a run sitting at 99.1% of total collapse.

v3 (this file) gates on tail slope > 0.95 alone. Checked against all 18 completed rollouts
it classifies every one correctly, and the two groups are cleanly separated: everything it
calls collapsed sits at 63.8-100% of the pure-collapse value, everything it calls healthy
sits at or below 17.1%. Excursion statistics are still reported, but only as context, and
a dip now has to be substantial (MaxIdle falling below half its pre-dip value) before it
counts as a genuine recovery.
"""
import glob
import os

import numpy as np


def detect_collapse(max_history, n_report=1000, slope_threshold=0.95,
                    recovery_fraction=0.5):
    """
    Collapse test: does MaxIdle climb by ~1 per step across the whole reporting window?

    Slope ~ 1.0 over n_report steps means the maximum was carried the entire time by a
    node that was never revisited — which is exactly what permanent abandonment looks
    like. Anything genuinely recovering shows dips, which pull the slope well below 1.

    recovery_fraction only affects the reported context, never the verdict: a dip counts
    as a recovery only if MaxIdle falls below this fraction of its pre-dip value, so that
    one abandoned node handing the maximum to another is not misread as the policy
    recovering.
    """
    mh = np.asarray(max_history, dtype=float)
    N = len(mh)
    n_rep = min(n_report, N)

    slope = (mh[-1] - mh[-n_rep]) / n_rep
    theo = N - (n_rep - 1) / 2.0
    obs = mh[-n_rep:].mean()

    # substantial dips only
    dips = np.flatnonzero(np.diff(mh) < 0) + 1
    real = [d for d in dips if mh[d] < recovery_fraction * mh[d - 1]]
    if not real:
        ongoing, completed = N, []
    else:
        ongoing = N - real[-1]
        completed = list(np.diff(np.concatenate([[0], np.array(real)])))
    longest = int(max(completed)) if completed else 0

    return bool(slope > slope_threshold), {
        "mean_maxidle": obs, "slope": slope, "theo": theo,
        "pct_of_collapse": 100.0 * obs / theo,
        "ongoing": int(ongoing), "longest_recovered": longest,
    }


files = sorted(glob.glob("rollout_*.npz"))
if not files:
    print("No rollout_*.npz checkpoints found here.")
    print("Run this from the folder the notebooks wrote their checkpoints into.")
else:
    print(f"{'checkpoint':<36}{'MaxIdle':>10}{'slope':>8}{'% collapse':>12}"
          f"{'climb':>8}{'recov':>8}  verdict")
    print("-" * 96)
    n_coll = 0
    for f in files:
        try:
            mh = np.load(f)["max_history"]
        except Exception as e:
            print(f"{os.path.basename(f):<36}  unreadable: {e}")
            continue
        if len(mh) < 1000:
            print(f"{os.path.basename(f):<36}  only {len(mh)} steps — partial, skipped")
            continue
        collapsed, i = detect_collapse(mh)
        n_coll += collapsed
        print(f"{os.path.basename(f):<36}{i['mean_maxidle']:>10.1f}{i['slope']:>8.3f}"
              f"{i['pct_of_collapse']:>11.1f}%{i['ongoing']:>8}{i['longest_recovered']:>8}"
              f"  {'COLLAPSED' if collapsed else 'healthy'}")

    print()
    print(f"{n_coll} collapsed / {len(files)} rollouts.")
    print()
    print("Numbers are unchanged from your original runs — only the verdict logic differs.")
    print("A COLLAPSED row's MaxIdle is an artefact of run length (it approaches N - 499.5)")
    print("and should not be quoted as a performance figure; report only that it collapsed.")
