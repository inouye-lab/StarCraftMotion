"""Per-feature KDE bandwidths committed into code, used by the eval-time
NLL metrics so a value can be looked up without re-scanning rollouts on
every run.

Bandwidths are set to the **RMSE of model error** (`pred − GT`) measured on
a strong reference variant — the model's typical per-step error magnitude.
This puts each (agent, t) entry near the metric's informative regime
(`(err / bw)² ≈ 1`) rather than saturating in the tail (current scores too
low) or compressing near the ceiling (bw too wide).

To refresh, run `scripts/estimate_kde_bandwidths_from_rollouts.py` against a
representative trained variant and paste the printed dict here. **Do not
re-anchor mid-project**: changing the bandwidth invalidates absolute-score
comparisons across runs. Pick once per evaluation campaign.

Why fixed-and-error-scaled (not Silverman of GT σ): the previous Silverman
recipe set `bw = 1.06 · σ_GT · R^(-1/5)`, which produces bandwidths much
smaller than the model's actual error scale — pushing residuals into the
quadratic tail and dragging Kinematic scores to ~0.07. Fixed bandwidths
also remove the rollout-count coupling: scores no longer shift when R
changes between runs.
"""

from __future__ import annotations

from typing import Dict


# Measured via scripts/estimate_kde_bandwidths_from_rollouts.py against
# `hmart_intent_aux_clsft` (CAT-K fine-tuned), 100 scenarios from the
# adversarial test split, ~47M valid samples per kinematic feature and
# ~47M valid (agent, t) entries for distance-to-nearest. Teleport-gated at
# 10 cells/s. Refresh if the model class or dataset changes meaningfully.
FEATURE_BANDWIDTHS: Dict[str, float] = {
    "linear_speed_nll": 1.2020,           # cells / s
    "linear_accel_nll": 13.4893,          # cells / s² (computed but
                                          # excluded from Kinematic composite —
                                          # see kinematic_nll.py for rationale)
    "angular_speed_nll": 3.6373,          # rad / s
    "distance_to_nearest_nll": 0.4971,    # cells
}


def default_bandwidth(metric_name: str) -> float:
    """Convenience: fixed bandwidth for a registered metric.

    Raises KeyError if the metric has no recorded bandwidth — callers may
    pass an explicit override via `MetricCtx.bandwidths` to handle metrics
    not listed here.
    """
    return FEATURE_BANDWIDTHS[metric_name]
