"""Offline agent-oracle intent F1 evaluation.

Two-stage pipeline:
  Stage 1 (compute_intent_candidates.py) -> selected rollouts, GT events,
    pred events, threshold-free candidate pairs.
  Stage 2 (eval_intent_metrics.py)       -> per-coarse-action precision /
    recall / F1 by sweeping (w, d) thresholds against the candidate CSV.

v1 evaluates `mode="own"` only because aux/* predictions are saved only
under /obs_p{1,2}/own/ in the rollout HDF5 schema (see sc_rollout_io.py).
"""
