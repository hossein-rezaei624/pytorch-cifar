"""
Analysis script for CGR diagnostic logs produced by cgr_with_diag.py
(when run with --cgr_diag_log).

Computes ACROSS ALL SEEDS:
  (b.1) Cross-seed Spearman rank correlation of per-sample variance vectors
        --> mean ± std over the 10 pairs of seeds.
  (a.2) Variance vs forgetting-event Spearman correlation
        --> computed separately for EACH seed; reported as mean ± std over seeds.
  (b.2) Diagnostic table comparing CGR vs random / high-loss / low-confidence
        --> each cell computed separately for EACH seed; reported as mean ± std.
  (c)   Within-seed Spearman rank correlation between per-sample variance
        computed with a small window (E_small, default 2) and a larger window
        (E_large, default 5). Anchors the sanity check for the E=2 selection
        (Concern 4). Computed separately for EACH seed; reported as mean ± std.
  (d)   Boundary-intuition verification (Concern 2): for five selection rules,
        characterizes the end-of-training margin and correctness distribution
        of the selected samples, plus the overlap between CGR and a direct
        low-|margin| selection. Shows that CGR selects Swayamdipta-style
        ambiguous samples (variability-based, migrating from low margin to
        high margin during a single task), not persistently-near-boundary
        samples. Computed separately for EACH seed; reported as mean ± std.
        (d) has been reframed as a TRAJECTORY-OUTCOME analysis (NOT a
        geometric boundary analysis). Renamed categories:
            well_classified_end     — correct at end AND top-80% margin within class
            low_margin_correct_end  — correct at end AND bottom-20% margin within class
            misclassified_end       — incorrect at end of task-1 training
        Configurable via margin_type ('prob' | 'logit') and aggregation
        ('final' | 'last_E'). Percentile ranks use scipy.stats.rankdata with
        method='average' (ties handled) and are reported both globally and
        per-class, with mean and median.
  (d2)  Selection-time boundary diagnostics (Concern 2, extended). For five
        selection rules, characterizes selected samples DURING THAT SAMPLE'S
        DIAGNOSTIC PASS in epoch E (NOT a common end-of-epoch checkpoint —
        see cgr_with_diag.py module docstring for the timing caveat).
        Reports point-in-time snapshots AS PRIMARY, plus first-E-window
        mean/range as complementary statistics. Metrics include: strict
        sign-change rate (logit_m[:-1]*logit_m[1:] < 0), correctness, prob
        margin (signed target + predicted top-two), logit margin (signed
        target + predicted top-two), nearest pairwise |z_y - z_k| (target
        and predicted), feature-space distances d_target_signed,
        d_target_absmin, d_pred, and d on the correctly-classified subset
        (where d_target = d_pred). Requires the extended log format with
        all diag_* fields produced by the updated cgr_with_diag.py.

Usage:
    python analyze_cgr_diag.py --diag_dir cgr_diag_logs --E 4 --buffer_size 1000
    # optional: --E_small 2 --E_large 5   (defaults shown)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, rankdata


# ----------------------------- I/O -----------------------------

def load_seed_logs(diag_dir):
    paths = sorted(Path(diag_dir).glob('cgr_diag_seed*.pt'))
    if not paths:
        raise FileNotFoundError(f"No 'cgr_diag_seed*.pt' files found in {diag_dir}")
    # weights_only=False is required for torch >=2.6 to load the numpy scalars in the log.
    # Only run this script on trusted .pt files you produced yourself.
    return [torch.load(p, map_location='cpu', weights_only=False) for p in paths]


# ------------------------- Metrics ---------------------------

def variance_from_eval_confidence(log, E):
    """CGR's actual variance signal: variance of eval-mode confidence over first E epochs."""
    conf = log['cgr_confidence_by_sample']
    return conf[:E].var(dim=0).numpy()


def forgetting_events(correct):
    """Toneva-style forgetting events: # of correct -> incorrect transitions over training."""
    correct = correct.bool()
    transitions = correct[:-1] & ~correct[1:]
    return transitions.sum(dim=0).numpy()


# --------------------- (b.1) Cross-seed -----------------------

def cross_seed_spearman(logs, E):
    variances = [variance_from_eval_confidence(log, E) for log in logs]
    lens = {v.shape[0] for v in variances}
    if len(lens) != 1:
        raise ValueError(f"Variance vectors differ across seeds: {lens}")

    rhos = []
    pairs = []
    n = len(variances)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(variances[i], variances[j])
            rhos.append(r)
            pairs.append((logs[i]['seed'], logs[j]['seed']))
    return float(np.mean(rhos)), float(np.std(rhos, ddof=1)), rhos, pairs


# ---------------- (a.2) Variance vs forgetting (all seeds) ----------------

def variance_vs_forgetting_per_seed(logs, E):
    """Compute Spearman ρ between variance and forgetting events for EACH seed."""
    results = []
    for log in logs:
        variance = variance_from_eval_confidence(log, E)
        forgetting = forgetting_events(log['diag_correct'])
        r, p = spearmanr(variance, forgetting)
        results.append({'seed': log['seed'], 'rho': float(r), 'p': float(p)})
    rhos = [r['rho'] for r in results]
    return results, float(np.mean(rhos)), float(np.std(rhos, ddof=1))


# ---------------- (c) Within-seed E_small vs E_large (Concern 4) ---------

def within_seed_E_small_vs_E_large_spearman(logs, E_small, E_large):
    """Per-seed Spearman ρ between per-sample σ² computed over the first
    E_small epochs and over the first E_large epochs. Anchors the sanity
    check for the E=2 selection (AE Concern 4)."""
    if E_small >= E_large:
        raise ValueError(f"E_small ({E_small}) must be < E_large ({E_large})")
    results = []
    for log in logs:
        n_epochs = log['cgr_confidence_by_sample'].shape[0]
        if E_large > n_epochs:
            raise ValueError(f"E_large={E_large} exceeds recorded epochs ({n_epochs}) "
                             f"for seed {log['seed']}")
        v_small = variance_from_eval_confidence(log, E_small)
        v_large = variance_from_eval_confidence(log, E_large)
        r, p = spearmanr(v_small, v_large)
        results.append({'seed': log['seed'], 'rho': float(r), 'p': float(p)})
    rhos = [r['rho'] for r in results]
    return results, float(np.mean(rhos)), float(np.std(rhos, ddof=1))


# ---------------- (b.2) Diagnostic table (all seeds) ----------------------

def diagnostic_table_one_seed(log, E, buffer_size, random_seed):
    target_conf_all = log['cgr_confidence_by_sample']  # (n_epochs, n_samples)
    n_epochs = target_conf_all.shape[0]
    labels = log['diag_labels'].numpy()

    # Per-sample selection scores (first E epochs, matching CGR's window and Figure 3 b.1)
    variance = variance_from_eval_confidence(log, E)
    mean_conf_E = target_conf_all[:E].mean(dim=0).numpy()
    mean_loss_E = log['diag_loss'][:E].mean(dim=0).numpy()

    # Per-sample reporting metrics.
    #   margin: averaged over the FIRST E epochs (matches CGR's selection
    #     window and Figure 3 b.1; consistent with mean target confidence).
    #     End-of-training margin would be uninformative because the model has
    #     converged and margins are large for almost all samples.
    #   forgetting events: over ALL training epochs (Toneva's definition;
    #     needs the full trajectory — first E would give at most E-1 transitions).
    #   mean target confidence: over the FIRST E epochs (matches Figure 3 b.1
    #     and CGR's selection window).
    margin_first_E = log['diag_margin'][:E].mean(dim=0).numpy()
    forgetting = forgetting_events(log['diag_correct'])
    mean_conf_report = mean_conf_E  # = target_conf_all[:E].mean(dim=0).numpy()

    # Per-class top-K (K = buffer_size // num_classes seen in task 1)
    unique_classes = np.unique(labels[labels >= 0])
    num_classes = len(unique_classes)
    k_per_class = buffer_size // num_classes

    def top_k_per_class(score, descending=True):
        out = []
        for c in unique_classes:
            idx = np.where(labels == c)[0]
            order = np.argsort(score[idx])
            if descending:
                order = order[::-1]
            out.append(idx[order[:k_per_class]])
        return np.concatenate(out)

    rng = np.random.default_rng(random_seed)
    rules = {
        'Random':          np.concatenate([
                               rng.choice(np.where(labels == c)[0],
                                          size=min(k_per_class, (labels == c).sum()),
                                          replace=False)
                               for c in unique_classes
                           ]),
        'High loss':       top_k_per_class(mean_loss_E, descending=True),
        'High confidence': top_k_per_class(mean_conf_E, descending=True),
        'Low confidence':  top_k_per_class(mean_conf_E, descending=False),
        'CGR (variance)':  top_k_per_class(variance, descending=True),
    }

    row_dict = {}
    for name, idx in rules.items():
        row_dict[name] = {
            'mean_margin': float(margin_first_E[idx].mean()),
            'mean_forgetting': float(forgetting[idx].mean()),
            'mean_target_conf': float(mean_conf_report[idx].mean()),
        }
    return row_dict, k_per_class, num_classes


def diagnostic_table_all_seeds(logs, E, buffer_size):
    """Compute the diagnostic table per seed, then aggregate to mean ± std."""
    per_seed_rows = []
    k_per_class, num_classes = None, None
    for log in logs:
        row_dict, k, nc = diagnostic_table_one_seed(
            log, E, buffer_size,
            random_seed=int(log['seed']) if str(log['seed']).isdigit() else 0
        )
        per_seed_rows.append(row_dict)
        k_per_class, num_classes = k, nc

    # Aggregate across seeds
    rule_names = list(per_seed_rows[0].keys())
    agg = {}
    for name in rule_names:
        agg[name] = {}
        for metric in ['mean_margin', 'mean_forgetting', 'mean_target_conf']:
            vals = [seed_row[name][metric] for seed_row in per_seed_rows]
            agg[name][metric + '_mean'] = float(np.mean(vals))
            agg[name][metric + '_std']  = float(np.std(vals, ddof=1))
            agg[name][metric + '_per_seed'] = [float(v) for v in vals]
    return agg, per_seed_rows, k_per_class, num_classes


# ---------------- (d) Boundary-intuition verification (Concern 2) --------

def _top_k_per_class(scores, labels, k, descending=True):
    """Per-class top-K selection helper (respects class-balance constraint)."""
    out = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        order = np.argsort(-scores[ci] if descending else scores[ci])[:k]
        out.append(ci[order])
    return np.concatenate(out)


def _percentile_ranks(values):
    """Return percentile ranks in [0, 100], handling ties via average ranks
    (rankdata method='average'), scaled by (N-1) so min→0, max→100.
    Per GPT Point 5."""
    n = len(values)
    if n == 1:
        return np.array([50.0])
    ranks = rankdata(values, method='average') - 1
    return 100.0 * ranks / (n - 1)


def _percentile_ranks_per_class(values, labels):
    """Percentile ranks computed WITHIN each class (matches CGR's per-class
    selection budget). Per GPT Point 5."""
    out = np.zeros(len(values), dtype=np.float64)
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        out[ci] = _percentile_ranks(values[ci])
    return out


def boundary_intuition_test(logs, E, buffer_size, margin_type='prob',
                             aggregation='final', margin_pctl_threshold=20.0):
    """TRAJECTORY-OUTCOME analysis (not a geometric boundary analysis).
    For each of five selection rules, characterize the end-of-training
    correctness and margin distribution of the selected samples. Answers the
    question "what happens to CGR-selected samples by end of task-1 training?"
    — NOT "are they near the decision boundary?" (see selection_time_diagnostics
    and overlap_with_direct_boundary for the boundary-diagnostic analyses).

    Category renaming (was resolved/still_boundary/outlier):
      well_classified_end  — correct AND margin in top-80% within class
      low_margin_correct_end — correct AND margin in bottom-20% within class
      misclassified_end    — incorrect (regardless of margin)

    Args:
        margin_type: 'prob' → diag_margin (signed prob margin, p_y - max_{k!=y} p_k)
                     'logit' → diag_logit_margin (signed logit margin, z_y - max_{k!=y} z_k)
                     Both share sign but differ in magnitude and cross-sample ranks.
        aggregation: 'final' → margin[n_epochs-1] and correct[n_epochs-1]
                     'last_E' → margin over the last E epochs (mean), correctness over
                                the last E epochs (mean rate; samples deemed
                                "correct-in-last-E" if correct in >= ceil(E/2) of those epochs).
        margin_pctl_threshold: default 20.0 matches CGR's selection budget
                     (K/N = 100/500 = 20% within each class).
    """
    if margin_type not in ('prob', 'logit'):
        raise ValueError(f"margin_type must be 'prob' or 'logit', got {margin_type}")
    if aggregation not in ('final', 'last_E'):
        raise ValueError(f"aggregation must be 'final' or 'last_E', got {aggregation}")

    margin_key = 'diag_margin' if margin_type == 'prob' else 'diag_logit_margin'
    per_seed_records = []

    for log in logs:
        conf   = log['cgr_confidence_by_sample'].numpy()
        margin = log[margin_key].numpy()
        correct = log['diag_correct'].numpy().astype(bool)
        loss   = log['diag_loss'].numpy()
        labels = log['diag_labels'].numpy()
        seed   = int(log['seed']) if str(log['seed']).isdigit() else 0

        n_epochs = conf.shape[0]

        if aggregation == 'final':
            end_margin = margin[n_epochs - 1]
            end_correct = correct[n_epochs - 1]
        else:  # 'last_E'
            end_margin = margin[n_epochs - E : n_epochs].mean(axis=0)
            end_correct = correct[n_epochs - E : n_epochs].mean(axis=0) >= 0.5

        n_classes = len(np.unique(labels))
        per_class = buffer_size // n_classes

        # Per-class low-margin threshold (matches CGR's class-balanced selection).
        low_thr_per_class = np.zeros(len(labels))
        for c in np.unique(labels):
            ci = np.where(labels == c)[0]
            low_thr_per_class[ci] = np.percentile(end_margin[ci], margin_pctl_threshold)

        pctl_global = _percentile_ranks(end_margin)
        pctl_class  = _percentile_ranks_per_class(end_margin, labels)

        rng = np.random.default_rng(seed)
        sigma2 = np.var(conf[:E], axis=0)
        mean_conf_E = conf[:E].mean(axis=0)
        mean_loss_E = loss[:E].mean(axis=0)
        rules = {
            'CGR (high variance)': _top_k_per_class(sigma2, labels, per_class, descending=True),
            'Random': np.concatenate([
                rng.choice(np.where(labels == c)[0],
                           size=min(per_class, (labels == c).sum()),
                           replace=False) for c in np.unique(labels)]),
            'High confidence': _top_k_per_class(mean_conf_E, labels, per_class, descending=True),
            'Low confidence':  _top_k_per_class(mean_conf_E, labels, per_class, descending=False),
            'High loss':       _top_k_per_class(mean_loss_E, labels, per_class, descending=True),
        }

        seed_row = {}
        for name, idx in rules.items():
            sfm = end_margin[idx]
            sfc = end_correct[idx]
            in_low = sfm < low_thr_per_class[idx]
            seed_row[name] = {
                'frac_correct_end':          float(sfc.mean()),
                'median_pctl_global':        float(np.median(pctl_global[idx])),
                'mean_pctl_global':          float(np.mean(pctl_global[idx])),
                'median_pctl_class':         float(np.median(pctl_class[idx])),
                'mean_pctl_class':           float(np.mean(pctl_class[idx])),
                'well_classified_end':       float((sfc & ~in_low).mean()),
                'low_margin_correct_end':    float((sfc &  in_low).mean()),
                'misclassified_end':         float((~sfc).mean()),
            }
        per_seed_records.append(seed_row)

    agg = {}
    metrics = ['frac_correct_end', 'median_pctl_global', 'mean_pctl_global',
               'median_pctl_class', 'mean_pctl_class',
               'well_classified_end', 'low_margin_correct_end', 'misclassified_end']
    for name in per_seed_records[0]:
        agg[name] = {}
        for metric in metrics:
            vals = [r[name][metric] for r in per_seed_records]
            agg[name][f'{metric}_mean'] = float(np.mean(vals))
            agg[name][f'{metric}_std']  = float(np.std(vals, ddof=1))
    return agg, per_seed_records


def overlap_with_direct_boundary(logs, E, buffer_size):
    """Fraction of CGR-selected samples that also appear in an alternative
    per-class bottom-K selection under various "direct boundary proximity"
    criteria. Small overlap means CGR's variability-based selection differs
    from what these criteria would pick.

    Reports overlap for five criteria, each computed both AT epoch E (index
    E-1, i.e. from that sample's diagnostic pass in epoch E) and OVER the
    first-E window (mean across epochs 0..E-1):
      d_pred:    feature-space signed distance to predicted-region boundary
      m_pred:    predicted top-two logit gap z_yhat - max_{k!=yhat} z_k (>=0)
      nearest_pair_logit_gap: min_{k!=y} |z_y - z_k| (>=0, target-based)
      abs_m_tgt: |z_y - max_{k!=y} z_k|  (>=0)
      signed_m_tgt: z_y - max_{k!=y} z_k (signed; smallest = most confidently
                    misclassified, NOT closest boundary — kept for comparison)

    Requires the extended log format with diag_logit_margin_pred,
    diag_nearest_pair_logit_gap, and diag_feat_dist_pred present.
    """
    criteria_names = ['d_pred', 'm_pred', 'nearest_pair_gap',
                      'abs_m_tgt', 'signed_m_tgt']
    result = {f'{c}_at_E': [] for c in criteria_names}
    result.update({f'{c}_over_E': [] for c in criteria_names})

    for log in logs:
        for req in ('diag_logit_margin_pred', 'diag_nearest_pair_logit_gap',
                    'diag_feat_dist_pred', 'diag_logit_margin'):
            if req not in log:
                raise KeyError(f"Log missing '{req}'. Re-run cgr_with_diag.py "
                               "after the Concern-2 extended update.")

        conf     = log['cgr_confidence_by_sample'].numpy()
        labels   = log['diag_labels'].numpy()
        d_pred   = log['diag_feat_dist_pred'].numpy()
        m_pred   = log['diag_logit_margin_pred'].numpy()
        nearest  = log['diag_nearest_pair_logit_gap'].numpy()
        logit_m  = log['diag_logit_margin'].numpy()

        n_classes = len(np.unique(labels))
        per_class = buffer_size // n_classes
        sigma2 = np.var(conf[:E], axis=0)
        cgr_sel = set(_top_k_per_class(sigma2, labels, per_class, descending=True))

        # Score = the quantity by which we pick the "smallest per class".
        # For always-non-negative quantities, smallest = nearest to boundary.
        # For signed_m_tgt, smallest = most confidently misclassified.
        score_at_E = {
            'd_pred':           d_pred[E-1],
            'm_pred':           m_pred[E-1],
            'nearest_pair_gap': nearest[E-1],
            'abs_m_tgt':        np.abs(logit_m[E-1]),
            'signed_m_tgt':     logit_m[E-1],
        }
        score_over_E = {
            'd_pred':           d_pred[:E].mean(axis=0),
            'm_pred':           m_pred[:E].mean(axis=0),
            'nearest_pair_gap': nearest[:E].mean(axis=0),
            'abs_m_tgt':        np.abs(logit_m[:E]).mean(axis=0),
            'signed_m_tgt':     logit_m[:E].mean(axis=0),
        }

        for name, sc in score_at_E.items():
            alt = set(_top_k_per_class(sc, labels, per_class, descending=False))
            result[f'{name}_at_E'].append(len(cgr_sel & alt) / len(cgr_sel))
        for name, sc in score_over_E.items():
            alt = set(_top_k_per_class(sc, labels, per_class, descending=False))
            result[f'{name}_over_E'].append(len(cgr_sel & alt) / len(cgr_sel))

    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in result.items()}


def cgr_margin_trajectory(logs, E, buffer_size, late_window=None):
    """For CGR-selected samples, compute mean margin early (first E epochs) vs
    late (last `late_window` epochs). Descriptive training-dynamics result —
    NOT a geometric boundary verification (see overlap_with_direct_boundary
    and selection_time_diagnostics for those).

    By default late_window equals E for symmetry with the early window.
    Uses both prob and logit margin.
    """
    if late_window is None:
        late_window = E

    out = {}
    for label, key in [('prob', 'diag_margin'), ('logit', 'diag_logit_margin')]:
        early_all, late_all = [], []
        for log in logs:
            conf = log['cgr_confidence_by_sample'].numpy()
            m = log[key].numpy()
            labels = log['diag_labels'].numpy()
            n_epochs = m.shape[0]
            n_classes = len(np.unique(labels))
            per_class = buffer_size // n_classes
            sigma2 = np.var(conf[:E], axis=0)
            selected = _top_k_per_class(sigma2, labels, per_class, descending=True)
            early_all.append(float(m[:E, selected].mean()))
            late_all.append(float(m[n_epochs - late_window : n_epochs, selected].mean()))
        out[label] = {
            'early_mean': float(np.mean(early_all)),
            'early_std':  float(np.std(early_all, ddof=1)),
            'late_mean':  float(np.mean(late_all)),
            'late_std':   float(np.std(late_all, ddof=1)),
            'late_window': late_window,
        }
    return out


# ---------------- (d2) Selection-time boundary diagnostics (Concern 2, extended) ----

def selection_time_diagnostics(logs, E, buffer_size):
    """Per selection rule, characterize the samples PRIMARILY during each
    sample's diagnostic pass in epoch E (see cgr_with_diag.py module docstring for the timing caveat: this is NOT a common end-of-epoch
    checkpoint), and COMPLEMENTARILY over the first-E window (mean across
    epochs 0..E-1). Uses all margin variants + feature-space distances +
    correctly-classified-subset d.

    Sign-change definition (strict, per GPT Point 4):
        (logit_m[:E-1] * logit_m[1:E] < 0).any(axis=0)
    counts a change only when the sign STRICTLY crosses zero.

    Requires the extended log format with keys diag_margin, diag_margin_pred,
    diag_logit_margin, diag_logit_margin_pred, diag_nearest_pair_logit_gap,
    diag_nearest_pair_logit_gap_pred, diag_feat_dist_target,
    diag_feat_dist_target_absmin, diag_feat_dist_pred (produced by the updated
    cgr_with_diag.py).
    """
    required = ['diag_margin', 'diag_margin_pred',
                'diag_logit_margin', 'diag_logit_margin_pred',
                'diag_nearest_pair_logit_gap', 'diag_nearest_pair_logit_gap_pred',
                'diag_feat_dist_target', 'diag_feat_dist_target_absmin',
                'diag_feat_dist_pred', 'diag_correct']

    rules = ['CGR (high variance)', 'Random', 'High confidence',
             'Low confidence', 'High loss']

    # Metric keys: <at_E | over_E>_<metric_name>. See paper insertion.
    metric_names_at_E = [
        'sign_change_strict',   # strict sign-change during epochs 0..E-1
        'frac_correct_at_E',    # correctness during diagnostic pass in epoch E (diag_correct[E-1])
        'prob_margin_signed',   # p_y - max_{k!=y} p_k
        'prob_margin_pred',     # p_yhat - max_{k!=yhat} p_k (>=0)
        'logit_margin_signed',  # z_y - max_{k!=y} z_k
        'abs_logit_margin',     # |z_y - max_{k!=y} z_k|
        'logit_margin_pred',    # z_yhat - max_{k!=yhat} z_k (>=0)
        'nearest_pair_gap',     # min_{k!=y}    |z_y - z_k|
        'nearest_pair_gap_pred',# min_{k!=yhat} |z_yhat - z_k|
        'd_target_signed',      # signed feat-space distance to target
        'd_target_absmin',      # feat-space distance to nearest tgt-vs-competitor
        'd_pred',               # feat-space distance to predicted-region boundary
        'd_on_correct_subset',  # d = d_pred = d_target on correctly-classified subset only
        'frac_low_absm_class',  # per-class bottom-20% by |logit margin| membership
        'frac_low_dpred_class', # per-class bottom-20% by d_pred membership
    ]
    metric_names_over_E = [
        'prob_margin_signed_mean', 'prob_margin_signed_range',
        'prob_margin_pred_mean',   'prob_margin_pred_range',
        'logit_margin_signed_mean','logit_margin_signed_range',
        'abs_logit_margin_mean',   'abs_logit_margin_range',
        'logit_margin_pred_mean',  'logit_margin_pred_range',
        'nearest_pair_gap_mean',   'nearest_pair_gap_range',
        'nearest_pair_gap_pred_mean','nearest_pair_gap_pred_range',
        'd_target_signed_mean',    'd_target_signed_range',
        'd_target_absmin_mean',    'd_target_absmin_range',
        'd_pred_mean',             'd_pred_range',
    ]

    per_seed = {r: {k: [] for k in metric_names_at_E + metric_names_over_E}
                for r in rules}
    ovl_keys = ['CGR_vs_low_absm_at_E', 'CGR_vs_low_dpred_at_E',
                'CGR_vs_low_nearest_pair_gap_at_E',
                'CGR_vs_low_m_pred_at_E']
    ovl_all = {k: [] for k in ovl_keys}

    for log in logs:
        for req in required:
            if req not in log:
                raise KeyError(f"Log for seed {log.get('seed', '?')} missing "
                               f"'{req}'. Re-run cgr_with_diag.py after the "
                               "Concern-2 extended update.")

        conf     = log['cgr_confidence_by_sample'].numpy()
        loss     = log['diag_loss'].numpy()
        labels   = log['diag_labels'].numpy()
        correct  = log['diag_correct'].numpy().astype(bool)
        prob_m   = log['diag_margin'].numpy()
        prob_m_p = log['diag_margin_pred'].numpy()
        logit_m  = log['diag_logit_margin'].numpy()
        logit_mp = log['diag_logit_margin_pred'].numpy()
        near_g   = log['diag_nearest_pair_logit_gap'].numpy()
        near_gp  = log['diag_nearest_pair_logit_gap_pred'].numpy()
        d_tgt_s  = log['diag_feat_dist_target'].numpy()
        d_tgt_a  = log['diag_feat_dist_target_absmin'].numpy()
        d_prd    = log['diag_feat_dist_pred'].numpy()
        seed = int(log['seed']) if str(log['seed']).isdigit() else 0

        n_classes = len(np.unique(labels))
        per_class = buffer_size // n_classes

        # Selection rules
        rng = np.random.default_rng(seed)
        sigma2 = np.var(conf[:E], axis=0)
        mean_conf_E = conf[:E].mean(axis=0)
        mean_loss_E = loss[:E].mean(axis=0)
        selections = {
            'CGR (high variance)': _top_k_per_class(sigma2, labels, per_class, descending=True),
            'Random': np.concatenate([
                rng.choice(np.where(labels == c)[0],
                           size=min(per_class, (labels == c).sum()),
                           replace=False) for c in np.unique(labels)]),
            'High confidence': _top_k_per_class(mean_conf_E, labels, per_class, descending=True),
            'Low confidence':  _top_k_per_class(mean_conf_E, labels, per_class, descending=False),
            'High loss':       _top_k_per_class(mean_loss_E, labels, per_class, descending=True),
        }

        # ---- Point-in-time snapshots at epoch index E-1 ----
        prob_m_E   = prob_m[E-1];   prob_mp_E = prob_m_p[E-1]
        logit_m_E  = logit_m[E-1];  logit_mp_E = logit_mp[E-1]
        near_g_E   = near_g[E-1];   near_gp_E  = near_gp[E-1]
        d_tgt_s_E  = d_tgt_s[E-1];  d_tgt_a_E  = d_tgt_a[E-1]
        d_prd_E    = d_prd[E-1]
        corr_E     = correct[E-1]
        abs_lm_E   = np.abs(logit_m_E)

        # Strict sign change over first E epochs (per GPT Point 4)
        strict_sign_change = np.any(
            logit_m[:E-1] * logit_m[1:E] < 0, axis=0
        )

        # Per-class bottom-20% membership indicators
        def _bot20(scores, labels):
            out = np.zeros(len(labels), dtype=bool)
            for c in np.unique(labels):
                ci = np.where(labels == c)[0]
                out[ci] = scores[ci] <= np.percentile(scores[ci], 20)
            return out
        low_absm_class = _bot20(abs_lm_E, labels)
        low_dp_class   = _bot20(d_prd_E, labels)

        # ---- Window aggregates over first E epochs ----
        def _mean_range_over_E(arr_epochs_samples, idx):
            """Given (E, N) array and indices, return per-selection mean and
            range (max-min per sample, then averaged)."""
            slice_ = arr_epochs_samples[:E, idx]  # (E, |idx|)
            per_sample_mean = slice_.mean(axis=0)
            per_sample_range = slice_.max(axis=0) - slice_.min(axis=0)
            return float(per_sample_mean.mean()), float(per_sample_range.mean())

        for name, idx in selections.items():
            row = per_seed[name]
            # AT epoch E
            row['sign_change_strict'].append(float(strict_sign_change[idx].mean()))
            row['frac_correct_at_E'].append(float(corr_E[idx].mean()))
            row['prob_margin_signed'].append(float(prob_m_E[idx].mean()))
            row['prob_margin_pred'].append(float(prob_mp_E[idx].mean()))
            row['logit_margin_signed'].append(float(logit_m_E[idx].mean()))
            row['abs_logit_margin'].append(float(abs_lm_E[idx].mean()))
            row['logit_margin_pred'].append(float(logit_mp_E[idx].mean()))
            row['nearest_pair_gap'].append(float(near_g_E[idx].mean()))
            row['nearest_pair_gap_pred'].append(float(near_gp_E[idx].mean()))
            row['d_target_signed'].append(float(d_tgt_s_E[idx].mean()))
            row['d_target_absmin'].append(float(d_tgt_a_E[idx].mean()))
            row['d_pred'].append(float(d_prd_E[idx].mean()))
            # d on correctly-classified subset (d_pred == d_target there)
            idx_corr = idx[corr_E[idx]]
            row['d_on_correct_subset'].append(
                float(d_prd_E[idx_corr].mean()) if len(idx_corr) > 0 else float('nan'))
            row['frac_low_absm_class'].append(float(low_absm_class[idx].mean()))
            row['frac_low_dpred_class'].append(float(low_dp_class[idx].mean()))
            # OVER first-E window (mean and range across epochs, per sample)
            for arr, key_base in [
                (prob_m,           'prob_margin_signed'),
                (prob_m_p,         'prob_margin_pred'),
                (logit_m,          'logit_margin_signed'),
                (np.abs(logit_m),  'abs_logit_margin'),
                (logit_mp,         'logit_margin_pred'),
                (near_g,           'nearest_pair_gap'),
                (near_gp,          'nearest_pair_gap_pred'),
                (d_tgt_s,          'd_target_signed'),
                (d_tgt_a,          'd_target_absmin'),
                (d_prd,            'd_pred'),
            ]:
                mn, rg = _mean_range_over_E(arr, idx)
                row[f'{key_base}_mean'].append(mn)
                row[f'{key_base}_range'].append(rg)

        # Overlaps: CGR vs bottom-K under various criteria during epoch E's diagnostic pass
        cgr_set = set(selections['CGR (high variance)'])
        for name, sc in [
            ('CGR_vs_low_absm_at_E',              abs_lm_E),
            ('CGR_vs_low_dpred_at_E',             d_prd_E),
            ('CGR_vs_low_nearest_pair_gap_at_E',  near_g_E),
            ('CGR_vs_low_m_pred_at_E',            logit_mp_E),
        ]:
            alt = set(_top_k_per_class(sc, labels, per_class, descending=False))
            ovl_all[name].append(len(cgr_set & alt) / len(cgr_set))

    # Aggregate mean±std with ddof=1
    agg = {}
    for name, dd in per_seed.items():
        agg[name] = {}
        for k, v in dd.items():
            vals = np.array(v, dtype=float)
            # Handle NaN (e.g., d_on_correct_subset when no correct samples)
            good = vals[~np.isnan(vals)]
            agg[name][k] = (float(np.mean(good)) if len(good) > 0 else float('nan'),
                             float(np.std(good, ddof=1)) if len(good) > 1 else float('nan'))
    ovl = {k: (float(np.mean(v)), float(np.std(v, ddof=1)))
           for k, v in ovl_all.items()}
    return agg, ovl


# ------------------------- Reporting -------------------------

def print_b1(mean_rho, std_rho, all_rhos, pairs, n_seeds):
    n_pairs = len(all_rhos)
    print(f"\n=== (b.1) Cross-seed Spearman correlation of variance vectors ===")
    print(f"Number of seeds: {n_seeds}  ({n_pairs} pairs)")
    print(f"Mean ρ ± std: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"Per-pair ρ values:")
    for (s1, s2), r in zip(pairs, all_rhos):
        print(f"  (seed {s1}, seed {s2}): ρ = {r:.4f}")
    print(f"\n  Paper insertion: \\bar\\rho = {mean_rho:.2f} \\pm {std_rho:.2f}")


def print_a2(results, mean_rho, std_rho):
    print(f"\n=== (a.2) Variance vs forgetting events (ALL seeds) ===")
    print(f"Per-seed ρ values:")
    for r in results:
        sig = '***' if r['p'] < 1e-50 else ('**' if r['p'] < 1e-10 else '')
        print(f"  seed {r['seed']}: ρ = {r['rho']:.4f}  (p = {r['p']:.3e})  {sig}")
    print(f"\nMean ρ ± std over {len(results)} seeds: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"\n  Paper insertion: \\rho = {mean_rho:.2f} \\pm {std_rho:.2f}")


def print_d(agg, margin_type, aggregation, n_seeds):
    """Print the trajectory-outcome analysis (was: boundary-intuition test)."""
    tag = f"margin={margin_type}, aggregation={aggregation}"
    print(f"\n=== (d) Trajectory-outcome analysis at end of task-1 training [{tag}] ===")
    print(f"(averaged over {n_seeds} seeds; per-class thresholds; NOT a geometric boundary test)\n")
    header = (f"{'Rule':<22} {'Correct@end':>14} {'MedPctlGlob':>13} {'MeanPctlGlob':>13} "
              f"{'MedPctlCls':>12} {'MeanPctlCls':>12} "
              f"{'WellClsEnd':>13} {'LowMargEnd':>13} {'MisclsEnd':>13}")
    print(header); print('-' * len(header))
    for name, d in agg.items():
        f = lambda k, dg=3: f"{d[k+'_mean']:>6.{dg}f}±{d[k+'_std']:.{dg}f}"
        print(f"{name:<22} {f('frac_correct_end'):>14} "
              f"{f('median_pctl_global',1):>13} {f('mean_pctl_global',1):>13} "
              f"{f('median_pctl_class',1):>12} {f('mean_pctl_class',1):>12} "
              f"{f('well_classified_end'):>13} {f('low_margin_correct_end'):>13} "
              f"{f('misclassified_end'):>13}")


def print_overlap(ovl_dict):
    """Print the overlap-with-direct-boundary results (multi-criterion,
    both at-epoch-E and over-first-E-window)."""
    print(f"\n=== Overlap of CGR selection with per-class bottom-K under alternative criteria ===")
    print(f"(Chance overlap under Random selection ≈ 20% for K/N_class = 0.2.)")
    print(f"\n{'Criterion':<24} {'in epoch-E pass':>18} {'over first-E window':>22}")
    print('-' * 66)
    criteria = ['d_pred', 'm_pred', 'nearest_pair_gap', 'abs_m_tgt', 'signed_m_tgt']
    for c in criteria:
        at_E = ovl_dict.get(f'{c}_at_E', (float('nan'), float('nan')))
        ov_E = ovl_dict.get(f'{c}_over_E', (float('nan'), float('nan')))
        print(f"{c:<24} {at_E[0]:>10.3f}±{at_E[1]:.3f}    {ov_E[0]:>13.3f}±{ov_E[1]:.3f}")
    print("(signed_m_tgt shows the ordering pathology GPT flagged: 'smallest' picks "
          "confidently-wrong samples, not near-boundary. Included for comparison.)")


def print_trajectory(traj):
    """Print the CGR-selected early-vs-late margin trajectory (both prob and logit)."""
    print(f"\n=== (d-descriptive) CGR-selected samples: margin trajectory ===")
    for label, key in [('probability margin', 'prob'), ('logit margin', 'logit')]:
        r = traj[key]
        print(f"  {label:<20}: early (first E) = {r['early_mean']:.3f} ± {r['early_std']:.3f}    "
              f"late (last {r['late_window']}) = {r['late_mean']:.3f} ± {r['late_std']:.3f}")
    print("  (Descriptive training dynamics — NOT a geometric boundary claim.)")


def print_d2(agg, ovl, n_seeds):
    print(f"\n=== (d2) Selection-time boundary diagnostics [Concern 2, extended] ===")
    print(f"(measured during each sample's diagnostic pass in epoch E; NOT a common")
    print(f" end-of-epoch checkpoint. Averaged over {n_seeds} seeds.)\n")

    print("--- POINT-IN-TIME snapshots during each sample's diagnostic pass in epoch E (primary) ---")
    header = (f"{'Rule':<22} {'SignChgSt 1..E':>15} {'Correct@E':>12} "
              f"{'ProbMSgn':>11} {'ProbMPrd':>11} {'LogitMSgn':>11} "
              f"{'|LogitM|':>11} {'LogitMPrd':>11}")
    print(header); print('-' * len(header))
    for name, d in agg.items():
        def f(k, dg=3):
            mu, sd = d[k]
            return f"{mu:>5.{dg}f}±{sd:.{dg}f}"
        print(f"{name:<22} {f('sign_change_strict'):>15} {f('frac_correct_at_E'):>12} "
              f"{f('prob_margin_signed'):>11} {f('prob_margin_pred'):>11} "
              f"{f('logit_margin_signed',2):>11} {f('abs_logit_margin',2):>11} "
              f"{f('logit_margin_pred',2):>11}")

    print()
    header2 = (f"{'Rule':<22} {'NearPairGap':>13} {'NearPairPrd':>13} "
               f"{'d_tgt_signed':>13} {'d_tgt_absmin':>13} {'d_pred':>13} "
               f"{'d_on_correct':>13}")
    print(header2); print('-' * len(header2))
    for name, d in agg.items():
        def f(k, dg=3):
            mu, sd = d[k]
            return f"{mu:>5.{dg}f}±{sd:.{dg}f}"
        print(f"{name:<22} {f('nearest_pair_gap',2):>13} {f('nearest_pair_gap_pred',2):>13} "
              f"{f('d_target_signed',2):>13} {f('d_target_absmin',2):>13} "
              f"{f('d_pred',2):>13} {f('d_on_correct_subset',2):>13}")

    print()
    header3 = (f"{'Rule':<22} {'bot20% |m|':>13} {'bot20% d_pred':>15}")
    print(header3); print('-' * len(header3))
    for name, d in agg.items():
        def f(k, dg=3):
            mu, sd = d[k]
            return f"{mu:>5.{dg}f}±{sd:.{dg}f}"
        print(f"{name:<22} {f('frac_low_absm_class'):>13} {f('frac_low_dpred_class'):>15}")

    print(f"\n--- WINDOW aggregates over first E epochs (complementary) ---")
    print(f"(Per-sample mean and range across epochs 0..E-1; then averaged across selection.)")
    keys_pairs = [
        ('prob_margin_signed', 'ProbMSgn'),
        ('logit_margin_signed', 'LogitMSgn'),
        ('abs_logit_margin', '|LogitM|'),
        ('logit_margin_pred', 'LogitMPrd'),
        ('nearest_pair_gap', 'NearPairGap'),
        ('d_pred', 'd_pred'),
    ]
    print(f"\n{'Rule':<22} " + " ".join(f"{lbl+' mean':>13}" for _, lbl in keys_pairs))
    print('-' * (23 + 14*len(keys_pairs)))
    for name, d in agg.items():
        def f(k, dg=3):
            mu, sd = d[k]
            return f"{mu:>5.{dg}f}±{sd:.{dg}f}"
        row = f"{name:<22} " + " ".join(f"{f(kk+'_mean',2):>13}" for kk, _ in keys_pairs)
        print(row)
    print(f"\n{'Rule':<22} " + " ".join(f"{lbl+' rng':>13}" for _, lbl in keys_pairs))
    print('-' * (23 + 14*len(keys_pairs)))
    for name, d in agg.items():
        def f(k, dg=3):
            mu, sd = d[k]
            return f"{mu:>5.{dg}f}±{sd:.{dg}f}"
        row = f"{name:<22} " + " ".join(f"{f(kk+'_range',2):>13}" for kk, _ in keys_pairs)
        print(row)

    print(f"\n--- Overlaps of CGR selection with per-class bottom-K during epoch-E diagnostic pass ---")
    for label, key in [
        ('by |logit margin|         ', 'CGR_vs_low_absm_at_E'),
        ('by d_pred (feat-space)    ', 'CGR_vs_low_dpred_at_E'),
        ('by nearest_pair_logit_gap ', 'CGR_vs_low_nearest_pair_gap_at_E'),
        ('by logit_margin_pred      ', 'CGR_vs_low_m_pred_at_E'),
    ]:
        mu, sd = ovl[key]
        print(f"  {label}: {mu:.3f} ± {sd:.3f}")


def print_c(results, mean_rho, std_rho, E_small, E_large):
    print(f"\n=== (c) Within-seed Spearman: σ² at E={E_small} vs σ² at E={E_large} "
          f"[Concern 4 anchor] ===")
    print(f"Per-seed ρ values:")
    for r in results:
        sig = '***' if r['p'] < 1e-50 else ('**' if r['p'] < 1e-10 else '')
        print(f"  seed {r['seed']}: ρ = {r['rho']:.4f}  (p = {r['p']:.3e})  {sig}")
    rhos = [r['rho'] for r in results]
    print(f"\nMean ρ ± std over {len(results)} seeds: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"Range: [{min(rhos):.4f}, {max(rhos):.4f}]")
    print(f"\n  Paper insertion: "
          f"\\bar\\rho_{{E={E_small},E={E_large}}} = {mean_rho:.3f} \\pm {std_rho:.3f}")


def print_b2(agg, per_seed_rows, k_per_class, num_classes, n_seeds):
    print(f"\n=== (b.2) Diagnostic table (averaged over {n_seeds} seeds) ===")
    print(f"Per-class budget K = {k_per_class}  ({num_classes} classes seen in task 1)\n")

    rule_names = list(agg.keys())
    header = f"{'Rule':<18} {'Margin (mean±std)':>22} {'Forget (mean±std)':>22} {'MeanConf (mean±std)':>22}"
    print(header)
    print('-' * len(header))
    for name in rule_names:
        d = agg[name]
        print(f"{name:<18} "
              f"{d['mean_margin_mean']:>7.4f} ± {d['mean_margin_std']:.4f}    "
              f"{d['mean_forgetting_mean']:>7.3f} ± {d['mean_forgetting_std']:.3f}    "
              f"{d['mean_target_conf_mean']:>7.4f} ± {d['mean_target_conf_std']:.4f}")

    print("\nPer-seed breakdown:")
    for name in rule_names:
        print(f"  {name}:")
        d = agg[name]
        for metric_pretty, metric_key in [('margin', 'mean_margin_per_seed'),
                                          ('forget', 'mean_forgetting_per_seed'),
                                          ('conf',   'mean_target_conf_per_seed')]:
            vals = d[metric_key]
            print(f"    {metric_pretty}: {[f'{v:.4f}' for v in vals]}")

    # LaTeX table
    print("\n--- LaTeX (paste into Table tab:diagnostic) ---")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Selection rule & Mean margin $\downarrow$ & Forgetting events $\uparrow$ & Mean target conf. \\")
    print(r"\midrule")
    for name in rule_names:
        d = agg[name]
        print(f"{name} & "
              f"${d['mean_margin_mean']:.3f} \\pm {d['mean_margin_std']:.3f}$ & "
              f"${d['mean_forgetting_mean']:.2f} \\pm {d['mean_forgetting_std']:.2f}$ & "
              f"${d['mean_target_conf_mean']:.3f} \\pm {d['mean_target_conf_std']:.3f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_dir', type=str, required=True,
                        help='Directory containing cgr_diag_seed*.pt files.')
    parser.add_argument('--E', type=int, default=4,
                        help='CGR variance window (should match what was used in training).')
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='Buffer size used in the run; controls per-class top-K.')
    parser.add_argument('--E_small', type=int, default=2,
                        help='Small window for the within-seed E_small-vs-E_large check (Concern 4). '
                             'Default 2.')
    parser.add_argument('--E_large', type=int, default=5,
                        help='Large window for the within-seed E_small-vs-E_large check (Concern 4). '
                             'Default 5.')
    args = parser.parse_args()

    logs = load_seed_logs(args.diag_dir)
    print(f"Loaded {len(logs)} seed logs from {args.diag_dir}")
    for log in logs:
        print(f"  seed={log['seed']}  E={log['E']}  n_epochs={log['n_epochs']}  "
              f"n_samples={log['n_sample_per_task']}  buffer_size={log['buffer_size']}")

    # (b.1) cross-seed
    if len(logs) >= 2:
        mean_rho, std_rho, all_rhos, pairs = cross_seed_spearman(logs, args.E)
        print_b1(mean_rho, std_rho, all_rhos, pairs, len(logs))
    else:
        print("\n(b.1) Cross-seed correlation skipped: need >= 2 seeds.")

    # (c) within-seed E_small vs E_large -- anchors Concern 4 sanity check
    c_results, c_mean, c_std = within_seed_E_small_vs_E_large_spearman(
        logs, args.E_small, args.E_large
    )
    print_c(c_results, c_mean, c_std, args.E_small, args.E_large)

    # (b.2) diagnostic table -- averaged across all seeds
    agg, per_seed_rows, k_per_class, num_classes = diagnostic_table_all_seeds(
        logs, args.E, args.buffer_size
    )
    print_b2(agg, per_seed_rows, k_per_class, num_classes, len(logs))

    # (d) trajectory-outcome analysis -- Concern 2 (was boundary-intuition test).
    # Report all four combinations: {prob, logit} margin × {final, last_E} aggregation.
    for margin_type in ('prob', 'logit'):
        for aggregation in ('final', 'last_E'):
            agg_d, _ = boundary_intuition_test(
                logs, args.E, args.buffer_size,
                margin_type=margin_type, aggregation=aggregation,
            )
            print_d(agg_d, margin_type, aggregation, len(logs))
    # Overlap-with-direct-boundary: multi-criterion (Concern 2, extended per GPT Point 6)
    try:
        ovl_dict = overlap_with_direct_boundary(logs, args.E, args.buffer_size)
        print_overlap(ovl_dict)
    except KeyError as e:
        print(f"\n[Skipping overlap analysis] {e}")
    # CGR-selected sample margin trajectory (early vs late; both prob and logit)
    traj = cgr_margin_trajectory(logs, args.E, args.buffer_size)
    print_trajectory(traj)

    # (d2) selection-time boundary diagnostics -- Concern 2, extended
    # (requires extended log format with diag_logit_margin and diag_feat_dist_pred)
    try:
        agg_d2, ovl_d2 = selection_time_diagnostics(logs, args.E, args.buffer_size)
        print_d2(agg_d2, ovl_d2, len(logs))
    except KeyError as e:
        print(f"\n[Skipping (d2)] {e}")


if __name__ == '__main__':
    main()
