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

Usage:
    python analyze_cgr_diag.py --diag_dir cgr_diag_logs --E 4 --buffer_size 1000
    # optional: --E_small 2 --E_large 5   (defaults shown)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


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
    return float(np.mean(rhos)), float(np.std(rhos)), rhos, pairs


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
    return results, float(np.mean(rhos)), float(np.std(rhos))


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
            agg[name][metric + '_std']  = float(np.std(vals))
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


def boundary_intuition_test(logs, E, buffer_size, final_epoch_idx=None,
                             margin_pctl_threshold=20.0):
    """For each of five selection rules, characterize the end-of-training margin
    and correctness distribution of the selected samples. Tests whether CGR-
    selected samples remain near the decision boundary at end of training. They
    mostly do not — they migrate from ambiguous (low margin) to well-classified
    (high margin), matching Swayamdipta et al. (2020) ambiguous-then-learned."""
    per_seed_records = []
    for log in logs:
        conf   = log['cgr_confidence_by_sample'].numpy()
        margin = log['diag_margin'].numpy()
        correct = log['diag_correct'].numpy().astype(bool)
        loss   = log['diag_loss'].numpy()
        labels = log['diag_labels'].numpy()
        seed   = int(log['seed']) if str(log['seed']).isdigit() else 0

        n_epochs = conf.shape[0]
        _final = (n_epochs - 1) if final_epoch_idx is None else final_epoch_idx
        final_margin = margin[_final]
        final_correct = correct[_final]

        n_classes = len(np.unique(labels))
        per_class = buffer_size // n_classes
        low_thr = np.percentile(final_margin, margin_pctl_threshold)
        margin_ranks = np.argsort(np.argsort(final_margin))

        rng = np.random.default_rng(seed)
        rules = {}
        sigma2 = np.var(conf[:E], axis=0)
        rules['CGR (high variance)'] = _top_k_per_class(sigma2, labels, per_class, descending=True)
        rules['Random'] = np.concatenate([
            rng.choice(np.where(labels == c)[0],
                       size=min(per_class, (labels == c).sum()),
                       replace=False)
            for c in np.unique(labels)
        ])
        mean_conf_E = conf[:E].mean(axis=0)
        mean_loss_E = loss[:E].mean(axis=0)
        rules['High confidence'] = _top_k_per_class(mean_conf_E, labels, per_class, descending=True)
        rules['Low confidence']  = _top_k_per_class(mean_conf_E, labels, per_class, descending=False)
        rules['High loss']       = _top_k_per_class(mean_loss_E, labels, per_class, descending=True)

        seed_row = {}
        for name, idx in rules.items():
            sfm = final_margin[idx]
            sfc = final_correct[idx]
            seed_row[name] = {
                'frac_correct':   float(sfc.mean()),
                'median_pctl':    float(np.median(margin_ranks[idx] / len(labels) * 100)),
                'resolved':       float((sfc & (sfm >= low_thr)).mean()),
                'still_boundary': float((sfc & (sfm <  low_thr)).mean()),
                'outlier':        float((~sfc & (sfm < low_thr)).mean()),
            }
        per_seed_records.append(seed_row)

    agg = {}
    for name in per_seed_records[0]:
        agg[name] = {}
        for metric in ['frac_correct','median_pctl','resolved','still_boundary','outlier']:
            vals = [r[name][metric] for r in per_seed_records]
            agg[name][f'{metric}_mean'] = float(np.mean(vals))
            agg[name][f'{metric}_std']  = float(np.std(vals, ddof=1))
    return agg, per_seed_records


def overlap_with_direct_boundary(logs, E, buffer_size):
    """Fraction of CGR-selected samples that are also in a 'lowest |margin|'
    per-class top-K selection. Small overlap means CGR is selecting
    variability-based (ambiguous) samples rather than persistently-near-
    boundary samples."""
    overlaps = []
    for log in logs:
        conf = log['cgr_confidence_by_sample'].numpy()
        margin = log['diag_margin'].numpy()
        labels = log['diag_labels'].numpy()
        n_classes = len(np.unique(labels))
        per_class = buffer_size // n_classes

        sigma2 = np.var(conf[:E], axis=0)
        cgr_sel = set(_top_k_per_class(sigma2, labels, per_class, descending=True))
        abs_marg = np.abs(margin[:E]).mean(axis=0)
        bnd_sel = set(_top_k_per_class(abs_marg, labels, per_class, descending=False))
        overlaps.append(len(cgr_sel & bnd_sel) / len(cgr_sel))
    return float(np.mean(overlaps)), float(np.std(overlaps, ddof=1))


def cgr_margin_trajectory(logs, E, buffer_size, late_window=5):
    """For CGR-selected samples, compute mean margin early (first E epochs) vs
    late (last `late_window` epochs). Shows the ambiguous-then-learned
    trajectory within a single task."""
    early_all, late_all = [], []
    for log in logs:
        conf = log['cgr_confidence_by_sample'].numpy()
        margin = log['diag_margin'].numpy()
        labels = log['diag_labels'].numpy()
        n_epochs = margin.shape[0]
        n_classes = len(np.unique(labels))
        per_class = buffer_size // n_classes
        sigma2 = np.var(conf[:E], axis=0)
        selected = _top_k_per_class(sigma2, labels, per_class, descending=True)
        early_all.append(float(margin[:E, selected].mean()))
        late_all.append(float(margin[n_epochs - late_window : n_epochs, selected].mean()))
    return (float(np.mean(early_all)), float(np.std(early_all, ddof=1)),
            float(np.mean(late_all)),  float(np.std(late_all, ddof=1)))


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


def print_d(agg, ovl_mean, ovl_std, early_mu, early_sd, late_mu, late_sd, n_seeds):
    print(f"\n=== (d) Boundary-intuition verification [Concern 2] ===")
    print(f"(end-of-training characterization, averaged over {n_seeds} seeds)\n")
    header = f"{'Rule':<22} {'Correct':>14} {'MargPctl':>12} {'Resolved':>14} {'Boundary':>14} {'Outlier':>14}"
    print(header); print('-' * len(header))
    for name, d in agg.items():
        print(f"{name:<22} "
              f"{d['frac_correct_mean']:>6.3f}±{d['frac_correct_std']:.3f}    "
              f"{d['median_pctl_mean']:>5.1f}±{d['median_pctl_std']:.1f}    "
              f"{d['resolved_mean']:>6.3f}±{d['resolved_std']:.3f}    "
              f"{d['still_boundary_mean']:>6.3f}±{d['still_boundary_std']:.3f}    "
              f"{d['outlier_mean']:>6.3f}±{d['outlier_std']:.3f}")
    print(f"\nOverlap between CGR selection and direct low-|margin| selection: "
          f"{ovl_mean:.3f} ± {ovl_std:.3f}")
    print(f"CGR-selected samples' mean margin: early (first E) = {early_mu:.3f} ± {early_sd:.3f}, "
          f"late (last 5) = {late_mu:.3f} ± {late_sd:.3f}")
    print(f"\n  Paper insertion (for Table 17 caption and §4 paragraph):")
    print(f"    Overlap value: {ovl_mean:.3f}")
    print(f"    Early margin:  {early_mu:.3f} \\pm {early_sd:.3f}")
    print(f"    Late margin:   {late_mu:.3f} \\pm {late_sd:.3f}")


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

    # (d) boundary-intuition verification -- Concern 2
    agg_d, _ = boundary_intuition_test(logs, args.E, args.buffer_size)
    ovl_mean, ovl_std = overlap_with_direct_boundary(logs, args.E, args.buffer_size)
    early_mu, early_sd, late_mu, late_sd = cgr_margin_trajectory(logs, args.E, args.buffer_size)
    print_d(agg_d, ovl_mean, ovl_std, early_mu, early_sd, late_mu, late_sd, len(logs))


if __name__ == '__main__':
    main()
