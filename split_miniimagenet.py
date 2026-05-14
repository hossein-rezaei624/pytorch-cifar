"""
Analysis script for CGR diagnostic logs produced by cgr_with_diag.py
(when run with --cgr_diag_log).

Computes:
  (b.1) Cross-seed Spearman rank correlation of per-sample variance vectors
  (a.2) Spearman correlation between confidence variance and forgetting-event count
  (b.2) Diagnostic table comparing selection rules

Usage:
  python analyze_cgr_diag.py --diag_dir cgr_diag_logs \\
      --E 4 --buffer_size 1000

The buffer_size and number of classes per task only affect the per-class budget
used to form the top-K selection sets in (b.2).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


# ----------------------------- I/O -----------------------------

def load_seed_logs(diag_dir):
    """Load every cgr_diag_seed*.pt file from diag_dir, sorted by seed."""
    paths = sorted(Path(diag_dir).glob('cgr_diag_seed*.pt'))
    if not paths:
        raise FileNotFoundError(f"No 'cgr_diag_seed*.pt' files found in {diag_dir}")
    return [torch.load(p, map_location='cpu') for p in paths]


# ------------------------- Metrics ---------------------------

def variance_from_eval_confidence(log, E):
    """CGR's actual variance signal: variance of eval-mode confidence over first E epochs."""
    conf = log['cgr_confidence_by_sample']  # (n_epochs, n_samples)
    return conf[:E].var(dim=0).numpy()


def forgetting_events(correct):
    """Toneva-style forgetting events: # of correct -> incorrect transitions over training."""
    correct = correct.bool()
    # transitions: correct[t] = True AND correct[t+1] = False
    transitions = correct[:-1] & ~correct[1:]
    return transitions.sum(dim=0).numpy()


# --------------------- (b.1) Cross-seed -----------------------

def cross_seed_spearman(logs, E):
    variances = [variance_from_eval_confidence(log, E) for log in logs]

    # Sanity check: all variance vectors should have the same length
    lens = {v.shape[0] for v in variances}
    if len(lens) != 1:
        raise ValueError(f"Variance vectors have inconsistent lengths across seeds: {lens}")

    rhos = []
    n = len(variances)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(variances[i], variances[j])
            rhos.append(r)
    return float(np.mean(rhos)), float(np.std(rhos)), rhos


# ---------------- (a.2) Variance vs forgetting ----------------

def variance_vs_forgetting(log, E):
    variance = variance_from_eval_confidence(log, E)
    forgetting = forgetting_events(log['diag_correct'])
    r, p = spearmanr(variance, forgetting)
    return float(r), float(p)


# ---------------- (b.2) Diagnostic table ----------------------

def diagnostic_table(log, E, buffer_size, last_k_for_margin=5, random_seed=0):
    """
    For each selection rule, pick top-K per class, then report:
      - mean margin (averaged over the last `last_k_for_margin` training epochs for stability)
      - mean forgetting events
      - mean target confidence (averaged over all training epochs)

    Selection rules:
      - Random: K random samples per class
      - High loss: top-K per class by mean loss over first E epochs
      - Low confidence: bottom-K per class by mean target conf over first E epochs
      - CGR (variance): top-K per class by variance over first E epochs (CGR's actual rule)
    """
    n_epochs = log['diag_target_conf'].shape[0]
    labels = log['diag_labels'].numpy()
    n_samples = labels.shape[0]

    # Per-sample selection scores (computed over first E epochs to match CGR's window)
    variance = variance_from_eval_confidence(log, E)
    mean_conf_E = log['diag_target_conf'][:E].mean(dim=0).numpy()
    mean_loss_E = log['diag_loss'][:E].mean(dim=0).numpy()

    # Per-sample reporting metrics
    margin_late = log['diag_margin'][-last_k_for_margin:].mean(dim=0).numpy()  # margin at end of training
    forgetting = forgetting_events(log['diag_correct'])
    mean_conf_all = log['diag_target_conf'].mean(dim=0).numpy()

    # Determine per-class budget K
    unique_classes = np.unique(labels[labels >= 0])
    num_classes = len(unique_classes)
    k_per_class = buffer_size // num_classes  # CGR's per-class budget after task 1

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
        'Random':         np.concatenate([
                              rng.choice(np.where(labels == c)[0],
                                         size=min(k_per_class, (labels == c).sum()),
                                         replace=False)
                              for c in unique_classes
                          ]),
        'High loss':      top_k_per_class(mean_loss_E, descending=True),
        'Low confidence': top_k_per_class(mean_conf_E, descending=False),
        'CGR (variance)': top_k_per_class(variance, descending=True),
    }

    rows = []
    for name, idx in rules.items():
        rows.append({
            'rule': name,
            'n_selected': len(idx),
            'mean_margin': float(margin_late[idx].mean()),
            'mean_forgetting': float(forgetting[idx].mean()),
            'mean_target_conf': float(mean_conf_all[idx].mean()),
        })
    return rows, k_per_class, num_classes


# ------------------------- Reporting -------------------------

def print_b1(mean_rho, std_rho, all_rhos, n_seeds):
    n_pairs = len(all_rhos)
    print(f"\n=== (b.1) Cross-seed Spearman correlation of variance vectors ===")
    print(f"Number of seeds: {n_seeds}  ({n_pairs} pairs)")
    print(f"Mean ρ ± std: {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"Per-pair ρ values: {[f'{r:.4f}' for r in all_rhos]}")
    print(f"\n  Paper insertion: bar rho = {mean_rho:.2f} pm {std_rho:.2f}")


def print_a2(rho, p, seed):
    print(f"\n=== (a.2) Variance vs forgetting events (seed {seed}) ===")
    print(f"Spearman ρ = {rho:.4f}   (p = {p:.3e})")
    print(f"\n  Paper insertion: rho = {rho:.2f}")


def print_b2(rows, k_per_class, num_classes, seed):
    print(f"\n=== (b.2) Diagnostic table (seed {seed}) ===")
    print(f"Per-class budget K = {k_per_class}  ({num_classes} classes seen in task 1)\n")
    header = f"{'Rule':<18} {'#sel':>5} {'Margin':>10} {'Forget':>8} {'MeanConf':>9}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['rule']:<18} {r['n_selected']:>5d} "
              f"{r['mean_margin']:>10.4f} {r['mean_forgetting']:>8.3f} "
              f"{r['mean_target_conf']:>9.4f}")
    # LaTeX
    print("\n--- LaTeX (paste into Table tab:diagnostic) ---")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Selection rule & Mean margin $\downarrow$ & Forgetting events $\uparrow$ & Mean target conf. \\")
    print(r"\midrule")
    for r in rows:
        print(f"{r['rule']} & {r['mean_margin']:.3f} & {r['mean_forgetting']:.2f} & {r['mean_target_conf']:.3f} \\\\")
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
    parser.add_argument('--last_k_for_margin', type=int, default=5,
                        help='Average margin over the last K training epochs for the report column.')
    parser.add_argument('--report_seed', type=int, default=None,
                        help='Which seed to use for (a.2) and (b.2). Defaults to the first available.')
    args = parser.parse_args()

    logs = load_seed_logs(args.diag_dir)
    print(f"Loaded {len(logs)} seed logs from {args.diag_dir}")
    for log in logs:
        print(f"  seed={log['seed']}  E={log['E']}  n_epochs={log['n_epochs']}  "
              f"n_samples={log['n_sample_per_task']}  buffer_size={log['buffer_size']}")

    # Pick the seed to use for (a.2) and (b.2)
    if args.report_seed is None:
        report_log = logs[0]
    else:
        matching = [l for l in logs if l['seed'] == args.report_seed]
        if not matching:
            raise ValueError(f"No log for seed={args.report_seed}; available: {[l['seed'] for l in logs]}")
        report_log = matching[0]
    report_seed = report_log['seed']

    # (b.1) cross-seed
    if len(logs) >= 2:
        mean_rho, std_rho, all_rhos = cross_seed_spearman(logs, args.E)
        print_b1(mean_rho, std_rho, all_rhos, len(logs))
    else:
        print("\n(b.1) Cross-seed correlation skipped: need >= 2 seeds.")

    # (a.2) variance vs forgetting
    rho, p = variance_vs_forgetting(report_log, args.E)
    print_a2(rho, p, report_seed)

    # (b.2) diagnostic table
    rows, k_per_class, num_classes = diagnostic_table(
        report_log, args.E, args.buffer_size, last_k_for_margin=args.last_k_for_margin
    )
    print_b2(rows, k_per_class, num_classes, report_seed)


if __name__ == '__main__':
    main()
