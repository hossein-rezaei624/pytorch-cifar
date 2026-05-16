# CGR diagnostic experiments — usage

These two files produce the numbers/table for the Reviewer-1 Concern-1 rebuttal:

- **(b.1) Cross-seed Spearman ρ** of per-sample variance vectors → §5.3 sub-point (b.1) and the rebuttal as `\bar\rho = X.XX ± Y.YY`.
- **(b.2) Diagnostic table** comparing CGR vs Random / High-loss / High-confidence / Low-confidence → Table `tab:diagnostic` (mean ± std over 5 seeds).

Note: an earlier draft included a Spearman correlation between confidence variance and forgetting events (a.2). On real CGR runs this correlation came out *negative* (≈ −0.30) because early-high-variance samples are unstable early but become stable later, while Toneva's forgetting events count instability over the full trajectory. Rather than try to explain a negative correlation, we drop the (a.2) analysis from the paper and let the forgetting-events column in the (b.2) diagnostic table do the work — there, CGR's *low* forgetting count compared to Random / High-loss / Low-confidence is itself the positive finding. The (a.2) Spearman is no longer computed by `analyze_cgr_diag.py`.

## Files

- `cgr_with_diag.py` — drop-in replacement for `models/cgr.py`. Adds a `--cgr_diag_log` flag; when set, dumps per-sample diagnostics from task 1 to disk. When the flag is off, behaviour is identical to your original.
- `analyze_cgr_diag.py` — standalone analysis script. Loads the per-seed `.pt` files and prints all three results (mean ± std for a.2 and b.2, all 10 seed pairs for b.1), plus a LaTeX-ready diagnostic table.

## How the logging works

With `--cgr_diag_log`, during **task 1 only**:

1. The same eval-mode forward pass that CGR already does on `not_aug_inputs` (lines 339–353 of your original `cgr.py`) is run for **every epoch of task 1**, not just the first $E$. CGR's selection still only uses the first $E$ rows of `self.confidence_by_sample`, so its behaviour is unchanged.
2. From that eval-mode pass, per-sample target confidence, margin, correctness, and cross-entropy loss are recorded at every epoch.
3. After the last epoch of task 1, everything is saved to `cgr_diag_logs/cgr_diag_seed<S>.pt`.

For task 2 onward the diagnostic code is a no-op; CGR continues training normally. All diagnostics come from CGR's eval-mode forward pass on `not_aug_inputs` — not from the train-mode SGD pass.

## Running the experiments

### 1. Install the patched `cgr.py`

```bash
cp models/cgr.py models/cgr.py.bak
cp cgr_with_diag.py models/cgr.py
```

### 2. Run task-1 diagnostics across 5 seeds

For Split CIFAR-100 with buffer 1000 (matches your main-experiments cell):

```bash
for SEED in 0 1 2 3 4; do
  python utils/main.py \
      --model cgr \
      --dataset seq-cifar100 \
      --buffer_size 1000 \
      --load_best_args \
      --seed $SEED \
      --cgr_diag_log \
      --cgr_diag_dir cgr_diag_logs
done
```

Each run saves `cgr_diag_seed<S>.pt` to `cgr_diag_logs/`.

**You only need the first task's data**, so you can interrupt each run with Ctrl-C as soon as it prints `[CGR-Diag] Saved task-1 diagnostics to ...` (that line appears at the end of task 1's last epoch, before task 2 starts). That saves substantial GPU time.

### 3. Analyze

```bash
python analyze_cgr_diag.py \
    --diag_dir cgr_diag_logs \
    --E 4 \
    --buffer_size 1000
```

Match `--E` to the value of `E` you used in the run for CIFAR-100 at buffer 1000 (per your paper's Table 1: `E = 4` for that cell).

Sample output (your numbers will differ; these are illustrative):

```
Loaded 5 seed logs from cgr_diag_logs
  seed=0  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=1  ...

=== (b.1) Cross-seed Spearman correlation of variance vectors ===
Number of seeds: 5  (10 pairs)
Mean ρ ± std: 0.7142 ± 0.0421
Per-pair ρ values:
  (seed 0, seed 1): ρ = 0.7203
  ...
  Paper insertion: \bar\rho = 0.71 \pm 0.04

=== (b.2) Diagnostic table (averaged over 5 seeds) ===
Per-class budget K = 100  (10 classes seen in task 1)

Rule                    Margin (mean±std)   Forget (mean±std)   MeanConf (mean±std)
Random              -0.0168 ± 0.0095     3.99 ± 0.20      0.3015 ± 0.0056
High loss           -0.3543 ± 0.0097     4.93 ± 0.22      0.0972 ± 0.0093
High confidence      0.3685 ± 0.0188     2.54 ± 0.27      0.5622 ± 0.0127
Low confidence      -0.3673 ± 0.0081     4.97 ± 0.20      0.0719 ± 0.0033
CGR (variance)       0.1840 ± 0.0172     3.07 ± 0.24      0.4433 ± 0.0082

--- LaTeX (paste into Table tab:diagnostic) ---
\begin{tabular}{lccc}
...
\end{tabular}
```

## What gets reported in each row

For each selection rule, three columns:

- **Mean margin** — averaged over the **first $E$ epochs** of task-1 training. Matches CGR's selection window and Figure 3 sub-figure (b.1). End-of-training margin would be uninformative because the model has converged and margins are large for nearly all samples.
- **Mean forgetting events** — over **all training epochs** (Toneva et al.'s definition; needs the full trajectory).
- **Mean target confidence** — over the **first $E$ epochs** (matches Figure 3 b.1 and CGR's selection window).

The mean ± std is computed across the 5 training seeds.

## Sanity checks

After the first seed runs, verify:

```python
import torch
d = torch.load('cgr_diag_logs/cgr_diag_seed0.pt')
print({k: (v.shape, v.dtype) if torch.is_tensor(v) else v for k, v in d.items()})
print('cgr_conf range:', d['cgr_confidence_by_sample'].min().item(),
      d['cgr_confidence_by_sample'].max().item())
print('margin range:', d['diag_margin'].min().item(),
      d['diag_margin'].max().item())
print('epoch-0 acc:', d['diag_correct'][0].float().mean().item())
print('last-epoch acc:', d['diag_correct'][-1].float().mean().item())
```

Expected:
- `cgr_confidence_by_sample` and `diag_target_conf` in `[0, 1]`.
- `diag_margin` in `[-1, 1]`.
- epoch-0 correctness ≈ 0.10 (chance for 10-class task 1 on CIFAR-100).
- last-epoch correctness > 0.9 (model has converged on task 1).
- `n_sample_per_task = 5000` for Split CIFAR-100 (10 classes × 500 samples).
- `cgr_confidence_by_sample` has all 50 rows filled (not just the first 4) when `--cgr_diag_log` is set.

If anything looks off, paste the sanity-check output and the first few rows of `cgr_confidence_by_sample[:5, :10]` and I'll debug.

## Restoring the original `cgr.py`

```bash
cp models/cgr.py.bak models/cgr.py
```

## Caveats

- All diagnostics come from CGR's existing eval-mode forward pass on `not_aug_inputs`, so variance, margin, correctness, and loss are all on the same footing. There is **no inconsistency** between the variance signal and the other diagnostics — all are computed from the same forward-pass logits.
- The extra eval-mode forward passes during epochs $E+1$ through $50$ are the only extra compute introduced. For Split CIFAR-100 task 1 with batch 32, that's roughly 7K extra forward passes per seed (~70 seconds on RTX 8000), so 5 seeds add maybe 6 minutes total — negligible.
- The "Random" selection rule in (b.2) is reproducibly seeded from the log's own seed value, so each run's "Random" comparison uses different random samples (which is what you want — averaging the same random selection 5 times would give zero variance and be misleading).
- For (a.2) the mean ± std is over per-seed Spearman ρ values, not over per-sample correlations. Each seed gives one ρ, and we report mean and std of those 5 ρ values.
