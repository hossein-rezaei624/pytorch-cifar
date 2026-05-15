# CGR diagnostic experiments — usage

This produces the three numbers/tables for the Reviewer-1 Concern-1 rebuttal:

- **(b.1) Cross-seed Spearman ρ** of per-sample variance vectors → goes into §5.3 sub-point (b.1) and the rebuttal as `\bar\rho = X.XX ± Y.YY`.
- **(a.2) Spearman ρ between variance and forgetting events** → goes into §5.3 sub-point (a.2) as `ρ = Z.ZZ`.
- **(b.2) Diagnostic table** (4 rows × 3 metrics) comparing CGR vs Random / High-loss / Low-confidence → goes into Table `tab:diagnostic`.

## Files

- `cgr_with_diag.py` — drop-in replacement for `models/cgr.py` in your Mammoth fork. Adds a `--cgr_diag_log` flag; when set, dumps per-sample diagnostics for task 1 to disk. When the flag is off, behavior is identical to your original.
- `analyze_cgr_diag.py` — standalone analysis script. Loads the dumped logs and prints all three results (plus a LaTeX-ready diagnostic table).

## How it works (mechanics)

`cgr_with_diag.py` adds two things during **task 1 only**, gated by `--cgr_diag_log`:

1. Records per-sample target probability, margin, correctness, and per-sample loss from the **same train-mode SGD forward pass** CGR already does. No extra forward passes; cost is negligible.
2. After the last epoch of task 1, saves all of the above plus CGR's existing eval-mode confidence trajectory (`self.confidence_by_sample`) to `cgr_diag_logs/cgr_diag_seed<S>.pt`.

The variance vector for analyses (b.1) and (a.2) is computed from CGR's actual eval-mode confidence (the same statistic CGR uses for buffer selection), not from train-mode confidence — so the numbers reflect CGR's true selection signal.

For task 2 and beyond the diagnostic code is a no-op; CGR continues training normally.

## Running the experiments

### 1. Install the patched `cgr.py`

```bash
# Back up the original
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
      --cgr_diag_dir cgr_diag_logs/cifar100_buf1000
done
```

Each run saves `cgr_diag_seed<S>.pt` to `cgr_diag_logs/cifar100_buf1000/`.

**You only strictly need the first task's data**, so you can interrupt each run with Ctrl-C once it has printed the `[CGR-Diag] Saved task-1 diagnostics to ...` line if you want to save time. Or let the runs complete normally; the diagnostic dump happens once per run regardless.

If you ran with `--n_epochs` different from 50, pass the same value here; the script reads it from the dataset config / your CLI.

### 3. Analyze

```bash
python analyze_cgr_diag.py \
    --diag_dir cgr_diag_logs/cifar100_buf1000 \
    --E 4 \
    --buffer_size 1000
```

Output (numbers will be your actual ones):

```
Loaded 5 seed logs from cgr_diag_logs/cifar100_buf1000
  seed=0  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=1  ...

=== (b.1) Cross-seed Spearman correlation of variance vectors ===
Number of seeds: 5  (10 pairs)
Mean ρ ± std: 0.72 ± 0.04
Per-pair ρ values: ['0.7142', '0.7401', ...]

  Paper insertion: bar rho = 0.72 pm 0.04

=== (a.2) Variance vs forgetting events (seed 0) ===
Spearman ρ = 0.6541   (p = 1.2e-300)

  Paper insertion: rho = 0.65

=== (b.2) Diagnostic table (seed 0) ===
Per-class budget K = 100  (10 classes seen in task 1)

Rule                #sel     Margin   Forget  MeanConf
-----------------------------------------------------
Random               1000     0.7821    0.42     0.812
High loss            1000     0.1234    5.62     0.198
Low confidence       1000     0.0851    7.13     0.143
CGR (variance)       1000     0.4012    2.87     0.521

--- LaTeX (paste into Table tab:diagnostic) ---
\begin{tabular}{lccc}
\toprule
Selection rule & Mean margin $\downarrow$ & Forgetting events $\uparrow$ & Mean target conf. \\
\midrule
Random & 0.782 & 0.42 & 0.812 \\
High loss & 0.123 & 5.62 & 0.198 \\
Low confidence & 0.085 & 7.13 & 0.143 \\
CGR (variance) & 0.401 & 2.87 & 0.521 \\
\bottomrule
\end{tabular}
```

The CGR row should sit between Random (high margin, low forgetting) and Low confidence (very low margin, high forgetting) — the "ambiguous middle" framing in the paper. **If the numbers don't show that pattern, rewrite the (b.2) prose to match what they do show.** Don't paste the placeholder narrative without checking the actual numbers.

### 4. Restore the original `cgr.py` if you want

```bash
cp models/cgr.py.bak models/cgr.py
```

## Quick sanity checks before trusting the output

- The 5 per-seed `.pt` files should each be the same size (within a few KB). Wildly different sizes = something is off.
- `n_sample_per_task` should equal `5000` for Split CIFAR-100 (10 classes × 500 samples).
- `cgr_confidence_by_sample[:E]` should have entries in `[0, 1]` (it's a softmax probability).
- `diag_correct[0, :].float().mean()` (task-1 epoch-0 accuracy) should be roughly chance (~10% for 10-class CIL on task 1); `diag_correct[-1, :].float().mean()` should be high (>0.9 typically after 50 epochs of training).
- `diag_target_conf` should have values in `[0, 1]`; `diag_margin` in `[-1, 1]`.

If anything looks off, send me the output of:

```python
import torch
d = torch.load('cgr_diag_logs/cifar100_buf1000/cgr_diag_seed0.pt')
for k, v in d.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={v.float().min().item():.4f} max={v.float().max().item():.4f}")
    else:
        print(f"{k}: {v}")
```

and I can debug.

## Caveats

- The diagnostic uses train-mode forward-pass metrics for margin, forgetting, and per-sample loss. This matches Toneva et al.'s original forgetting-events definition (which uses train-mode predictions). The variance for (b.1) and (a.2) uses CGR's existing eval-mode confidence — i.e., CGR's actual selection signal. So there is a small train/eval inconsistency between the variance signal and the forgetting signal in (a.2); this is intentional (use each method's own framing) and noted in the analysis script comments.
- For (b.2), I average the margin over the last 5 training epochs by default (`--last_k_for_margin 5`) rather than just the final epoch, for stability. This is a minor cosmetic choice and won't change the qualitative pattern.
- The (a.2) and (b.2) numbers are reported for a single seed by default (the first one). You can pass `--report_seed N` to use a specific seed. If you want averaged numbers, modify the script — but for the rebuttal a single representative seed is typically fine.
