Below are the final results:


```
Loaded 5 seed logs from cgr_diag_logs
  seed=0  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=1  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=2  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=3  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=4  E=4  n_epochs=50  n_samples=5000  buffer_size=1000

=== (b.1) Cross-seed Spearman correlation of variance vectors ===
Number of seeds: 5  (10 pairs)
Mean ρ ± std: 0.4887 ± 0.0134
Per-pair ρ values:
  (seed 0, seed 1): ρ = 0.5023
  (seed 0, seed 2): ρ = 0.4941
  (seed 0, seed 3): ρ = 0.4898
  (seed 0, seed 4): ρ = 0.4845
  (seed 1, seed 2): ρ = 0.4708
  (seed 1, seed 3): ρ = 0.5049
  (seed 1, seed 4): ρ = 0.5110
  (seed 2, seed 3): ρ = 0.4786
  (seed 2, seed 4): ρ = 0.4724
  (seed 3, seed 4): ρ = 0.4785

  Paper insertion: \bar\rho = 0.49 \pm 0.01

=== (b.2) Diagnostic table (averaged over 5 seeds) ===
Per-class budget K = 100  (10 classes seen in task 1)

Rule                    Margin (mean±std)      Forget (mean±std)    MeanConf (mean±std)
---------------------------------------------------------------------------------------
Random             -0.0168 ± 0.0095      3.989 ± 0.198     0.3015 ± 0.0056
High loss          -0.3543 ± 0.0097      4.929 ± 0.224     0.0972 ± 0.0093
High confidence     0.3685 ± 0.0188      2.544 ± 0.274     0.5622 ± 0.0127
Low confidence     -0.3673 ± 0.0081      4.966 ± 0.203     0.0719 ± 0.0033
CGR (variance)      0.1840 ± 0.0172      3.067 ± 0.244     0.4433 ± 0.0082

Per-seed breakdown:
  Random:
    margin: ['-0.0121', '-0.0206', '-0.0062', '-0.0333', '-0.0115']
    forget: ['3.6310', '3.9940', '3.9740', '4.1800', '4.1670']
    conf: ['0.3092', '0.2974', '0.3065', '0.2941', '0.3004']
  High loss:
    margin: ['-0.3635', '-0.3587', '-0.3523', '-0.3607', '-0.3364']
    forget: ['4.5160', '4.9340', '5.0960', '5.1570', '4.9440']
    conf: ['0.0886', '0.0936', '0.0883', '0.1036', '0.1121']
  High confidence:
    margin: ['0.4005', '0.3567', '0.3755', '0.3639', '0.3457']
    forget: ['2.0660', '2.7270', '2.4170', '2.6970', '2.8150']
    conf: ['0.5851', '0.5544', '0.5643', '0.5590', '0.5480']
  Low confidence:
    margin: ['-0.3717', '-0.3713', '-0.3571', '-0.3780', '-0.3585']
    forget: ['4.6010', '4.8870', '5.1160', '5.1240', '5.1010']
    conf: ['0.0770', '0.0700', '0.0738', '0.0674', '0.0713']
  CGR (variance):
    margin: ['0.2125', '0.1787', '0.1936', '0.1649', '0.1703']
    forget: ['2.6100', '3.1080', '3.0850', '3.2050', '3.3260']
    conf: ['0.4575', '0.4392', '0.4471', '0.4355', '0.4372']

--- LaTeX (paste into Table tab:diagnostic) ---
\begin{tabular}{lccc}
\toprule
Selection rule & Mean margin $\downarrow$ & Forgetting events $\uparrow$ & Mean target conf. \\
\midrule
Random & $-0.017 \pm 0.009$ & $3.99 \pm 0.20$ & $0.302 \pm 0.006$ \\
High loss & $-0.354 \pm 0.010$ & $4.93 \pm 0.22$ & $0.097 \pm 0.009$ \\
High confidence & $0.368 \pm 0.019$ & $2.54 \pm 0.27$ & $0.562 \pm 0.013$ \\
Low confidence & $-0.367 \pm 0.008$ & $4.97 \pm 0.20$ & $0.072 \pm 0.003$ \\
CGR (variance) & $0.184 \pm 0.017$ & $3.07 \pm 0.24$ & $0.443 \pm 0.008$ \\
\bottomrule
\end{tabular}

```
