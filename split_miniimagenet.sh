Loaded 5 seed logs from cgr_diag_logs_new/cgr_diag_logs
  seed=0  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=1  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=2  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=3  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
  seed=4  E=4  n_epochs=50  n_samples=5000  buffer_size=1000
Config checks passed: E, buffer_size, epoch counts, tensor shapes, and diag_labels are consistent across all 5 seeds.

=== (b.1) Cross-seed Spearman correlation of variance vectors ===
Number of seeds: 5  (10 pairs)
Mean ρ ± std: 0.4887 ± 0.0141
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

=== (c) Within-seed Spearman: σ² at E=2 vs σ² at E=5 [Concern 4 anchor] ===
Per-seed ρ values:
  seed 0: ρ = 0.4412  (p = 2.559e-237)  ***
  seed 1: ρ = 0.4508  (p = 7.322e-249)  ***
  seed 2: ρ = 0.4279  (p = 7.662e-222)  ***
  seed 3: ρ = 0.4509  (p = 5.051e-249)  ***
  seed 4: ρ = 0.4680  (p = 1.229e-270)  ***

Mean ρ ± std over 5 seeds: 0.4478 ± 0.0147
Range: [0.4279, 0.4680]

  Paper insertion: \bar\rho_{E=2,E=5} = 0.448 \pm 0.015

=== (b.2) Diagnostic table (averaged over 5 seeds) ===
Per-class budget K = 100  (10 classes seen in task 1)
(Margins are CIL-style: competitors k range over ALL output classes.)

Rule                  ProbMargin (mean±std)   LogitMargin (mean±std)      Forget (mean±std)    MeanConf (mean±std)
------------------------------------------------------------------------------------------------------------------
Random             -0.0168 ± 0.0106    -0.5191 ± 0.2200      3.989 ± 0.222     0.3015 ± 0.0063
High loss          -0.3543 ± 0.0108    -3.3493 ± 1.4244      4.929 ± 0.250     0.0972 ± 0.0104
High confidence     0.3685 ± 0.0210     1.3148 ± 0.1566      2.544 ± 0.306     0.5622 ± 0.0141
Low confidence     -0.3673 ± 0.0091    -2.4876 ± 0.3353      4.966 ± 0.226     0.0719 ± 0.0037
CGR (variance)      0.1840 ± 0.0193     0.2365 ± 0.4347      3.067 ± 0.272     0.4433 ± 0.0091

Per-seed breakdown:
  Random:
    prob_margin: ['-0.0121', '-0.0206', '-0.0062', '-0.0333', '-0.0115']
    logit_margin: ['-0.3117', '-0.4113', '-0.3588', '-0.7460', '-0.7675']
    forget: ['3.6310', '3.9940', '3.9740', '4.1800', '4.1670']
    conf: ['0.3092', '0.2974', '0.3065', '0.2941', '0.3004']
  High loss:
    prob_margin: ['-0.3635', '-0.3587', '-0.3523', '-0.3607', '-0.3364']
    logit_margin: ['-2.2538', '-2.5406', '-2.3037', '-4.1850', '-5.4636']
    forget: ['4.5160', '4.9340', '5.0960', '5.1570', '4.9440']
    conf: ['0.0886', '0.0936', '0.0883', '0.1036', '0.1121']
  High confidence:
    prob_margin: ['0.4005', '0.3567', '0.3755', '0.3639', '0.3457']
    logit_margin: ['1.5110', '1.3249', '1.4173', '1.1873', '1.1337']
    forget: ['2.0660', '2.7270', '2.4170', '2.6970', '2.8150']
    conf: ['0.5851', '0.5544', '0.5643', '0.5590', '0.5480']
  Low confidence:
    prob_margin: ['-0.3717', '-0.3713', '-0.3571', '-0.3780', '-0.3585']
    logit_margin: ['-2.2086', '-2.3658', '-2.1819', '-2.7577', '-2.9243']
    forget: ['4.6010', '4.8870', '5.1160', '5.1240', '5.1010']
    conf: ['0.0770', '0.0700', '0.0738', '0.0674', '0.0713']
  CGR (variance):
    prob_margin: ['0.2125', '0.1787', '0.1936', '0.1649', '0.1703']
    logit_margin: ['0.6717', '0.4127', '0.5559', '-0.2383', '-0.2195']
    forget: ['2.6100', '3.1080', '3.0850', '3.2050', '3.3260']
    conf: ['0.4575', '0.4392', '0.4471', '0.4355', '0.4372']

--- LaTeX (paste into Table tab:diagnostic) ---
\begin{tabular}{lcccc}
\toprule
Selection rule & Mean prob. margin & Mean logit margin & Forgetting events $\uparrow$ & Mean target conf. \\
\midrule
Random & $-0.017 \pm 0.011$ & $-0.519 \pm 0.220$ & $3.99 \pm 0.22$ & $0.302 \pm 0.006$ \\
High loss & $-0.354 \pm 0.011$ & $-3.349 \pm 1.424$ & $4.93 \pm 0.25$ & $0.097 \pm 0.010$ \\
High confidence & $0.368 \pm 0.021$ & $1.315 \pm 0.157$ & $2.54 \pm 0.31$ & $0.562 \pm 0.014$ \\
Low confidence & $-0.367 \pm 0.009$ & $-2.488 \pm 0.335$ & $4.97 \pm 0.23$ & $0.072 \pm 0.004$ \\
CGR (variance) & $0.184 \pm 0.019$ & $0.237 \pm 0.435$ & $3.07 \pm 0.27$ & $0.443 \pm 0.009$ \\
\bottomrule
\end{tabular}

=== (d) Trajectory-outcome analysis [margin=prob, aggregation=final] ===
(measured in each sample's diagnostic pass in the final training epoch; averaged over 5 seeds; per-class thresholds; NOT a geometric boundary test)
(Percentile-rank interpretation: HIGH pctl on signed-margin = well-classified;
 LOW pctl on |margin| or d_pred = near boundary — proper boundary-proximity measures.)

--- Signed-margin percentiles and counts ---
Rule                     Correct(final-pass)   MedPctlGlob  MeanPctlGlob   MedPctlCls  MeanPctlCls    WellClsEnd    LowMargEnd     MisclsEnd
--------------------------------------------------------------------------------------------------------------------------------------------
CGR (high variance)              0.973±0.011      62.8±1.3      58.9±0.4     63.4±1.0     59.5±0.5   0.884±0.007   0.088±0.010   0.027±0.011
Random                           0.960±0.007      50.1±0.8      50.2±0.5     49.7±1.7     50.1±0.7   0.805±0.008   0.154±0.015   0.040±0.007
High confidence                  0.975±0.009      65.5±1.8      60.9±0.8     67.0±1.4     61.9±0.7   0.902±0.004   0.073±0.008   0.025±0.009
Low confidence                   0.927±0.010      34.7±0.8      38.0±0.8     33.0±1.3     37.4±0.9   0.670±0.010   0.258±0.013   0.073±0.010
High loss                        0.932±0.012      36.7±1.3      39.5±0.9     35.1±1.1     39.0±1.0   0.686±0.014   0.246±0.020   0.068±0.012

--- |margin| percentiles (boundary-proximity in margin space; LOW = near boundary) ---
Rule                     MedAbsMargGlob   MeanAbsMargGlob   MedAbsMargCls   MeanAbsMargCls
------------------------------------------------------------------------------------------
CGR (high variance)            62.8±1.4          59.0±0.5        63.4±1.0         59.6±0.7
Random                         50.1±0.9          50.2±0.6        49.6±1.7         50.1±0.7
High confidence                65.5±1.8          61.0±0.9        67.0±1.4         61.9±0.8
Low confidence                 34.8±0.8          38.1±0.9        33.1±1.3         37.4±1.0
High loss                      36.7±1.4          39.6±1.0        35.2±1.1         39.1±1.0

--- d_pred percentiles (boundary-proximity in feature space; LOW = near boundary) ---
Rule                     MedDpredGlob   MeanDpredGlob   MedDpredCls   MeanDpredCls
----------------------------------------------------------------------------------
CGR (high variance)          63.0±1.3        59.2±0.5      63.5±1.0       59.7±0.7
Random                       50.1±0.9        50.2±0.6      49.8±1.4       50.1±0.7
High confidence              65.8±1.4        61.3±0.9      67.0±1.3       62.1±0.8
Low confidence               34.5±0.9        37.7±0.8      33.1±1.2       37.1±1.0
High loss                    36.4±1.1        39.2±0.9      35.1±1.1       38.8±0.9

=== (d) Trajectory-outcome analysis [margin=prob, aggregation=last_E] ===
(measured averaged across each sample's diagnostic passes in the last E training epochs; averaged over 5 seeds; per-class thresholds; NOT a geometric boundary test)
(Percentile-rank interpretation: HIGH pctl on signed-margin = well-classified;
 LOW pctl on |margin| or d_pred = near boundary — proper boundary-proximity measures.)

--- Signed-margin percentiles and counts ---
Rule                     Correct(final-pass)   MedPctlGlob  MeanPctlGlob   MedPctlCls  MeanPctlCls    WellClsEnd    LowMargEnd     MisclsEnd
--------------------------------------------------------------------------------------------------------------------------------------------
CGR (high variance)              0.981±0.009      63.9±0.5      59.5±0.3     64.3±0.8     60.1±0.4   0.886±0.012   0.095±0.008   0.019±0.009
Random                           0.968±0.008      50.2±1.8      50.4±1.0     50.6±2.4     50.4±1.2   0.812±0.010   0.156±0.014   0.032±0.008
High confidence                  0.984±0.008      67.3±1.8      61.8±1.0     68.8±1.5     62.9±1.3   0.898±0.011   0.086±0.009   0.016±0.008
Low confidence                   0.938±0.011      33.4±0.4      36.8±0.5     31.7±0.9     35.9±0.6   0.657±0.011   0.281±0.015   0.062±0.011
High loss                        0.943±0.014      34.9±0.9      38.2±0.7     33.1±1.3     37.6±0.8   0.679±0.017   0.264±0.024   0.057±0.014

--- |margin| percentiles (boundary-proximity in margin space; LOW = near boundary) ---
Rule                     MedAbsMargGlob   MeanAbsMargGlob   MedAbsMargCls   MeanAbsMargCls
------------------------------------------------------------------------------------------
CGR (high variance)            63.9±0.4          59.5±0.5        64.3±0.8         60.2±0.6
Random                         50.1±1.8          50.3±1.1        50.6±2.1         50.3±1.2
High confidence                67.4±1.7          61.9±1.0        68.9±1.5         63.0±1.2
Low confidence                 33.5±0.3          36.9±0.6        31.7±0.7         36.0±0.7
High loss                      35.0±0.7          38.4±0.7        33.5±1.3         37.7±0.9

--- d_pred percentiles (boundary-proximity in feature space; LOW = near boundary) ---
Rule                     MedDpredGlob   MeanDpredGlob   MedDpredCls   MeanDpredCls
----------------------------------------------------------------------------------
CGR (high variance)          65.1±0.9        60.5±0.6      65.7±1.2       61.3±0.8
Random                       50.0±1.9        50.2±0.9      50.2±2.1       50.3±1.1
High confidence              68.9±1.7        62.9±1.0      70.1±1.7       64.2±1.1
Low confidence               31.7±1.1        35.5±0.9      29.8±1.5       34.5±0.9
High loss                    33.7±1.2        37.3±0.8      32.3±1.4       36.6±1.0

=== (d) Trajectory-outcome analysis [margin=logit, aggregation=final] ===
(measured in each sample's diagnostic pass in the final training epoch; averaged over 5 seeds; per-class thresholds; NOT a geometric boundary test)
(Percentile-rank interpretation: HIGH pctl on signed-margin = well-classified;
 LOW pctl on |margin| or d_pred = near boundary — proper boundary-proximity measures.)

--- Signed-margin percentiles and counts ---
Rule                     Correct(final-pass)   MedPctlGlob  MeanPctlGlob   MedPctlCls  MeanPctlCls    WellClsEnd    LowMargEnd     MisclsEnd
--------------------------------------------------------------------------------------------------------------------------------------------
CGR (high variance)              0.973±0.011      62.8±1.2      58.9±0.4     63.1±1.0     59.5±0.5   0.885±0.008   0.088±0.010   0.027±0.011
Random                           0.960±0.007      50.1±1.0      50.2±0.5     49.6±1.8     50.1±0.7   0.806±0.009   0.153±0.016   0.040±0.007
High confidence                  0.975±0.009      65.4±1.7      61.0±0.8     66.7±1.2     61.9±0.7   0.901±0.003   0.074±0.007   0.025±0.009
Low confidence                   0.927±0.010      35.0±1.0      38.1±0.7     33.2±1.2     37.4±0.9   0.671±0.011   0.256±0.013   0.073±0.010
High loss                        0.932±0.012      37.0±1.3      39.6±0.9     35.3±1.5     39.1±0.9   0.687±0.015   0.245±0.021   0.068±0.012

--- |margin| percentiles (boundary-proximity in margin space; LOW = near boundary) ---
Rule                     MedAbsMargGlob   MeanAbsMargGlob   MedAbsMargCls   MeanAbsMargCls
------------------------------------------------------------------------------------------
CGR (high variance)            62.9±1.3          58.9±0.5        63.1±1.0         59.5±0.7
Random                         50.1±1.1          50.2±0.6        49.6±1.9         50.1±0.7
High confidence                65.4±1.7          61.0±0.8        66.7±1.3         61.9±0.8
Low confidence                 35.0±1.1          38.2±0.9        33.4±1.4         37.5±1.0
High loss                      37.0±1.2          39.7±0.9        35.4±1.5         39.2±1.0

--- d_pred percentiles (boundary-proximity in feature space; LOW = near boundary) ---
Rule                     MedDpredGlob   MeanDpredGlob   MedDpredCls   MeanDpredCls
----------------------------------------------------------------------------------
CGR (high variance)          63.0±1.3        59.2±0.5      63.5±1.0       59.7±0.7
Random                       50.1±0.9        50.2±0.6      49.8±1.4       50.1±0.7
High confidence              65.8±1.4        61.3±0.9      67.0±1.3       62.1±0.8
Low confidence               34.5±0.9        37.7±0.8      33.1±1.2       37.1±1.0
High loss                    36.4±1.1        39.2±0.9      35.1±1.1       38.8±0.9

=== (d) Trajectory-outcome analysis [margin=logit, aggregation=last_E] ===
(measured averaged across each sample's diagnostic passes in the last E training epochs; averaged over 5 seeds; per-class thresholds; NOT a geometric boundary test)
(Percentile-rank interpretation: HIGH pctl on signed-margin = well-classified;
 LOW pctl on |margin| or d_pred = near boundary — proper boundary-proximity measures.)

--- Signed-margin percentiles and counts ---
Rule                     Correct(final-pass)   MedPctlGlob  MeanPctlGlob   MedPctlCls  MeanPctlCls    WellClsEnd    LowMargEnd     MisclsEnd
--------------------------------------------------------------------------------------------------------------------------------------------
CGR (high variance)              0.981±0.009      64.8±0.9      60.1±0.5     65.5±1.0     61.1±0.6   0.897±0.012   0.085±0.015   0.019±0.009
Random                           0.968±0.008      50.2±1.5      50.3±0.9     50.4±2.6     50.3±1.0   0.804±0.006   0.164±0.014   0.032±0.008
High confidence                  0.984±0.008      68.0±1.8      62.5±0.9     69.9±1.5     64.0±1.0   0.910±0.006   0.073±0.011   0.016±0.008
Low confidence                   0.938±0.011      32.4±1.1      36.1±0.7     30.2±1.3     34.8±0.8   0.640±0.013   0.299±0.009   0.062±0.011
High loss                        0.943±0.014      34.0±0.7      37.7±0.8     32.6±1.5     36.8±1.0   0.664±0.013   0.279±0.016   0.057±0.014

--- |margin| percentiles (boundary-proximity in margin space; LOW = near boundary) ---
Rule                     MedAbsMargGlob   MeanAbsMargGlob   MedAbsMargCls   MeanAbsMargCls
------------------------------------------------------------------------------------------
CGR (high variance)            64.8±0.9          60.1±0.6        65.4±1.2         61.1±0.8
Random                         50.2±1.5          50.3±0.9        50.1±2.5         50.3±1.1
High confidence                68.2±1.7          62.4±0.9        69.9±1.4         64.0±1.0
Low confidence                 32.6±0.9          36.2±0.8        30.4±1.4         35.0±0.9
High loss                      34.4±0.8          37.9±0.7        32.7±1.4         37.0±0.9

--- d_pred percentiles (boundary-proximity in feature space; LOW = near boundary) ---
Rule                     MedDpredGlob   MeanDpredGlob   MedDpredCls   MeanDpredCls
----------------------------------------------------------------------------------
CGR (high variance)          65.1±0.9        60.5±0.6      65.7±1.2       61.3±0.8
Random                       50.0±1.9        50.2±0.9      50.2±2.1       50.3±1.1
High confidence              68.9±1.7        62.9±1.0      70.1±1.7       64.2±1.1
Low confidence               31.7±1.1        35.5±0.9      29.8±1.5       34.5±0.9
High loss                    33.7±1.2        37.3±0.8      32.3±1.4       36.6±1.0

=== Overlap of CGR selection with per-class bottom-K under alternative criteria ===
(Chance overlap under Random selection ≈ 20% for K/N_class = 0.2.)

Criterion                   in epoch-E pass    over first-E window
------------------------------------------------------------------
d_pred                        0.066±0.006            0.011±0.006
m_pred                        0.066±0.005            0.008±0.004
nearest_pair_gap              0.059±0.004            0.001±0.001
abs_m_tgt                     0.077±0.006            0.024±0.006
signed_m_tgt                  0.071±0.010            0.055±0.035
(signed_m_tgt shows the ordering pathology GPT flagged: 'smallest' picks confidently-wrong samples, not near-boundary. Included for comparison.)

=== (d-descriptive) CGR-selected samples: margin trajectory ===
  probability margin  : early (first E) = 0.184 ± 0.019    late (last 4) = 0.929 ± 0.016
  logit margin        : early (first E) = 0.237 ± 0.435    late (last 4) = 11.038 ± 1.024
  (Descriptive training dynamics — NOT a geometric boundary claim.)

=== (d2) Selection-time boundary diagnostics [Concern 2, extended] ===
(measured during each sample's diagnostic pass in epoch E; NOT a common
 end-of-epoch checkpoint. Averaged over 5 seeds.)

--- POINT-IN-TIME snapshots during each sample's diagnostic pass in epoch E (primary) ---
Rule                    SignChgSt 1..E    Correct@E    ProbMSgn    ProbMPrd   LogitMSgn    |LogitM|   LogitMPrd
---------------------------------------------------------------------------------------------------------------
CGR (high variance)        0.969±0.006  0.826±0.011 0.500±0.025 0.633±0.027   1.83±0.13   2.46±0.16   2.35±0.15
Random                     0.634±0.005  0.503±0.015 0.073±0.012 0.405±0.025   0.05±0.07   1.85±0.12   1.43±0.10
High confidence            0.659±0.003  0.925±0.004 0.596±0.033 0.635±0.029   2.29±0.17   2.44±0.15   2.43±0.15
Low confidence             0.113±0.015  0.033±0.006 -0.425±0.024 0.317±0.029  -2.44±0.12   2.47±0.12   1.08±0.10
High loss                  0.244±0.028  0.116±0.034 -0.368±0.056 0.351±0.022  -2.28±0.30   2.62±0.18   1.23±0.08

Rule                     NearPairLogitGap    NearPairProbGap  d_tgt_signed  d_tgt_absmin        d_pred  d_on_correct
--------------------------------------------------------------------------------------------------------------------
CGR (high variance)             2.23±0.15        0.582±0.025     1.55±0.14     1.89±0.13     1.99±0.13     2.20±0.16
Random                          1.13±0.08        0.279±0.015     0.04±0.06     0.94±0.06     1.20±0.08     1.57±0.11
High confidence                 2.40±0.16        0.623±0.031     1.93±0.15     2.02±0.14     2.05±0.13     2.16±0.15
Low confidence                  0.33±0.02        0.023±0.002    -1.97±0.08     0.26±0.01     0.91±0.09     0.34±0.07
High loss                       0.47±0.05        0.062±0.017    -1.82±0.21     0.38±0.04     1.04±0.06     1.18±0.19

Rule                      bot20% |m|   bot20% d_pred
----------------------------------------------------
CGR (high variance)      0.077±0.006     0.066±0.006
Random                   0.198±0.012     0.201±0.016
High confidence          0.092±0.008     0.053±0.011
Low confidence           0.090±0.011     0.262±0.008
High loss                0.103±0.012     0.244±0.009

--- WINDOW aggregates over first E epochs (complementary) ---
(Per-sample mean and range across epochs 0..E-1; then averaged across selection.)

Rule                   ProbMSgn mean LogitMSgn mean |LogitM| mean LogitMPrd mean NearPairGap mean   d_pred mean
-----------------------------------------------------------------------------------------------------------
CGR (high variance)        0.18±0.02     0.24±0.43     2.30±0.55     1.84±0.34     1.43±0.11     1.90±0.47
Random                    -0.02±0.01    -0.52±0.22     1.80±0.24     1.23±0.17     0.82±0.03     1.28±0.23
High confidence            0.37±0.02     1.31±0.16     1.98±0.16     1.85±0.11     1.72±0.05     1.88±0.18
Low confidence            -0.37±0.01    -2.49±0.34     2.51±0.33     1.04±0.16     0.28±0.01     1.10±0.23
High loss                 -0.35±0.01    -3.35±1.42     3.56±1.49     1.63±0.86     0.44±0.12     1.81±1.10

Rule                    ProbMSgn rng LogitMSgn rng  |LogitM| rng LogitMPrd rng NearPairGap rng    d_pred rng
-----------------------------------------------------------------------------------------------------------
CGR (high variance)        1.15±0.02     6.74±2.23     4.67±2.23     3.84±1.32     3.26±0.44     3.97±1.82
Random                     0.70±0.02     4.05±0.98     3.19±0.99     2.36±0.66     1.59±0.10     2.45±0.90
High confidence            0.83±0.01     4.24±0.70     3.24±0.67     3.04±0.43     2.92±0.15     3.02±0.65
Low confidence             0.46±0.02     3.77±1.26     3.69±1.27     2.08±0.65     0.56±0.05     2.24±0.89
High loss                  0.62±0.05     8.08±6.02     7.64±5.94     4.10±3.34     0.96±0.34     4.75±4.28

--- Overlaps of CGR selection with per-class bottom-K during epoch-E diagnostic pass ---
  by |logit margin|         : 0.077 ± 0.006
  by d_pred (feat-space)    : 0.066 ± 0.006
  by nearest_pair_logit_gap : 0.059 ± 0.004
  by logit_margin_pred      : 0.066 ± 0.005

--- MEDIAN and IQR (robust) for skewed quantities at epoch-E diagnostic pass ---
(Reported as: within-selection median  ±  cross-seed std of that median;
              within-selection IQR (P75-P25)  ±  cross-seed std of that IQR.)

  prob_margin_pred:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      0.701±0.046     0.444±0.027
    Random                   0.343±0.035     0.522±0.041
    High confidence          0.694±0.044     0.467±0.027
    Low confidence           0.243±0.032     0.396±0.040
    High loss                0.272±0.033     0.444±0.027

  logit_margin_pred:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      2.163±0.174     2.046±0.171
    Random                   1.022±0.074     1.580±0.155
    High confidence          2.133±0.210     2.201±0.166
    Low confidence           0.778±0.091     1.163±0.108
    High loss                0.847±0.083     1.313±0.081

  abs_logit_margin:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      2.263±0.163     1.945±0.126
    Random                   1.474±0.121     1.907±0.161
    High confidence          2.151±0.201     2.182±0.160
    Low confidence           2.192±0.074     1.964±0.153
    High loss                2.334±0.140     2.270±0.126

  nearest_pair_gap:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      2.070±0.168     2.196±0.160
    Random                   0.578±0.052     1.383±0.096
    High confidence          2.118±0.212     2.227±0.158
    Low confidence           0.206±0.014     0.352±0.029
    High loss                0.240±0.007     0.443±0.029

  nearest_pair_prob_gap:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      0.675±0.047     0.563±0.023
    Random                   0.106±0.006     0.504±0.041
    High confidence          0.689±0.047     0.486±0.026
    Low confidence           0.008±0.000     0.022±0.001
    High loss                0.008±0.001     0.031±0.006

  d_target_signed:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      1.810±0.148     1.891±0.146
    Random                   0.014±0.057     2.506±0.180
    High confidence          1.842±0.163     1.770±0.139
    Low confidence          -1.813±0.067     1.604±0.076
    High loss               -1.829±0.132     1.979±0.124

  d_target_absmin:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      1.821±0.144     1.781±0.145
    Random                   0.494±0.052     1.180±0.058
    High confidence          1.847±0.158     1.754±0.145
    Low confidence           0.168±0.013     0.288±0.029
    High loss                0.195±0.012     0.367±0.030

  d_pred:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      1.892±0.151     1.631±0.158
    Random                   0.891±0.060     1.336±0.126
    High confidence          1.860±0.153     1.726±0.150
    Low confidence           0.656±0.075     0.992±0.087
    High loss                0.716±0.079     1.099±0.075

  d_on_correct_subset:
    Rule                             Median              IQR
    ---------------------- ---------------- ----------------
    CGR (high variance)      2.077±0.159     1.439±0.140
    Random                   1.289±0.079     1.587±0.140
    High confidence          1.964±0.177     1.662±0.129
    Low confidence           0.243±0.033     0.335±0.113
    High loss                0.779±0.126     1.303±0.236
