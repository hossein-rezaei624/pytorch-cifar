import torch, numpy as np
d = torch.load('cgr_diag_logs/cgr_diag_seed0.pt', weights_only=False, map_location='cpu')
correct = d['diag_correct'].numpy().astype(bool)

# 1. sign(logit_margin) matches correctness
assert ((d['diag_logit_margin'].numpy() > 0) == correct).all(), 'sign mismatch'

# 2. d_target == d_pred on correctly-classified samples
assert torch.allclose(d['diag_feat_dist_target'][correct],
                      d['diag_feat_dist_pred'][correct], atol=1e-5), 'd_target != d_pred on correct'

# 3. d_pred always ≥ 0 (allow tiny float noise)
assert (d['diag_feat_dist_pred'].numpy() >= -1e-6).all(), 'd_pred has negatives'

# 4. d_target_absmin always ≥ 0
assert (d['diag_feat_dist_target_absmin'].numpy() >= -1e-6).all(), 'd_target_absmin has negatives'

# 5. Nearest-pair logit gap equals |logit margin| on correctly-classified samples
assert np.allclose(d['diag_nearest_pair_logit_gap'].numpy()[correct],
                   np.abs(d['diag_logit_margin'].numpy()[correct]), atol=1e-5), 'nearest_pair != |logit margin| on correct'

# 6. Predicted top-two logit gap equals |logit margin| on correctly-classified samples
#    (redundancy proof: y_hat = y for correct samples)
assert np.allclose(d['diag_logit_margin_pred'].numpy()[correct],
                   np.abs(d['diag_logit_margin'].numpy()[correct]), atol=1e-5), 'logit_margin_pred != |logit margin| on correct'

# 7. No NaN or -1 leftover from allocation
assert not np.isnan(d['diag_logit_margin'].numpy()).any(), 'NaN in logit_margin'
assert (d['diag_labels'].numpy() >= 0).all(), 'unset labels'

print('All sanity checks passed')
