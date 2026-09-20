import torch
d = torch.load('cgr_diag_logs/cgr_diag_seed0.pt', weights_only=False, map_location='cpu')
correct = d['diag_correct'].numpy().astype(bool)
assert (d['diag_logit_margin'].numpy() > 0 == correct).all()
assert torch.allclose(d['diag_feat_dist_target'][correct], d['diag_feat_dist_pred'][correct], atol=1e-5)
assert (d['diag_feat_dist_target_absmin'] >= -1e-6).all()
# Nearest-pair gap equals |logit margin| on correctly-classified samples
import numpy as np
assert np.allclose(d['diag_nearest_pair_logit_gap'].numpy()[correct],
                   np.abs(d['diag_logit_margin'].numpy()[correct]), atol=1e-5)
print('All sanity checks passed')
