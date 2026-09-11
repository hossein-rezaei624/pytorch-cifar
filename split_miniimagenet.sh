"""
Modified cgr.py with diagnostic logging for the rebuttal experiments.

IMPORTANT NOTE ON TIMING: The diagnostic forward pass runs INSIDE `observe`,
i.e. BEFORE that batch's SGD update. This is faithful to CGR's own confidence
recording, but consequently different samples within the same epoch are
evaluated under slightly different model states. The tensors below therefore
record, for each (sample, epoch) pair, the eval-mode quantity computed
DURING THAT SAMPLE'S DIAGNOSTIC PASS in that epoch — not a common
end-of-epoch checkpoint. Downstream analyses should describe them as such.

When --cgr_diag_log is set, this version:
  * During task 1 ONLY, runs CGR's existing eval-mode forward pass on
    not_aug_inputs for ALL epochs of the task (instead of just the first E).
  * Records, from that same eval-mode pass, per-sample per-epoch:
      -- target confidence (CGR's existing signal)
      -- probability margin p_y - max_{k!=y} p_k                     (`diag_margin`, signed)
      -- probability margin p_yhat - max_{k!=yhat} p_k               (`diag_margin_pred`, >=0)
      -- correctness (argmax == label)                                (`diag_correct`)
      -- per-sample cross-entropy loss                                (`diag_loss`)
      -- logit margin z_y - max_{k!=y} z_k                            (`diag_logit_margin`, signed)
      -- logit margin z_yhat - max_{k!=yhat} z_k                      (`diag_logit_margin_pred`, >=0)
      -- nearest pairwise |z_y - z_k| over k != y                     (`diag_nearest_pair_logit_gap`, >=0)
      -- nearest pairwise |p_y - p_k| over k != y                     (`diag_nearest_pair_probability_gap`, >=0)
    Probability margin and logit margin share sign but not magnitude or
    cross-sample ranking; nearest-pair gaps and target/pred aggregated
    margins are ALSO different for misclassified samples (see below).
    All are recorded so downstream analyses can use the most appropriate one.
  * ALSO records, from the same eval-mode pass, per-sample feature-space
    quantities using the classifier head's weight matrix:
      -- d_target_signed = min_{k != y}   (z_y   - z_k) / ||w_y   - w_k||_2   (`diag_feat_dist_target`, signed)
      -- d_target_absmin = min_{k != y}  |(z_y   - z_k)|/ ||w_y   - w_k||_2   (`diag_feat_dist_target_absmin`, >=0)
      -- d_pred          = min_{k != yhat}(z_yhat- z_k) / ||w_yhat - w_k||_2  (`diag_feat_dist_pred`, >=0)
    where w_k are the rows of the classifier head at epoch e. For correctly
    classified samples all three coincide (d_target_signed = d_target_absmin
    = d_pred, and all equal the signed distance to the nearest boundary of
    the correct-class region). For misclassified samples:
      -- d_target_signed is the most-violated normalized pairwise target
         constraint (negative). NOT a geometric distance to the target-region
         boundary.
      -- d_target_absmin is the geometric distance to the nearest complete
         target-vs-competitor hyperplane. It is generally a LOWER BOUND on
         the distance to the full target-class region and may correspond to
         an inactive pairwise hyperplane (the target region is the intersection
         of many half-spaces; the nearest pairwise hyperplane can be closer
         than the region itself).
      -- d_pred remains the signed distance to the nearest boundary of the
         predicted-class region.
  * After the last epoch of task 1, saves all of the above plus CGR's
    confidence trajectory to disk as cgr_diag_logs/cgr_diag_seed<S>.pt
    (one file per run).

Run separately for each seed (--seed 0, --seed 1, ...). Each run produces
one .pt file. Combine across seeds by hand.

When --cgr_diag_log is NOT set, behaviour is identical to your original cgr.py.

All additions are marked with `# === DIAG: ... === / === END DIAG ===` blocks.
"""

import os
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='CGR: Confidence-Guided Reply for Buffer-Based Continual Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--E', type=int, default=4,
                        help='Epoch for selecting samples')
    # === DIAG: new CLI args ===
    parser.add_argument('--cgr_diag_log', action='store_true',
                        help='If set, record per-sample diagnostics during task 1 (from CGR\'s eval forward pass) and save to disk.')
    parser.add_argument('--cgr_diag_dir', type=str, default='cgr_diag_logs',
                        help='Directory to save per-seed diagnostic logs.')
    # === END DIAG ===
    return parser


def distribute_samples(probabilities, M):
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    discrepancy = M - sum(samples.values())
    for key in samples:
        if discrepancy == 0:
            break
        if discrepancy > 0:
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            samples[key] -= 1
            discrepancy += 1
    return samples


def distribute_excess(lst, check_bound):
    total_excess = sum(val - check_bound for val in lst if val > check_bound)
    recipients = [i for i, val in enumerate(lst) if val < check_bound]
    num_recipients = len(recipients)
    avg_share, remainder = divmod(total_excess, num_recipients)
    lst = [val if val <= check_bound else check_bound for val in lst]
    for idx in recipients:
        lst[idx] += avg_share
    for idx in recipients[:remainder]:
        lst[idx] += 1
    for i, val in enumerate(lst):
        if val > check_bound:
            return distribute_excess(lst, check_bound)
    return lst


def adjust_values_integer_include_all(a, b):
    excess = {}
    shortage = {}
    total_excess = 0
    for k in a:
        if k in b:
            if a[k] > b[k]:
                excess[k] = a[k] - b[k]
                total_excess += a[k] - b[k]
                a[k] = b[k]
            elif a[k] < b[k]:
                shortage[k] = b[k] - a[k]
        else:
            shortage[k] = float('inf')
    while total_excess > 0 and shortage:
        per_key_excess = max(total_excess // len(shortage), 1)
        for k in list(shortage):
            if total_excess == 0:
                break
            if shortage[k] == float('inf'):
                increment = per_key_excess
            else:
                increment = min(shortage[k], per_key_excess)
            a[k] += increment
            total_excess -= increment
            if shortage[k] != float('inf'):
                shortage[k] -= increment
                if shortage[k] == 0:
                    del shortage[k]
    for key in a:
        a[key] = int(a[key])
    return a


class Cgr(ContinualModel):
    NAME = 'cgr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Cgr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.unique_classes = set()
        self.mapping = {}
        self.reverse_mapping = {}
        self.confidence_by_sample = None
        self.n_sample_per_task = None
        self.class_portion = []
        self.dist_task_prev = None
        self.dist_class_prev = None
        # === DIAG: per-sample diagnostic tensors (allocated in begin_task for task 1 only) ===
        # All tensors are shape (n_epochs, n_sample_per_task) unless noted. Each entry is
        # recorded during that sample's eval-mode diagnostic pass in that epoch (see the
        # module docstring for the timing caveat — this is NOT a common end-of-epoch
        # checkpoint).
        self.diag_margin = None                        # p_y - max_{k!=y} p_k (signed prob margin)
        self.diag_margin_pred = None                   # p_yhat - max_{k!=yhat} p_k (>=0)
        self.diag_correct = None                       # bool: argmax == label
        self.diag_loss = None                          # per-sample cross-entropy loss
        self.diag_labels = None                        # (n_sample_per_task,) global class id
        self.diag_logit_margin = None                  # z_y - max_{k!=y} z_k (signed logit margin)
        self.diag_logit_margin_pred = None             # z_yhat - max_{k!=yhat} z_k (>=0)
        self.diag_nearest_pair_logit_gap = None        # min_{k!=y}    |z_y    - z_k| (>=0)
        self.diag_nearest_pair_probability_gap = None # min_{k!=y} |p_y - p_k| (>=0)
        # Feature-space (affine-head) distances (Concern 2):
        self.diag_feat_dist_target = None              # min_{k!=y} (z_y - z_k)/||w_y - w_k||_2 (signed)
        self.diag_feat_dist_target_absmin = None       # min_{k!=y} |(z_y - z_k)|/||w_y - w_k||_2 (>=0)
        self.diag_feat_dist_pred = None                # min_{k!=yhat} (z_yhat - z_k)/||w_yhat - w_k||_2 (>=0)
        # === END DIAG ===

    def _diag_active(self):
        """True iff diagnostic logging is enabled AND we're in task 1."""
        return getattr(self.args, 'cgr_diag_log', False) and self.task == 1

    # === DIAG: helper for feature-space distance analysis (Concern 2) ===
    def _diag_get_classifier_head(self):
        """Return the last nn.Linear in the network, assumed to be the classifier
        head. Robust to networks wrapped in Sequential / ContinualLearner
        modules (as in Mammoth's ResNet18 setup)."""
        last_linear = None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise RuntimeError("_diag_get_classifier_head: no nn.Linear found in self.net")
        return last_linear

    def _diag_feat_distances(self, logits, labels_dev):
        """Compute per-sample feature-space quantities using the classifier
        head's weight matrix. The bias term (b_y - b_k) is already absorbed
        into the logit difference z_y - z_k = (w_y - w_k)^T phi + (b_y - b_k),
        so only the pairwise weight-vector norms ||w_i - w_j||_2 are needed
        for the denominator.

        Three per-sample quantities are returned:

        d_target_signed(i) = min_{k != y_i} (z_{y_i} - z_k) / ||w_{y_i} - w_k||_2
            Interpretation depends on classification status:
              - Correctly classified (y_hat_i == y_i): all pairwise ratios >= 0,
                so d_target_signed = d_target_absmin = signed distance to the
                nearest boundary of the (correct) target-class region.
              - Misclassified (y_hat_i != y_i): negative; the MOST-VIOLATED
                normalized pairwise target constraint. NOT a geometric distance
                to the target-region boundary. Downstream analyses should
                describe it as the "minimum normalized target margin".

        d_target_absmin(i) = min_{k != y_i} |(z_{y_i} - z_k)| / ||w_{y_i} - w_k||_2
            Always >= 0. The distance to the nearest complete target-versus-
            competitor hyperplane. Coincides with d_target_signed for
            correctly-classified samples. For misclassified samples it is
            generally a LOWER BOUND on the distance from phi(x_i) to the full
            target region (which is the intersection of many half-spaces and
            would require a QP to compute exactly), and it may correspond to
            an inactive pairwise hyperplane.

        d_pred(i) = min_{k != y_hat_i} (z_{y_hat_i} - z_k) / ||w_{y_hat_i} - w_k||_2
            Always >= 0 (y_hat is the argmax by construction). Geometrically
            the signed perpendicular distance from phi(x_i) to the nearest
            boundary of the *predicted* class region under the affine head.

        For the cleanest "distance to correct-region boundary" analysis,
        restrict to samples where diag_correct == True; on that subset all
        three coincide.

        Args:
            logits: (B, C) raw logits from the eval-mode forward pass
            labels_dev: (B,) target class indices on the same device

        Returns:
            d_target_signed: (B,) see above.
            d_target_absmin: (B,) see above; always >= 0.
            d_pred:          (B,) see above; always >= 0.
        """
        head = self._diag_get_classifier_head()
        W = head.weight.detach()  # (C, d_feat)
        # Pairwise weight-vector distances ||w_i - w_j||_2 (C, C)
        # Small (100x100 for CIFAR-100) so cost is negligible.
        W_dist = torch.cdist(W.unsqueeze(0), W.unsqueeze(0), p=2).squeeze(0)

        B, C = logits.shape

        # -- Target-based ratios: (z_y - z_k) / ||w_y - w_k||, over k --
        target_logits = logits.gather(1, labels_dev.unsqueeze(1)).squeeze(1)  # (B,)
        z_diff_t = target_logits.unsqueeze(1) - logits                         # (B, C)
        w_diff_t = W_dist[labels_dev]                                          # (B, C)
        # avoid div-by-zero at k=y (0/0); mask k=y before reducing
        ratio_t = z_diff_t / w_diff_t.clamp(min=1e-12)                         # (B, C)
        mask_t = torch.zeros_like(ratio_t, dtype=torch.bool)
        mask_t.scatter_(1, labels_dev.unsqueeze(1), True)

        # d_target_signed: min over k != y of the signed ratio (may be negative)
        ratio_t_masked_inf = ratio_t.masked_fill(mask_t, float('inf'))
        d_target_signed = ratio_t_masked_inf.min(dim=1).values                 # (B,)

        # d_target_absmin: min over k != y of |ratio| — nearest single hyperplane
        # (mask k=y with +inf so it's excluded from the min over absolute values)
        abs_ratio_t_masked_inf = ratio_t.abs().masked_fill(mask_t, float('inf'))
        d_target_absmin = abs_ratio_t_masked_inf.min(dim=1).values             # (B,)

        # -- d_pred: same as d_target_signed but using argmax class --
        y_hat = logits.argmax(dim=1)                                            # (B,)
        pred_logits = logits.gather(1, y_hat.unsqueeze(1)).squeeze(1)           # (B,)
        z_diff_p = pred_logits.unsqueeze(1) - logits                            # (B, C)
        w_diff_p = W_dist[y_hat]                                                # (B, C)
        ratio_p = z_diff_p / w_diff_p.clamp(min=1e-12)
        mask_p = torch.zeros_like(ratio_p, dtype=torch.bool)
        mask_p.scatter_(1, y_hat.unsqueeze(1), True)
        d_pred = ratio_p.masked_fill(mask_p, float('inf')).min(dim=1).values    # (B,)

        return d_target_signed, d_target_absmin, d_pred
    # === END DIAG ===

    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number() // dataset.N_TASKS

    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        self.unique_classes = set()
        for _, labels, _, _ in train_loader:
            self.unique_classes.update(labels.numpy())
            if len(self.unique_classes) == dataset.N_CLASSES_PER_TASK:
                break
        self.mapping = {value: index for index, value in enumerate(self.unique_classes)}
        self.reverse_mapping = {index: value for value, index in self.mapping.items()}
        self.confidence_by_sample = torch.zeros((self.args.n_epochs, self.n_sample_per_task))

        # === DIAG: allocate task-1 diagnostic tensors ===
        # Numeric tensors are initialized with NaN so unset entries (should not exist,
        # but sanity-checkable) are detectable in the log.
        if self._diag_active():
            n_e = self.args.n_epochs
            n_s = self.n_sample_per_task
            def _nan(shape): return torch.full(shape, float('nan'))
            self.diag_margin = _nan((n_e, n_s))
            self.diag_margin_pred = _nan((n_e, n_s))
            self.diag_correct = torch.zeros((n_e, n_s), dtype=torch.bool)
            self.diag_loss = _nan((n_e, n_s))
            self.diag_labels = torch.full((n_s,), -1, dtype=torch.long)
            self.diag_logit_margin = _nan((n_e, n_s))
            self.diag_logit_margin_pred = _nan((n_e, n_s))
            self.diag_nearest_pair_logit_gap = _nan((n_e, n_s))
            self.diag_nearest_pair_probability_gap = _nan((n_e, n_s))
            self.diag_feat_dist_target = _nan((n_e, n_s))
            self.diag_feat_dist_target_absmin = _nan((n_e, n_s))
            self.diag_feat_dist_pred = _nan((n_e, n_s))
        # === END DIAG ===

    def _save_diag(self):
        """Dump task-1 diagnostics to disk at the end of task 1's last epoch."""
        if not self._diag_active():
            return
        save_dir = getattr(self.args, 'cgr_diag_dir', 'cgr_diag_logs')
        os.makedirs(save_dir, exist_ok=True)
        seed = getattr(self.args, 'seed', 'unknown')
        save_path = os.path.join(save_dir, f'cgr_diag_seed{seed}.pt')
        torch.save({
            'seed': seed,
            'E': self.args.E,
            'n_epochs': self.args.n_epochs,
            'n_sample_per_task': self.n_sample_per_task,
            'buffer_size': self.args.buffer_size,
            # CGR's eval-mode target confidence. With --cgr_diag_log this is
            # filled for ALL epochs of task 1; use [:E] for CGR's variance.
            'cgr_confidence_by_sample': self.confidence_by_sample.clone(),
            # ---- Diagnostics from the same eval-mode pass on not_aug_inputs ----
            # See module docstring for the timing caveat: each entry is
            # recorded during that sample's eval-mode pass in that epoch,
            # which is BEFORE the batch's SGD update. Different samples in
            # the same epoch see slightly different model states.
            'diag_correct': self.diag_correct.clone(),                                   # bool: argmax == label
            'diag_loss':    self.diag_loss.clone(),                                      # per-sample CE loss
            'diag_labels':  self.diag_labels.clone(),                                    # (n_s,) global class id
            # Scalar margin quantities (all shape (n_epochs, n_sample_per_task)):
            'diag_margin':                       self.diag_margin.clone(),                       # signed prob margin (target)
            'diag_margin_pred':                  self.diag_margin_pred.clone(),                  # prob margin (predicted, >=0)
            'diag_logit_margin':                 self.diag_logit_margin.clone(),                 # signed logit margin (target)
            'diag_logit_margin_pred':            self.diag_logit_margin_pred.clone(),            # logit margin (predicted, >=0)
            'diag_nearest_pair_logit_gap':       self.diag_nearest_pair_logit_gap.clone(),       # min_{k!=y}    |z_y    - z_k|
            'diag_nearest_pair_probability_gap':  self.diag_nearest_pair_probability_gap.clone(),  # min_{k!=y} |p_y - p_k|
            # Feature-space (affine-head) distances:
            'diag_feat_dist_target':         self.diag_feat_dist_target.clone(),         # signed min over k!=y of ratios
            'diag_feat_dist_target_absmin':  self.diag_feat_dist_target_absmin.clone(),  # min over k!=y of |ratios|
            'diag_feat_dist_pred':           self.diag_feat_dist_pred.clone(),           # signed min over k!=yhat of ratios (>=0)
            'class_mapping': dict(self.mapping),
        }, save_path)
        print(f"[CGR-Diag] Saved task-1 diagnostics to {save_path}")

    def end_epoch(self, dataset, train_loader):

        self.epoch += 1

        if self.epoch == self.args.n_epochs:
            # === DIAG: dump task-1 diagnostics before the buffer-update logic ===
            self._save_diag()
            # === END DIAG ===

            # ... rest of the function unchanged from original ...
            std_of_means_by_class = {class_id: 1 for class_id, __ in enumerate(self.unique_classes)}
            std_of_means_by_task = {task_id: 1 for task_id in range(self.task)}

            Confidence_mean = self.confidence_by_sample[:self.args.E].mean(dim=0)
            Variability = self.confidence_by_sample[:self.args.E].var(dim=0)

            sorted_indices_2 = np.argsort(Variability.numpy())
            top_indices_sorted = sorted_indices_2[::-1].copy()

            all_inputs, all_labels, all_not_aug_inputs, all_indices = [], [], [], []
            for data_1 in train_loader:
                inputs_1, labels_1, not_aug_inputs_1, indices_1 = data_1
                all_inputs.append(inputs_1)
                all_labels.append(labels_1)
                all_not_aug_inputs.append(not_aug_inputs_1)
                all_indices.append(indices_1)

            all_inputs = torch.cat(all_inputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_not_aug_inputs = torch.cat(all_not_aug_inputs, dim=0)
            all_indices = torch.cat(all_indices, dim=0)

            top_indices_sorted = torch.tensor(top_indices_sorted, dtype=torch.long)
            positions = torch.hstack([torch.where(all_indices == index)[0] for index in top_indices_sorted])

            all_images = all_not_aug_inputs[positions]
            all_labels = all_labels[positions]

            updated_std_of_means_by_class = {self.reverse_mapping[k]: 1 for k, _ in std_of_means_by_class.items()}
            self.class_portion.append(updated_std_of_means_by_class)
            updated_std_of_means_by_task = {k: 1 for k, v in std_of_means_by_task.items()}
            dist_task_before = distribute_samples(updated_std_of_means_by_task, self.args.buffer_size)

            if self.task > 1:
                dist_task = adjust_values_integer_include_all(dist_task_before.copy(), self.dist_task_prev)
            else:
                dist_task = dist_task_before

            dist_class = [distribute_samples(self.class_portion[i], dist_task[i]) for i in range(self.task)]
            self.dist_task_prev = dist_task

            dist = dist_class.pop()
            dist_last = dist.copy()
            dist = {self.mapping[k]: v for k, v in dist.items()}

            counter_class = [0 for _ in range(len(self.unique_classes))]
            condition = [dist[k] for k in range(len(dist))]

            check_bound = self.n_sample_per_task // len(self.unique_classes)
            for i in range(len(condition)):
                if condition[i] > check_bound:
                    condition = distribute_excess(condition, check_bound)
                    break

            images_list_ = []
            labels_list_ = []
            for i in range(all_labels.shape[0]):
                if counter_class[self.mapping[all_labels[i].item()]] < condition[self.mapping[all_labels[i].item()]]:
                    counter_class[self.mapping[all_labels[i].item()]] += 1
                    labels_list_.append(all_labels[i])
                    images_list_.append(all_images[i])
                if counter_class == condition:
                    break

            all_images_ = torch.stack(images_list_).to(self.device)
            all_labels_ = torch.stack(labels_list_).to(self.device)

            counter_manage = [{k: 0 for k, __ in dist_class[i].items()} for i in range(self.task - 1)]
            dist_class_merged = {}
            counter_manage_merged = {}
            dist_class_merged_prev = {}

            for d in dist_class:
                dist_class_merged.update(d)
            for f in counter_manage:
                counter_manage_merged.update(f)
            if self.task > 1:
                dist_class_merged_prev = self.dist_class_prev
                class_key = list(dist_class_merged.keys())
                temp_key = -1
                for k, value in dist_class_merged.items():
                    temp_key += 1
                    if value > dist_class_merged_prev[k]:
                        temp = value - dist_class_merged_prev[k]
                        dist_class_merged[k] -= temp
                        for hh in range(temp):
                            dist_class_merged[class_key[temp_key + hh + 1]] += 1

            self.dist_class_prev = dist_class_merged.copy()
            self.dist_class_prev.update(dist_last)

            if not self.buffer.is_empty():
                images_store = []
                labels_store = []
                for i in range(len(self.buffer)):
                    if counter_manage_merged[self.buffer.labels[i].item()] < dist_class_merged[self.buffer.labels[i].item()]:
                        counter_manage_merged[self.buffer.labels[i].item()] += 1
                        labels_store.append(self.buffer.labels[i])
                        images_store.append(self.buffer.examples[i])
                    if counter_manage_merged == dist_class_merged:
                        break
                images_store_ = torch.stack(images_store).to(self.device)
                labels_store_ = torch.stack(labels_store).to(self.device)
                all_images_ = torch.cat((images_store_, all_images_))
                all_labels_ = torch.cat((labels_store_, all_labels_))

            if not hasattr(self.buffer, 'examples'):
                self.buffer.init_tensors(all_images_, all_labels_, None, None)

            self.buffer.num_seen_examples += self.n_sample_per_task
            self.buffer.labels = all_labels_
            self.buffer.examples = all_images_

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        batch_x, batch_y = inputs, labels
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_combine = batch_x
        batch_y_combine = batch_y

        self.opt.zero_grad()

        # === DIAG: decide whether to run the eval forward pass on this step ===
        # Original CGR: only during the first E epochs (for variance computation).
        # With diag logging: for ALL epochs of task 1, so we get target confidence,
        # margin, correctness, and per-sample loss from the SAME eval-mode pass at
        # every epoch (needed for forgetting events and the diagnostic table).
        run_eval_pass = self.epoch < self.args.E
        if self._diag_active() and self.epoch < self.args.n_epochs:
            run_eval_pass = True
        # === END DIAG ===

        if run_eval_pass:
            targets = torch.tensor([self.mapping[val.item()] for val in labels]).to(self.device)
            confidence_batch = []
            self.net.eval()
            with torch.no_grad():
                cgr_logits = self.net(not_aug_inputs)
                soft_ = nn.functional.softmax(cgr_logits, dim=1)
                # Existing: per-sample target confidence into self.confidence_by_sample
                for i in range(targets.shape[0]):
                    confidence_batch.append(soft_[i, labels[i]].item())
                conf_tensor = torch.tensor(confidence_batch)
                self.confidence_by_sample[self.epoch, index_] = conf_tensor

                # === DIAG: record all per-sample per-epoch quantities from same eval pass ===
                # Recorded DURING THIS SAMPLE'S DIAGNOSTIC PASS in this epoch (before
                # the batch's SGD update) — NOT a common end-of-epoch checkpoint.
                if self._diag_active():
                    labels_dev = labels.to(self.device).long()

                    # ---- Probability-space margin quantities ----
                    target_prob = soft_.gather(1, labels_dev.unsqueeze(1)).squeeze(1)          # p_y
                    # target-based signed prob margin: p_y - max_{k!=y} p_k
                    soft_other_tgt = soft_.clone()
                    soft_other_tgt.scatter_(1, labels_dev.unsqueeze(1), float('-inf'))
                    max_other_prob_tgt = soft_other_tgt.max(dim=1).values
                    margin = (target_prob - max_other_prob_tgt).cpu()
                    # predicted-based (top-two) prob gap: p_yhat - max_{k!=yhat} p_k (>=0)
                    y_hat = soft_.argmax(dim=1)
                    pred_prob = soft_.gather(1, y_hat.unsqueeze(1)).squeeze(1)
                    soft_other_pred = soft_.clone()
                    soft_other_pred.scatter_(1, y_hat.unsqueeze(1), float('-inf'))
                    max_other_prob_pred = soft_other_pred.max(dim=1).values
                    margin_pred = (pred_prob - max_other_prob_pred).cpu()

                    # ---- Correctness (argmax == label) ----
                    correct = (y_hat == labels_dev).cpu()

                    # ---- Per-sample cross-entropy loss ----
                    per_sample_loss = F.cross_entropy(cgr_logits, labels_dev,
                                                     reduction='none').cpu()

                    # ---- Logit-space margin quantities ----
                    # target-based signed logit margin: z_y - max_{k!=y} z_k
                    target_logits = cgr_logits.gather(1, labels_dev.unsqueeze(1)).squeeze(1)
                    logit_other_tgt = cgr_logits.clone()
                    logit_other_tgt.scatter_(1, labels_dev.unsqueeze(1), float('-inf'))
                    max_other_logit_tgt = logit_other_tgt.max(dim=1).values
                    logit_margin = (target_logits - max_other_logit_tgt).cpu()
                    # predicted-based (top-two) logit gap: z_yhat - max_{k!=yhat} z_k (>=0)
                    pred_logits_val = cgr_logits.gather(1, y_hat.unsqueeze(1)).squeeze(1)
                    logit_other_pred = cgr_logits.clone()
                    logit_other_pred.scatter_(1, y_hat.unsqueeze(1), float('-inf'))
                    max_other_logit_pred = logit_other_pred.max(dim=1).values
                    logit_margin_pred = (pred_logits_val - max_other_logit_pred).cpu()

                    # ---- Nearest pairwise |target - k| gap, both logit and probability space
                    #      (target-based only; the predicted-based analog would equal the
                    #      top-two gap `*_margin_pred` because y_hat is argmax, so |z_yhat - z_k|
                    #      = z_yhat - z_k for all k != y_hat, and min = z_yhat - max_{k!=yhat} z_k
                    #      = logit_margin_pred). Target-based versions differ from |scalar margin|
                    #      for misclassified samples: |min|(diffs) != min |diffs|. ----
                    # target-based (logit): min_{k!=y} |z_y - z_k|
                    z_diff_tgt = target_logits.unsqueeze(1) - cgr_logits  # (B, C)
                    abs_diff_tgt = z_diff_tgt.abs()
                    mask_tgt = torch.zeros_like(abs_diff_tgt, dtype=torch.bool)
                    mask_tgt.scatter_(1, labels_dev.unsqueeze(1), True)
                    nearest_pair_logit_gap = abs_diff_tgt.masked_fill(
                        mask_tgt, float('inf')).min(dim=1).values.cpu()
                    # target-based (probability space): min_{k!=y} |p_y - p_k|
                    # Reuses `target_prob` (=p_y) and `soft_` (softmax probs) computed above.
                    p_diff_tgt = target_prob.unsqueeze(1) - soft_  # (B, C), p_y - p_k
                    abs_p_diff_tgt = p_diff_tgt.abs()
                    mask_p_tgt = torch.zeros_like(abs_p_diff_tgt, dtype=torch.bool)
                    mask_p_tgt.scatter_(1, labels_dev.unsqueeze(1), True)
                    nearest_pair_prob_gap = abs_p_diff_tgt.masked_fill(
                        mask_p_tgt, float('inf')).min(dim=1).values.cpu()

                    # ---- Feature-space distances (Concern 2) ----
                    d_target_signed, d_target_absmin, d_pred = \
                        self._diag_feat_distances(cgr_logits, labels_dev)
                    d_target_signed = d_target_signed.cpu()
                    d_target_absmin = d_target_absmin.cpu()
                    d_pred = d_pred.cpu()

                    # ---- Scatter all quantities into the tensors ----
                    if torch.is_tensor(index_):
                        idx_cpu = index_.detach().cpu().long()
                    else:
                        idx_cpu = torch.as_tensor(index_, dtype=torch.long)
                    e = self.epoch
                    self.diag_margin[e, idx_cpu] = margin
                    self.diag_margin_pred[e, idx_cpu] = margin_pred
                    self.diag_correct[e, idx_cpu] = correct
                    self.diag_loss[e, idx_cpu] = per_sample_loss
                    self.diag_labels[idx_cpu] = labels.detach().cpu().long()
                    self.diag_logit_margin[e, idx_cpu] = logit_margin
                    self.diag_logit_margin_pred[e, idx_cpu] = logit_margin_pred
                    self.diag_nearest_pair_logit_gap[e, idx_cpu] = nearest_pair_logit_gap
                    self.diag_nearest_pair_probability_gap[e, idx_cpu] = nearest_pair_prob_gap
                    self.diag_feat_dist_target[e, idx_cpu] = d_target_signed
                    self.diag_feat_dist_target_absmin[e, idx_cpu] = d_target_absmin
                    self.diag_feat_dist_pred[e, idx_cpu] = d_pred
                # === END DIAG ===
            self.net.train()

        # SGD forward + backward (unchanged)
        if self.buffer.is_empty():
            logits = self.net(batch_x_combine)
            novel_loss = self.loss(logits, batch_y_combine)
        else:
            mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            mem_x = mem_x.to(self.device)
            mem_y = mem_y.to(self.device)
            mem_x_combine = mem_x
            mem_y_combine = mem_y
            combined_inputs = torch.cat([mem_x_combine, batch_x_combine])
            combined_labels = torch.cat((mem_y_combine, batch_y_combine))
            combined_logits = self.net(combined_inputs)
            novel_loss = self.loss(combined_logits, combined_labels)

        novel_loss.backward()
        self.opt.step()

        return novel_loss.item()
