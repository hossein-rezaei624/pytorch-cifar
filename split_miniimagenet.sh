# ER + factorization regularizer baseline.
#
# Plain Experience Replay (ER) plus ONLY the causal-factorization objective from
# CIRL (Lv et al., CVPR 2022, "Causality Inspired Representation Learning for
# Domain Generalization", arXiv:2203.14237), isolated as a clean ladder row.
#
# This represents the *representation-learning* family of traditional OOD
# generalization methods (invariance + feature decorrelation), complementing the
# weighting family (ER-LAS) and the data-centric/augmentation family (er_aug,
# er_mixup).
#
# IMPORTANT SCOPING / NAMING:
#   - CIRL has three modules: causal intervention (Fourier amplitude mixing
#     across domains), causal factorization (the Barlow-Twins-style
#     cross-correlation loss used here), and an adversarial mask module.
#   - This baseline implements ONLY the factorization module. CIRL's intervention
#     and mask modules assume multiple jointly-observed source domains, which do
#     NOT exist in class-incremental CL (tasks are sequential, disjoint-class).
#   - The factorization objective itself needs no domain labels: it only enforces
#     that representations of an image and its augmented view have a near-identity
#     cross-correlation matrix (diagonal -> 1, off-diagonal -> 0), i.e. invariant
#     to the augmentation and dimension-wise decorrelated. We therefore adapt this
#     single, domain-label-free component and apply it to augmentation pairs.
#   - Report this row as "ER + factorization regularizer (adapted from CIRL's
#     factorization module)", NOT as "CIRL".
#
# Everything else is plain ER: standard softmax cross-entropy on the standard
# (dot-product) classifier logits, reservoir buffer, no reweighting, no proxy
# head, no correctness-guided buffer selection.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.pcr_transforms_aug import transforms_aug


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER with CIRL-style factorization regularizer.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--fact_lambda', type=float, default=1.0,
                        help='Weight of the factorization regularizer added to the CE loss.')
    return parser


def off_diagonal(x):
    # Flattened view of the off-diagonal elements of a square matrix.
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # Barlow-Twins-style cross-correlation between two views' representations.
    # Drives the diagonal of the (normalized) cross-correlation matrix toward 1
    # (invariance) and the off-diagonal toward 0 (dimension-wise decorrelation).
    # Matches CIRL's factorization module (off-diagonal weight 0.005).
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    return on_diag + 0.005 * off_diag


class ErFact(ContinualModel):
    NAME = 'er_fact'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErFact, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def _orig_aug_features(self, x):
        """Return (logits_on_combined, feats_orig, feats_aug) for a batch x.

        Builds a strongly augmented view of x, runs original and augmented views
        through the standard backbone (returnt='all' gives logits + features),
        and splits the features back into the original and augmented halves.
        """
        x = x.to(self.device)
        x_aug = torch.stack([transforms_aug[self.args.dataset](x[i].cpu())
                             for i in range(x.size(0))]).to(self.device)
        combined = torch.cat((x, x_aug))
        logits, feats = self.net(combined, returnt='all')
        feats_orig, feats_aug = torch.split(feats, x.size(0))
        return logits, feats_orig, feats_aug

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # --- Current task: original + augmented view ---
        cur_logits, cur_feat_o, cur_feat_a = self._orig_aug_features(inputs)
        cur_labels2 = torch.cat((labels, labels))  # labels for [orig; aug]

        # Standard CE on the combined (orig + aug) current batch.
        ce_loss = self.loss(cur_logits, cur_labels2)

        # Factorization regularizer is active from task 1 (current-view pair).
        fact = factorization_loss(cur_feat_o, cur_feat_a)

        # --- Buffer: original + augmented view ---
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_inputs = buf_inputs.to(self.device)
            buf_labels = buf_labels.to(self.device)

            buf_logits, buf_feat_o, buf_feat_a = self._orig_aug_features(buf_inputs)
            buf_labels2 = torch.cat((buf_labels, buf_labels))
            ce_loss = ce_loss + self.loss(buf_logits, buf_labels2)

            # Combine current + buffer view-pairs for the factorization term.
            com_feat_o = torch.cat((cur_feat_o, buf_feat_o))
            com_feat_a = torch.cat((cur_feat_a, buf_feat_a))
            fact = factorization_loss(com_feat_o, com_feat_a)

        loss = ce_loss + self.args.fact_lambda * fact

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Reservoir update with the (non-augmented) current-task samples.
        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return loss.item()
