# ER + GroupDRO-task baseline.
#
# A GroupDRO-style robust-optimization baseline adapted to rehearsal-based,
# class-incremental continual learning. Standard GroupDRO assumes group/domain
# labels in a static multi-environment dataset; here we adapt it by treating
# *task identity* as the group label, since tasks are the natural "environments"
# in class-incremental CL.
#
# Because tasks have disjoint class sets with N_CLASSES_PER_TASK classes each,
# the task (group) id of any sample is recovered directly from its label:
#       group(y) = y // n_classes_per_task
# This assigns group ids to both current-task and replay (buffer) samples
# without modifying the buffer to store task labels.
#
# Online GroupDRO update (Sagawa et al., 2020):
#   - maintain a weight q_g over groups (tasks)
#   - per step, compute mean loss L_g for each group PRESENT in the batch on the
#     combined current + replay batch
#   - multiplicative weight update: q_g <- q_g * exp(eta * L_g), then renormalize
#   - backprop the q-weighted sum of present-group losses
#
# Everything else matches plain ER: reservoir buffer, standard softmax CE,
# dot-product logits. No adaptive reweighting and no correctness-guided buffer.
#
# IMPLEMENTATION NOTE: This model is fully self-initializing inside observe(),
# so it does NOT require the training loop to call begin_train/begin_task (it is
# not in those hardcoded whitelists in utils/training.py). Total class count is
# read from the backbone (self.net.num_classes); the per-task class count is
# supplied via --n_classes_per_task (10 for CIFAR-100; 20 for Mini-/Tiny-ImageNet).
#
# FRAMING: report as "GroupDRO-style robust weighting with task identity as the
# group label", NOT as GroupDRO proper.

import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER with GroupDRO-style robust task weighting.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--gdro_eta', type=float, default=0.01,
                        help='GroupDRO group-weight step size (eta).')
    parser.add_argument('--n_classes_per_task', type=int, required=True,
                        help='Number of classes per task (e.g. 10 for Split CIFAR-100, '
                             '20 for Split Mini-/Tiny-ImageNet). Used to map labels to '
                             'task groups for GroupDRO.')
    return parser


# NOTE on the class name: Mammoth's registry (models/__init__.py) matches the
# file name to a class by stripping underscores and comparing lowercased names,
# i.e. it looks for a class whose name.lower() == 'ergroupdrotask'. The class
# must therefore be named 'Ergroupdrotask' (NOT 'ErGroupDRO'), otherwise model
# discovery raises a KeyError. The NAME attribute below is what --model expects.
class Ergroupdrotask(ContinualModel):
    NAME = 'er_groupdro_task'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Ergroupdrotask, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.n_classes_per_task = int(self.args.n_classes_per_task)
        # Total classes from the backbone classifier; q has one weight per task.
        self.total_classes = int(self.net.num_classes)
        self.n_tasks = self.total_classes // self.n_classes_per_task
        # Uniform initial group (task) weights.
        self.q = torch.ones(self.n_tasks, device=self.device) / self.n_tasks

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        # Combine current + replay batches (buffer samples transformed on retrieval).
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            cat_inputs = torch.cat((inputs, buf_inputs))
            cat_labels = torch.cat((labels, buf_labels))
        else:
            cat_inputs = inputs
            cat_labels = labels

        cat_inputs = cat_inputs.to(self.device)
        cat_labels = cat_labels.to(self.device)

        outputs = self.net(cat_inputs)
        # Per-sample cross-entropy so it can be grouped by task.
        per_sample_loss = F.cross_entropy(outputs, cat_labels, reduction='none')

        # Group (task) id for each sample from its label (disjoint class sets).
        group_ids = torch.div(cat_labels, self.n_classes_per_task,
                              rounding_mode='floor').long()
        present_groups = torch.unique(group_ids)

        # Mean loss per present group.
        group_losses = torch.stack([per_sample_loss[group_ids == g].mean()
                                    for g in present_groups])

        # --- GroupDRO multiplicative weight update (no grad) ---
        with torch.no_grad():
            updated = self.q[present_groups] * torch.exp(
                self.args.gdro_eta * group_losses.detach())
            self.q[present_groups] = updated
            self.q = self.q / self.q.sum()  # renormalize over all groups

        # Robust (q-weighted) loss over present groups; renormalize over the
        # present mass so the loss scale is stable when only some tasks appear.
        w_present = self.q[present_groups]
        w_present = w_present / (w_present.sum() + 1e-12)
        loss = (w_present * group_losses).sum()

        loss.backward()
        self.opt.step()

        # Reservoir update with the (non-augmented) current-task samples.
        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return loss.item()
