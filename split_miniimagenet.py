"""
Helper for injecting symmetric label noise into CL benchmarks (CSReL-CL-Prv).

Used for the Concern-2 label-noise robustness experiment in the CGR rebuttal.

Convention: 'within-task' symmetric noise — each corrupted label is replaced
by a uniformly random *incorrect* class from the *same task*. This keeps the
corruption consistent with the CIL task structure: the model never sees a
sample whose label is outside the current task's class set.

For Split CIFAR-100 with contiguous class ranges per task (task 0 = classes
0-9, task 1 = 10-19, ...), within-task noise means: a sample originally
labeled with class 5 (task 0) can be relabeled as any other class in {0..9}.

Usage in dataset/idataset.py, inside SplitCifar100.__init__,
right after `self.Y_train = np.array(self.train_dataset.targets)`:

    from dataset.noisy_labels import inject_symmetric_label_noise
    inject_symmetric_label_noise(
        self.train_dataset,
        noise_rate=label_noise,
        n_classes_total=100,
        n_classes_per_task=10,
        within_task=True,
        seed=seed,
    )
    # Copy the same noisy labels to the no-augmentation dataset
    self.train_dataset_wo_augment.targets = list(self.train_dataset.targets)
"""
import numpy as np


def inject_symmetric_label_noise(
    dataset,
    noise_rate: float,
    n_classes_total: int,
    n_classes_per_task: int = None,
    within_task: bool = True,
    seed: int = 0,
    verbose: bool = True,
):
    """Replace each label in `dataset.targets` with a random incorrect class
    with probability `noise_rate`. Modifies `dataset.targets` in place.

    Args:
        dataset:            object with a `.targets` attribute (list or np.ndarray)
        noise_rate:         float in [0, 1]
        n_classes_total:    total number of classes
        n_classes_per_task: classes per task (required if within_task=True)
        within_task:        if True, replacement label is drawn from same task only
        seed:               RNG seed for reproducibility (use args.seed)
        verbose:            print a summary line
    """
    if noise_rate is None or noise_rate <= 0:
        return

    rng = np.random.RandomState(seed)
    targets = np.asarray(dataset.targets, dtype=np.int64).copy()
    n = len(targets)

    corrupt_mask = rng.rand(n) < noise_rate
    n_corrupt = int(corrupt_mask.sum())
    if n_corrupt == 0:
        return

    if within_task:
        assert n_classes_per_task is not None, \
            "Need n_classes_per_task for within-task noise."
        for i in np.where(corrupt_mask)[0]:
            orig = int(targets[i])
            task = orig // n_classes_per_task
            cls_lo = task * n_classes_per_task
            cls_hi = cls_lo + n_classes_per_task
            new_label = orig
            while new_label == orig:
                new_label = rng.randint(cls_lo, cls_hi)
            targets[i] = new_label
    else:
        for i in np.where(corrupt_mask)[0]:
            orig = int(targets[i])
            new_label = orig
            while new_label == orig:
                new_label = rng.randint(0, n_classes_total)
            targets[i] = new_label

    # Write back, preserving original container type
    if isinstance(dataset.targets, list):
        dataset.targets = targets.tolist()
    else:
        dataset.targets = targets

    if verbose:
        print(f"[noise] Corrupted {n_corrupt}/{n} labels "
              f"(rate={noise_rate}, within_task={within_task}, seed={seed})")
