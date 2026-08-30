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
        """Compute per-sample signed feature-space distance to the nearest
        boundary of (i) the target-class region and (ii) the predicted-class
        region. Uses only the classifier head's weight matrix; biases cancel
        because z = W*phi + b already includes bias.

        Args:
            logits: (B, C) raw logits from the eval-mode forward pass
            labels_dev: (B,) target class indices on the same device

        Returns:
            d_target: (B,) signed distance for target class (may be negative
                      if some competing class beats target)
            d_pred:   (B,) signed distance for argmax class (always >= 0)
        """
        head = self._diag_get_classifier_head()
        W = head.weight.detach()  # (C, d_feat)
        # Pairwise weight-vector distances ||w_i - w_j||_2 (C, C)
        # Small (100x100 for CIFAR-100) so cost is negligible.
        W_dist = torch.cdist(W.unsqueeze(0), W.unsqueeze(0), p=2).squeeze(0)

        B, C = logits.shape

        # -- d_target: (z_y - z_k) / ||w_y - w_k||, min over k != y --
        target_logits = logits.gather(1, labels_dev.unsqueeze(1)).squeeze(1)   # (B,)
        z_diff_t = target_logits.unsqueeze(1) - logits                          # (B, C)
        w_diff_t = W_dist[labels_dev]                                            # (B, C)
        # avoid div-by-zero at k=y (0/0); we'll mask that column afterwards
        ratio_t = z_diff_t / w_diff_t.clamp(min=1e-12)
        mask_t = torch.zeros_like(ratio_t, dtype=torch.bool)
        mask_t.scatter_(1, labels_dev.unsqueeze(1), True)
        ratio_t = ratio_t.masked_fill(mask_t, float('inf'))
        d_target = ratio_t.min(dim=1).values                                     # (B,)

        # -- d_pred: same but using argmax class --
        y_hat = logits.argmax(dim=1)                                             # (B,)
        pred_logits = logits.gather(1, y_hat.unsqueeze(1)).squeeze(1)            # (B,)
        z_diff_p = pred_logits.unsqueeze(1) - logits                             # (B, C)
        w_diff_p = W_dist[y_hat]                                                  # (B, C)
        ratio_p = z_diff_p / w_diff_p.clamp(min=1e-12)
        mask_p = torch.zeros_like(ratio_p, dtype=torch.bool)
        mask_p.scatter_(1, y_hat.unsqueeze(1), True)
        ratio_p = ratio_p.masked_fill(mask_p, float('inf'))
        d_pred = ratio_p.min(dim=1).values                                       # (B,)

        return d_target, d_pred
    # === END DIAG ===
