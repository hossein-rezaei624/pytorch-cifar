\subsection{From Gradient Imbalance to OOD Degradation: A Margin-Based View}

The gradient-imbalance analysis in Section 3.2 describes a CL-induced optimization mechanism that affects both i.i.d. and OOD performance, but does not by itself require an OOD assumption. We now make the connection to OOD degradation explicit through a margin-based argument: gradient-induced margin compression for past classes affects clean and corrupted inputs differently, and therefore provides a natural mechanism for the disproportionately large OOD drops observed in Figure 1 and Table 1.

For a sample $(x,y)$ with feature $z=f_\theta(x)$ and classifier weights $\{w_c\}_{c\in C_{1:t}}$, define the classification margin as
\[
m(x)
=
w_y^\top f_\theta(x)
-
\max_{c\neq y} w_c^\top f_\theta(x).
\]
The prediction at $x$ equals $y$ whenever $m(x)>0$, so clean i.i.d. correctness depends on the sign of $m(x)$.

For a label-preserving corruption $q$, as used in our OOD evaluation in Section 5.1, suppose the induced feature shift is bounded by
\[
\|f_\theta(q(x))-f_\theta(x)\|_2
\leq
L_\theta \delta,
\]
where $L_\theta$ captures the local sensitivity of the representation and $\delta$ measures the corruption magnitude. Let $\Delta=f_\theta(q(x))-f_\theta(x)$ and let
\[
c^\star=\arg\max_{c\neq y} w_c^\top f_\theta(q(x)).
\]
Then
\[
m(q(x))
=
(w_y-w_{c^\star})^\top f_\theta(x)
+
(w_y-w_{c^\star})^\top \Delta
\geq
m(x)
-
\max_{c\neq y}\|w_y-w_c\|_2 L_\theta\delta,
\]
where the inequality follows from the definition of $m(x)$ and the Cauchy--Schwarz inequality. Therefore, a sufficient condition for preserving the prediction under corruption is
\[
m(x)
>
\max_{c\neq y}\|w_y-w_c\|_2 L_\theta\delta.
\]

This condition makes the i.i.d./OOD asymmetry explicit. Preserving clean accuracy at $x$ requires only $m(x)>0$, whereas preserving accuracy under label-preserving corruption requires the margin to be large enough to absorb the feature displacement induced by $q$. Two models with comparable i.i.d. accuracy can therefore differ sharply in OOD accuracy if one has more compressed margins or a more corruption-sensitive representation.

The analysis of Section 3.2 explains why such margin compression naturally arises for past classes in rehearsal-based CL. Past-class weights receive only sparse positive updates through the limited buffer, while they repeatedly appear as non-target classes during current-task training and accumulate the repulsive gradients identified in the gradient-polarity analysis. At the same time, the encoder is primarily driven by current-task data, producing feature drift for past-task samples. These two effects act jointly on the margin for past-class samples: the target logit $w_y^\top f_\theta(x_{\mathrm{old}})$ can decrease as training proceeds, while competing logits $w_c^\top f_\theta(x_{\mathrm{old}})$ for $c\neq y$ can remain high or increase. The margin $m(x_{\mathrm{old}})$ can therefore contract over the task sequence. Under i.i.d. evaluation, this contraction affects 0-1 accuracy only once the margin crosses zero. Under OOD evaluation, however, even positive but small margins can fail the sufficient condition above, leading to a disproportionately larger drop in OOD accuracy.

This view also clarifies how AA-RR improves OOD robustness in addition to mitigating standard forgetting. The adaptive reweighting mechanism in Section 4.1 counteracts gradient imbalance and helps preserve past-class margins. The augmentation pipeline in Section 4.2 encourages representations that are less sensitive to label-preserving perturbations, corresponding to a smaller effective $L_\theta$ in the condition above. The correctness-guided buffer management in Section 4.3 preserves balanced and consistently learned replay examples, stabilizing the replay signal for past classes and indirectly helping maintain their margins throughout the task sequence.
