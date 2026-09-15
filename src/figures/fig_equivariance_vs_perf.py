"""
What does measured equivariance cost on the upright task?

The old equivariant_vs_test_loss_* figures answered a related question one config at
a time (396 PDFs, never two configs side by side). The question is comparative, so
this is one scatter over all 33 matched pairs.

Which test loss matters is the whole point here. Averaged over all probe angles,
augmentation helps in 33/33 configs -- but that is close to tautological, since the
baseline is being scored on rotated inputs it never trained on. The informative
axis is loss on the *upright* (angle 0) test set, where augmentation makes 31/33
configs worse. So the sweep's own data shows equivariance being bought, not
received free.

Test loss is not comparable across tasks (cross-entropy vs. MSE vs. flow error), so
y is the relative change (aug - base) / base, on a symlog axis because it spans
zero to ~1.9e3. Faceted by task family rather than colored by task: the all-pairs
CVD floor caps a categorical scatter at three hues, and there are six tasks.

Run from src/:  python figures/fig_equivariance_vs_perf.py
"""
import matplotlib.pyplot as plt
import numpy as np

import style as st

FAMILY = {
    "classification": "Grid / image tasks", "colorization": "Grid / image tasks",
    "fluid_flow": "Grid / image tasks", "stress_prediction": "Grid / image tasks",
    "fluid_flow_particles": "Particle tasks",
    "stress_prediction_particles": "Particle tasks",
}


def main(args):
    st.apply()
    cka = st.at_final_epoch(st.apply_filters(st.load("cka_by_epoch"), args))
    perf = st.apply_filters(st.load("perf_by_epoch"), args)
    perf = perf[perf.epoch == st.final_epochs(perf)]

    deep = cka[cka.layer_idx == cka.groupby(st.ID, observed=True)["layer_idx"].transform("max")]
    c = deep.groupby(st.ID, observed=True)["rbf_cka"].mean().unstack("augmented")
    clean = (perf[perf.angle == 0].groupby(st.ID, observed=True)["test_loss"]
                 .mean().unstack("augmented"))

    j = c.join(clean, lsuffix="_cka", rsuffix="_loss").dropna().reset_index()
    j["d_cka"] = j["True_cka"] - j["False_cka"]
    j["d_loss"] = (j["True_loss"] - j["False_loss"]) / j["False_loss"]
    j["family"] = j.task.map(FAMILY)

    fams = ["Grid / image tasks", "Particle tasks"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True,
                             constrained_layout=True)
    for ax, fam in zip(axes, fams):
        sub = j[j.family == fam]
        ax.axhline(0, color=st.AXIS, lw=1.0, zorder=1)
        ax.axvline(0, color=st.AXIS, lw=1.0, zorder=1)
        ax.scatter(sub.d_cka, sub.d_loss, s=54, color=st.AUG, alpha=0.85,
                   edgecolor=st.SURFACE, lw=1.0, zorder=3)
        # Selective labels only: a dense low-cost cluster (the colorization
        # configs) sits at ~(0.02, 0.02) and labelling every point there is
        # illegible. Label the extremes, which are the ones worth naming.
        notable = sub[(sub.d_loss > 0.3) | (sub.d_cka.abs() > 0.1)
                      | (sub.d_loss <= 0.001)]
        for r in notable.itertuples():
            ax.annotate(f"{r.model}/{r.dataset}", (r.d_cka, r.d_loss),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=6.4, color=st.INK_2)
        n_hidden = len(sub) - len(notable)
        if n_hidden:
            ax.annotate(f"+{n_hidden} low-cost configs unlabelled",
                        (0.02, 0.02), xycoords="axes fraction",
                        fontsize=6.8, color=st.MUTED)
        rho = np.corrcoef(sub.d_cka, np.log10(sub.d_loss + 1e-3))[0, 1]
        ax.set_title(f"{fam}   (n={len(sub)},  r={rho:+.2f} vs. log cost)",
                     fontsize=10, color=st.INK, loc="left")
        ax.set_xlabel("$\\Delta$ measured equivariance\n"
                      "(RBF CKA at deepest layer, augmented $-$ baseline)", fontsize=9)
    axes[0].set_yscale("symlog", linthresh=0.01, linscale=0.5)
    axes[0].set_ylabel("relative $\\Delta$ loss on the UPRIGHT test set\n"
                       "(augmented $-$ baseline) / baseline", fontsize=9)
    n_worse = int((j.d_loss > 0).sum())
    fig.suptitle(f"Equivariance is bought, not free: augmentation raises upright-task "
                 f"loss in {n_worse} of {len(j)} pairs\n"
                 "one point per matched pair · final epoch · mean over 10 seeds · "
                 "above the grey line = worse on upright inputs",
                 fontsize=11.5, color=st.INK)
    st.save(fig, "equivariance_vs_performance" + st.suffix(args))
    print(f"  worse on upright: {n_worse}/{len(j)}   median relative cost "
          f"{j.d_loss.median():+.3f}")
    print("  largest costs:")
    for r in j.nlargest(3, "d_loss").itertuples():
        print(f"    {r.task}/{r.model}/{r.dataset}: {r.d_loss:+.1f}x  (dCKA {r.d_cka:+.3f})")


if __name__ == "__main__":
    main(st.common_parser(__doc__).parse_args())
