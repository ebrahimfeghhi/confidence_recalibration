#!/usr/bin/env python
"""Does this arm just need more epochs? A 30-epoch arm already answered that.

WHY THIS FIGURE. "It works, it is only slow" is a claim about the SLOPE of held-out selected
DockQ, and it is testable two ways: fit the slope on the arm in flight, and look at an arm that
already ran three times longer. Both are drawn here on one axis so the extrapolation and the
actual outcome sit side by side.

THE CROPPED ARM (o2qczq -> t4jtjr) is the matched control, not a different problem: same pool,
same 92 held-out interfaces, same 150 candidates, same shipped reference, evaluated UNCROPPED at
every point shown. It reached +0.0048 at epoch 10 -- where the uncropped arm is now -- and twenty
further epochs ended at +0.0002.

    python -I run_logging/plot_longer_doesnt_help.py
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNC_EP = list(range(1, 11))
UNC = [-0.0001, 0.0014, 0.0026, 0.0057, 0.0050, 0.0024, 0.0016, 0.0028, 0.0048, 0.0036]
CRP_EP = [5, 10, 15, 20, 25, 30]
CRP = [0.0023, 0.0048, -0.0056, 0.0017, -0.0069, 0.0002]

n = len(UNC_EP); mx = sum(UNC_EP) / n; my = sum(UNC) / n
b = sum((x - mx) * (y - my) for x, y in zip(UNC_EP, UNC)) / sum((x - mx) ** 2 for x in UNC_EP)
a = my - b * mx

fig, ax = plt.subplots(figsize=(10.5, 6.0))
ax.axhline(0, color="#555", lw=1.8, ls="--", zorder=2)
ax.text(30.4, 0.0002, "shipped", fontsize=11, color="#555", va="center")

xs = list(range(1, 32))
ax.plot(xs, [a + b * x for x in xs], ":", color="#1b6ca8", lw=2.0, alpha=.8, zorder=3,
        label=f"fitted trend, uncropped arm ({b:+.5f}/epoch, t=+1.38)")
ax.plot(UNC_EP, UNC, "o-", color="#1b6ca8", lw=2.8, ms=8, zorder=5,
        label="UNCROPPED arm (in flight, 10 epochs)")
ax.plot(CRP_EP, CRP, "s-", color="#c0392b", lw=2.8, ms=8, zorder=5,
        label="CROPPED arm, 30 epochs -- the same question, already run")

ax.annotate("both arms sit here\nat epoch 10", xy=(10, 0.0048), xytext=(12.6, 0.0092),
            fontsize=11.5, weight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
ax.annotate("20 more epochs\nended HERE", xy=(30, 0.0002), xytext=(26.5, -0.0092),
            fontsize=11.5, weight="bold", color="#c0392b", ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#c0392b"))

ax.set_xlabel("epoch", fontsize=13)
ax.set_ylabel("held-out selected DockQ, ours minus shipped", fontsize=13)
ax.set_title("\"It just needs longer\" -- the 30-epoch arm already tested that",
             fontsize=14.5, weight="bold", pad=12)
ax.set_xlim(0, 32.5)
ax.grid(alpha=.25)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, loc="upper left", framealpha=.95)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("figures/longer_doesnt_help.png", dpi=150)
print("  -> figures/longer_doesnt_help.png")
