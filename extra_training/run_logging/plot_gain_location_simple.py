#!/usr/bin/env python
"""ONE panel, the only comparison that decides the arm: ordering everywhere vs ordering at the top.

WHY A SEPARATE SCRIPT. rel_heldout and rel_arm each carry six panels because each was built to
answer a different question, and the result is that the finding they share is invisible. This
draws two paired lines and nothing else.

Shipped is flat by construction -- its ipTM does not change between epochs -- so it is drawn as a
dashed reference at each height rather than as a series.

    python -I run_logging/plot_gain_location_simple.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EP = list(range(1, 11))
RHO_ALL = [0.2074, 0.2001, 0.2225, 0.2230, 0.2343, 0.2358, 0.2334, 0.2370, 0.2321, 0.2426]
RHO_TOP = [-0.1541, -0.1250, -0.1129, -0.1381, -0.1069, -0.1426, -0.1133, -0.1191, -0.1356, -0.1287]
SHIP_ALL, SHIP_TOP = 0.1709, -0.0607

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.axhline(0, color="#999", lw=1.0, zorder=1)

ax.plot(EP, RHO_ALL, "o-", color="#1b6ca8", lw=2.6, ms=7, zorder=4,
        label="OURS, all 150 poses")
ax.axhline(SHIP_ALL, ls="--", color="#1b6ca8", lw=2.0, alpha=.65, zorder=3,
           label="shipped, all 150 poses")
ax.plot(EP, RHO_TOP, "s-", color="#c0392b", lw=2.6, ms=7, zorder=4,
        label="OURS, top 10 poses")
ax.axhline(SHIP_TOP, ls="--", color="#c0392b", lw=2.0, alpha=.65, zorder=3,
           label="shipped, top 10 poses")

ax.annotate("BETTER than shipped\n(poses nobody picks)", xy=(10, RHO_ALL[-1]),
            xytext=(6.1, 0.283), fontsize=11.5, color="#1b6ca8", weight="bold",
            ha="center")
ax.annotate("WORSE than shipped\n(the only poses selection reads)", xy=(10, RHO_TOP[-1]),
            xytext=(5.6, -0.197), fontsize=11.5, color="#c0392b", weight="bold",
            ha="center")

ax.set_xlabel("epoch", fontsize=13)
ax.set_ylabel("within-interface Spearman vs true DockQ", fontsize=13)
ax.set_title("The ranking gain is in the wrong place, and 10 epochs did not move it",
             fontsize=14.5, weight="bold", pad=12)
ax.set_xticks(EP)
ax.set_ylim(-0.235, 0.325)
ax.tick_params(labelsize=12)
ax.grid(alpha=.25)
ax.legend(fontsize=11.5, loc="center left", framealpha=.95)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("figures/gain_location_simple.png", dpi=155)
print("  -> figures/gain_location_simple.png")
