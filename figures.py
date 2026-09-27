"""Render README charts from results/*.json into figures/."""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures"
C1, C2, C3, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9a9891", "#2b2f2b", "#e1e0d9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": "#666",
                     "ytick.color": "#666", "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def main():
    a = json.loads((HERE / "results" / "analysis.json").read_text())
    m = json.loads((HERE / "results" / "model.json").read_text())
    OUT.mkdir(exist_ok=True)

    d = pd.DataFrame(a["daily"]).fillna({"revenue": 0}); d["day"] = pd.to_datetime(d["day"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.plot(d["day"], d["revenue"] / 1e6, color=C1, lw=2); ax1.fill_between(d["day"], d["revenue"] / 1e6, color=C1, alpha=0.1)
    ax1.set(ylabel="Revenue ($M)", title="Daily revenue and active users, Oct–Nov 2019", ylim=(0, None))
    gap = pd.Timestamp("2019-11-15")
    ax1.annotate("15 Nov: 5.7M views,\nno purchases logged", xy=(gap, 0.5), xytext=(pd.Timestamp("2019-10-24"), 40),
                 arrowprops={"arrowstyle": "->", "color": INK}, color=INK, fontsize=9)
    ax2.plot(d["day"], d["active_users"] / 1e3, color=C3, lw=2)
    ax2.set(ylabel="Active users (k)", ylim=(0, None))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig(OUT / "daily.png"); plt.close(fig)

    f = [dict(r, category=r["category"] or "uncategorised") for r in a["funnel_by_category"]][::-1]
    y = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y + 0.2, [r["cart_rate"] * 100 for r in f], height=0.38, color=C1, label="Sessions with a cart add")
    ax.barh(y - 0.2, [r["purchase_rate"] * 100 for r in f], height=0.38, color=C2, label="Sessions with a purchase")
    ax.set_yticks(y, [r["category"] for r in f])
    ax.set(xlabel="% of sessions", title="Session funnel by main category")
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "funnel_by_category.png"); plt.close(fig)

    hw = a["hour_weekday"]
    grid = np.zeros((7, 24))
    for r in hw:
        grid[int(r["dow"]), int(r["hour"])] = r["purchases"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(grid / 1e3, cmap="Blues", aspect="auto")
    ax.set_yticks(range(7), DAYS); ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.grid(False)
    ax.set(xlabel="Hour (UTC)", title="Purchases (thousands) by weekday and hour")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.savefig(OUT / "hour_weekday.png"); plt.close(fig)

    pb = [r for r in a["smartphone_price_bands"] if r["views"] >= 10000]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar([f"${int(r['price_band'])}" + ("+" if r["price_band"] >= 1500 else "") for r in pb],
                  [r["purchases_per_view"] * 100 for r in pb], color=C1, width=0.6)
    ax.set(ylabel="Purchases per 100 views", title="Smartphones: conversion by price band")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "smartphone_price.png"); plt.close(fig)

    c = pd.DataFrame(a["cohorts"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (cohort, g) in enumerate(list(c.groupby("cohort"))[:4]):
        g = g[g["week_n"] >= 1]
        ax.plot(g["week_n"], g["active_share"] * 100, lw=2, marker="o", ms=4,
                color=[C1, C2, C3, "#4a3aa7"][i], label=f"First seen week of {pd.to_datetime(cohort):%d %b}")
    ax.set(xlabel="Weeks after first visit", ylabel="Active again (%)", title="Weekly return rate by first-visit cohort", ylim=(0, None))
    ax.legend(frameon=False)
    fig.savefig(OUT / "cohorts.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0] + [g[0] for g in m["gains"]], [0] + [g[1] * 100 for g in m["gains"]], color=C1, lw=2, label="Gradient boosting")
    ax.plot([0] + [g[0] for g in m["gains_rule"]], [0] + [g[1] * 100 for g in m["gains_rule"]], color=C2, lw=2,
            label="Rule: carted in the last 7 days")
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, label="Random")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Users targeted (% of active users)",
           ylabel="Next-15-day buyers reached (%)", title="Who buys in the next 15 days? (16–30 Nov test window)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "gains.png"); plt.close(fig)

    imp = m["importances"][:10][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.barh([i["feature"] for i in imp], [i["auc_drop"] for i in imp], color=C1, height=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(xlabel="Drop in AUC when shuffled", title="What drives purchase propensity")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "importance.png"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
