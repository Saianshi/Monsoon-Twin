"""
Renders the two charts the Avinya deck needs, styled to match the slide template.

Run from anywhere:

    python "C:\\IIT Guwahati\\Digital Twin\\avinya-twin\\data\\make_deck_charts.py"

Outputs (in the same data\ folder as this script):
    chart_rh_distribution.png   -> slide 4
    chart_vpd_envelope.png      -> slide 5   (only if the sim package imports)
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- deck styling
TEAL = "#5CE1E6"
DARK = "#2E3639"
ORANGE = "#FF741F"
BLUE = "#3987E5"
OLIVE = "#7C8F4D"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # swap to "Arial" if you have it installed
    "font.size": 12,
    "axes.edgecolor": DARK,
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.linewidth": 1.0,
    "figure.dpi": 200,
})

# ------------------------------------------------------------------- paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

OUT_DIR = SCRIPT_DIR                                    # PNGs yahin save honge
DEFAULT_WEATHER_CSV = os.path.join(SCRIPT_DIR, "guwahati_2025.csv")

MONSOON_MONTHS = (6, 7, 8)


def style_axes(ax):
    """Transparent panel, teal figure, only faint horizontal gridlines."""
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    ax.grid(axis="y", color=DARK, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def find_weather_csv() -> str:
    """Locate the cached hourly weather file, whatever it ended up being named."""
    patterns = [
        os.path.join(SCRIPT_DIR, "*.csv"),
        os.path.join(REPO_ROOT, "data", "**", "*.csv"),
        os.path.join(REPO_ROOT, "results", "weather*.csv"),
        os.path.join(REPO_ROOT, "*.csv"),
    ]
    seen = []
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            if path in seen:
                continue
            seen.append(path)
            try:
                head = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            cols = {c.lower() for c in head.columns}
            if any("rh" in c or "humid" in c for c in cols):
                return path
    raise FileNotFoundError(
        "Could not find an hourly weather CSV. Pass the path explicitly:\n"
        "    python make_deck_charts.py path/to/weather.csv"
    )


def load_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = None
    for cand in ("time", "date", "datetime", "timestamp", df.columns[0]):
        if cand in df.columns:
            time_col = cand
            break
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "rh" in lc and "in" not in lc.split("_")[-1:]:
            rename[c] = "RH_out"
        elif "relative_humidity" in lc or lc == "humidity":
            rename[c] = "RH_out"
    df = df.rename(columns=rename)

    if "RH_out" not in df.columns:
        raise KeyError(
            f"No outdoor humidity column found in {path}. Columns present: {list(df.columns)}"
        )
    return df


# ------------------------------------------------------------------- Chart A
def chart_rh_distribution(df: pd.DataFrame) -> str:
    monsoon = df[df.index.month.isin(MONSOON_MONTHS)]
    rh = monsoon["RH_out"].dropna().to_numpy()
    if rh.size == 0:
        raise ValueError("No June to August rows in the weather file.")

    mean_rh = rh.mean()
    above_90 = 100.0 * (rh >= 90).mean()
    n_years = monsoon.index.year.nunique()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(TEAL)
    style_axes(ax)

    bins = np.arange(30, 101, 2.5)
    counts, edges = np.histogram(rh, bins=bins)
    pct = 100.0 * counts / counts.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])

    colours = [ORANGE if c >= 90 else BLUE for c in centres]
    ax.bar(centres, pct, width=2.2, color=colours, edgecolor="none", alpha=0.92)

    ax.set_ylim(0, pct.max() * 1.18)
    ax.axvline(mean_rh, color=DARK, linewidth=1.8, linestyle="--")
    ax.annotate(
        f"mean {mean_rh:.1f}%",
        xy=(mean_rh, ax.get_ylim()[1] * 0.95),
        xytext=(-8, 0), textcoords="offset points",
        ha="right", va="top", fontsize=11, fontweight="bold", color=DARK,
    )

    ax.set_xlabel("Outdoor relative humidity (%)", fontsize=12)
    ax.set_ylabel("Share of monsoon hours (%)", fontsize=12)
    ax.set_title(
        "Outdoor humidity, Guwahati monsoon (June to August)",
        fontsize=13.5, fontweight="bold", pad=14, loc="left",
    )
    ax.set_xlim(30, 100)

    ax.text(
        0.0, -0.22,
        f"{above_90:.0f}% of monsoon hours sit at or above 90% RH (orange bars).  "
        f"{n_years} season{'s' if n_years > 1 else ''} of ERA5 reanalysis, hourly.",
        transform=ax.transAxes, fontsize=10, color=DARK, alpha=0.85,
    )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "chart_rh_distribution.png")
    fig.savefig(out, facecolor=TEAL, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    print(f"  mean monsoon RH  : {mean_rh:.2f}%")
    print(f"  hours >= 90% RH  : {above_90:.1f}%")
    print(f"  seasons in file  : {n_years}")
    return out


# ------------------------------------------------------------------- Chart B
def chart_vpd_envelope(df: pd.DataFrame) -> str | None:
    """Replay the twin at vent fully closed and vent fully open, fan off."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from sim.polyhouse import Polyhouse       # noqa
        from sim.psychro import vpd_kpa           # noqa
    except Exception as exc:
        print(
            "\n  Chart B skipped: could not import the sim package "
            f"({type(exc).__name__}: {exc}).\n"
            f"  Looked for it in: {REPO_ROOT}\n"
            "  Download that chart straight from the live app instead:\n"
            "  Why Ventilation Alone Fails page, hover the envelope chart, "
            "click the camera icon."
        )
        return None

    monsoon = df[df.index.month.isin(MONSOON_MONTHS)]
    start = monsoon.index[0]
    week = monsoon.loc[start:start + pd.Timedelta(days=7)]

    def boundary_run(vent_frac: float) -> np.ndarray:
        ph = Polyhouse()
        out = []
        for _, w in week.iterrows():
            for _ in range(12):                    # 12 x 5 min = 1 hour
                step = ph.step(
                    float(w.get("T_out", w.get("temperature_2m"))),
                    float(w["RH_out"]),
                    float(w.get("I_solar", w.get("shortwave_radiation", 0.0))),
                    vent_frac, 300.0,
                )
            out.append(step["VPD"])
        return np.asarray(out)

    floor = boundary_run(0.0)
    ceiling = boundary_run(1.0)
    lo = np.minimum(floor, ceiling)
    hi = np.maximum(floor, ceiling)

    # wide and short: deck ke content box me theek baithta hai
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    fig.patch.set_facecolor(TEAL)
    style_axes(ax)

    ax.axhspan(0.8, 1.2, color=OLIVE, alpha=0.35, zorder=0)

    ax.fill_between(week.index, lo, hi, color=BLUE, alpha=0.35, linewidth=0,
                    label="Reachable by some vent setting")
    ax.plot(week.index, hi, color=BLUE, linewidth=1.6, label="Ceiling (vent fully open)")
    ax.plot(week.index, lo, color=DARK, linewidth=1.2, alpha=0.7,
            label="Floor (vent fully closed)")

    # band label band ke ANDAR, halka background taaki curve ke upar padhne me aaye
    ax.text(
        0.012, 1.0, "0.8 to 1.2 kPa target band",
        transform=ax.get_yaxis_transform(), fontsize=9.5, color=DARK,
        va="center", ha="left", zorder=5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                  edgecolor="none", alpha=0.75),
    )

    ax.set_ylabel("Indoor VPD (kPa)", fontsize=11.5, labelpad=8)
    ax.set_title(
        "Achievable VPD envelope, representative monsoon week",
        fontsize=13.5, fontweight="bold", pad=12, loc="left",
    )

    ax.set_ylim(0, max(hi.max(), 1.2) * 1.10)
    ax.set_xlim(week.index[0], week.index[-1])

    # ek tick per din, seedha -- koi rotation nahi, koi overlap nahi
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=10.5)
    ax.tick_params(axis="y", labelsize=10.5)

    # legend plot ke NEECHE -- title aur curves dono se door
    ax.legend(
        frameon=False, fontsize=10.5, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        handlelength=1.8, columnspacing=2.0,
    )

    fig.text(
        0.5, -0.02,
        "The shaded band is every VPD value reachable at that hour, fan off. "
        "For most night hours the target sits entirely outside it.",
        ha="center", fontsize=10, color=DARK, alpha=0.85,
    )

    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.24)
    out = os.path.join(OUT_DIR, "chart_vpd_envelope.png")
    fig.savefig(out, facecolor=TEAL, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.exists(DEFAULT_WEATHER_CSV):
        path = DEFAULT_WEATHER_CSV
    else:
        path = find_weather_csv()

    print(f"Weather file: {path}")
    print(f"Output folder: {OUT_DIR}")

    df = load_weather(path)
    print("\nChart A, outdoor RH distribution")
    a = chart_rh_distribution(df)
    print(f"  written -> {a}")

    print("\nChart B, achievable VPD envelope")
    b = chart_vpd_envelope(df)
    if b:
        print(f"  written -> {b}")

    print("\nPut the PNGs into the dashed boxes on slides 4 and 5.")


if __name__ == "__main__":
    main()