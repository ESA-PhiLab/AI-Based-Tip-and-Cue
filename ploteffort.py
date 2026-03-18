import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator


PLOT_FONT_SIZE_TITLE = 16
PLOT_FONT_SIZE_AXIS = 18
PLOT_FONT_SIZE_TICKS = 14
PLOT_FONT_SIZE_LEGEND = 14
PLOT_FONT_SIZE_LEGEND_LARGE = 14

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "lines.linewidth": 1.4,
    "lines.antialiased": True,
    "axes.titlesize": PLOT_FONT_SIZE_TITLE,
    "axes.labelsize": PLOT_FONT_SIZE_AXIS,
    "xtick.labelsize": PLOT_FONT_SIZE_TICKS,
    "ytick.labelsize": PLOT_FONT_SIZE_TICKS,
    "legend.fontsize": PLOT_FONT_SIZE_LEGEND,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "figure.dpi": 120,
    "axes.labelpad": 12,
})


def plot_thesis_effort(month_values: dict[str, float], start: str = "2025-04", end: str = "2026-04") -> None:
    """Plot monthly thesis effort with smooth interpolation and consistent styling."""
    dates = pd.date_range(start=start, end=end, freq="MS")
    labels = [d.strftime("%B %Y") for d in dates]

    values = []
    for label in labels:
        if label not in month_values:
            raise ValueError(f"Missing value for '{label}'.")
        value = month_values[label]
        if not (0 <= value <= 3):
            raise ValueError(f"Value for '{label}' must be between 0 and 3.")
        values.append(value)

    x = np.arange(len(dates), dtype=float)
    y = np.array(values, dtype=float)

    interpolator = PchipInterpolator(x, y)
    x_dense = np.linspace(x.min(), x.max(), 400)
    y_dense = interpolator(x_dense)
    dense_dates = pd.to_datetime(dates[0]) + pd.to_timedelta(x_dense * 30.4375, unit="D")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dense_dates, y_dense)
    ax.scatter(dates, y, zorder=3)

    ax.set_xticks(dates)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(0, 2.0)
    ax.set_yticks(np.arange(0, 2.1, 0.5))

    ax.set_ylabel("Thesis Effort")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Thesis Effort")

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", width=1.2)

    plt.tight_layout()
    plt.show()


plot_thesis_effort({
    "April 2025": 1.0,
    "May 2025": 0.8,
    "June 2025": 1.0,
    "July 2025": 1.0,
    "August 2025": 0.2,
    "September 2025": 0.5,
    "October 2025": 0.5,
    "November 2025": 0.0,
    "December 2025": 0.5,
    "January 2026": 0.1,
    "February 2026": 1.6,
    "March 2026": 1.6,
    "April 2026": 1.6,
})