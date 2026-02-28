import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def plot_thesis_effort(month_values: dict[str, float], start: str = "2025-04", end: str = "2026-04") -> None:
    """Plot monthly effort (0..3) with smooth monotonic interpolation and clear axes."""
    dates = pd.date_range(start=start, end=end, freq="MS")
    labels = [d.strftime("%B %Y") for d in dates]

    values = []
    for label in labels:
        if label not in month_values:
            raise ValueError(f"Missing value for '{label}'.")
        v = month_values[label]
        if not (0 <= v <= 3):
            raise ValueError(f"Value for '{label}' must be between 0 and 3.")
        values.append(v)

    x = np.arange(len(dates), dtype=float)
    y = np.array(values, dtype=float)

    interpolator = PchipInterpolator(x, y)
    x_dense = np.linspace(x.min(), x.max(), 400)
    y_dense = interpolator(x_dense)
    dense_dates = pd.to_datetime(dates[0]) + pd.to_timedelta(x_dense * 30.4375, unit="D")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dense_dates, y_dense, linewidth=2)
    ax.scatter(dates, y, zorder=3)

    ax.set_xticks(dates)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 2.0)

    ax.set_ylabel("Thesis Effort (0–3)", fontsize=12)
    ax.set_xlabel("Month", fontsize=12)

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
    "February 2026": 1.5,
    "March 2026": 1.5,
    "April 2026": 1.5,
})