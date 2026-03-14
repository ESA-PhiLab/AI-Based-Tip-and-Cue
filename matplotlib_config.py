
PLOT_FONT_SIZE_TITLE = 16
PLOT_FONT_SIZE_AXIS = 16
PLOT_FONT_SIZE_TICKS = 14
PLOT_FONT_SIZE_LEGEND = 14
PLOT_FONT_SIZE_LEGEND_LARGE = 16

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

    "figure.dpi": 120
    "axes.labelpad": 12
})
