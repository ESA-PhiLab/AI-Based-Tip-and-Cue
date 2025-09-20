import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pykep as pk

from simulation.constellation import analyze_keplerian_constellation
from matplotlib.collections import PolyCollection
import plotly.graph_objects as go
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from PIL import Image
import os

import mpld3
import os


import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FuncFormatter, FormatStrFormatter

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.use("TkAgg")

def plot_earth(ax, radius=6371e3, color='lightgray', alpha=0.3, resolution=50):
    """Plots a wireframe Earth sphere on the given 3D axes."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')


def plot_constallation(planet_list_tip, planet_list_cue, R_earth=6371e3, plot_margin=500e3):

    nPlanes_tip, nSats_tip, a_tip = analyze_keplerian_constellation(planet_list_tip)
    nPlanes_cue, nSats_cue, a_cue = analyze_keplerian_constellation(planet_list_cue)
    r_max = np.max([a_tip, a_cue]) + plot_margin

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.axes(projection='3d')

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)
    ax.set_box_aspect([1, 1, 1])

    # Plot Earth
    plot_earth(ax, radius=R_earth, color='gray', alpha=0.3)

    # Plot CUE constellation
    cue_color = sns.color_palette("crest")[0]
    for i in range(nPlanes_cue * nSats_cue):
        color_idx = i // nSats_cue
        pk.orbit_plots.plot_planet(planet_list_cue[i], axes=ax, s=50, color=cue_color)

    # Plot TIP constellation
    tip_color = sns.color_palette("flare")[0]
    for i in range(nPlanes_tip * nSats_tip):
        color_idx = i // nSats_tip
        pk.orbit_plots.plot_planet(planet_list_tip[i], axes=ax, s=150, color=tip_color)

    # Manual legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Tip',
               markerfacecolor=tip_color, markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Cue',
               markerfacecolor=cue_color, markersize=8),
    ]
    ax.legend(handles=legend_elements)

    plt.show()



def plot_orbits(trajectories):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    x_all, y_all = [], []

    for name, data in trajectories.items():
        r_arr = np.array(data["r"])
        v_arr = np.array(data["v"])
        x_all.extend(r_arr[:, 0])
        y_all.extend(r_arr[:, 1])

        orbit_line, = ax[0].plot(r_arr[:, 0], r_arr[:, 1], label=name)
        color = orbit_line.get_color()
        ax[0].plot(r_arr[0, 0], r_arr[0, 1], marker='o', color=color, markersize=6, linestyle='None')
        ax[1].plot([np.linalg.norm(v) for v in v_arr], label=name, color=color)

    ax_limit = max(max(abs(min(x_all)), abs(max(x_all))), max(abs(min(y_all)), abs(max(y_all))))
    ax[0].set_xlim(-ax_limit, ax_limit)
    ax[0].set_ylim(-ax_limit, ax_limit)
    ax[0].set_aspect('equal', adjustable='box')

    ax[0].set_title("Orbit")
    ax[0].set_xlabel("X [m]")
    ax[0].set_ylabel("Y [m]")
    ax[0].legend()

    ax[1].set_title("Velocity magnitude")
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Speed [m/s]")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_fov_on_map(intersections, ax):

    # Load the Map Image
    Image.MAX_IMAGE_PIXELS = None
    folder_worldmap = os.path.dirname(os.path.realpath(__file__))
    img = mpimg.imread(os.path.join(folder_worldmap, "WorldMap2.jpg"))

    # Draw the Map
    ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
    ax.set_global()

    # Vereify the input format
    if intersections.shape[1] != 2:
        raise ValueError("Input Matrix should have 2 columns (lat, lon, alt)")

    n_points = intersections.shape[0]

    # Build the footprint
    if n_points == 4:
        fovs = [intersections]
    elif n_points == 8:
        fovs = [intersections[:4, :], intersections[4:, :]]
    else:
        raise ValueError("Edge-points of the footprint should be 4 o 8.")

    for fov in fovs:
        latitudes = fov[:, 0]
        longitudes = fov[:, 1]

        # close the polyong
        latitudes = np.append(latitudes, latitudes[0])
        longitudes = np.append(longitudes, longitudes[0])

        # Plot and fill the polygon
        ax.plot(longitudes, latitudes, color='red', linestyle='-', transform=ccrs.PlateCarree())
        ax.fill(longitudes, latitudes, color='red', alpha=0.1, transform=ccrs.PlateCarree())
        ax.scatter(longitudes, latitudes, color='black', s=0.001, transform=ccrs.PlateCarree())

    # One legend
    fov_line = Line2D([0], [0], color='red', lw=2, label='Field of View')
    ax.legend(handles=[fov_line], loc='lower left')

def plot_all_fov_footprints(all_fov_polygons, known_targets, extension = "", show_plot=True):
    Image.MAX_IMAGE_PIXELS = None
    fig, ax_map = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    folder_worldmap = os.path.dirname(os.path.realpath(__file__))
    img = mpimg.imread(os.path.join(folder_worldmap, "WorldMap2.jpg"))

    ax_map.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
    ax_map.set_global()

    for intersections in all_fov_polygons:
        n_points = intersections.shape[0]
        if intersections.shape[1] != 2:
            raise ValueError("Input Matrix should have 2 columns (lat, lon)")
        if n_points == 4:
            fovs = [intersections]
        elif n_points == 8:
            fovs = [intersections[:4, :], intersections[4:, :]]
        else:
            raise ValueError("Edge-points of the footprint should be 4 or 8.")

        for fov in fovs:
            latitudes = np.append(fov[:, 0], fov[0, 0])
            longitudes = np.append(fov[:, 1], fov[0, 1])
            ax_map.plot(longitudes, latitudes, color='red', linestyle='-', transform=ccrs.PlateCarree())
            ax_map.fill(longitudes, latitudes, color='red', alpha=0.1, transform=ccrs.PlateCarree())

    for target_geodetic in known_targets:
        ax_map.plot(target_geodetic[1], target_geodetic[0], marker='o', color='green',
                    markersize=4, transform=ccrs.PlateCarree())

    fov_line = Line2D([0], [0], color='red', lw=2, label='Field of View')
    ax_map.legend(handles=[fov_line], loc='lower left')

    ax_map.set_title("Satellite Footprints")

    # Always save if save_path is given

    html_path = f"footprints_{extension}.html"
    mpld3.save_html(fig, html_path)

    # Show only if interactive backend is used
    if show_plot:
        try:
            plt.show()
        except Exception:
            print("Warning: Could not display plot (non-interactive backend).")

    return fig

def plot_all_fov_footprints_plotly(all_fov_polygons, all_targets, observed_targets, nPlanes, nSats, extension="", verbose=True, plot_whale_trajectories=False, whale_trajectories=None):
    """Plot satellite footprints, targets, and optionally whale trajectories with Plotly."""
    fig = go.Figure()

    # Whale trajectories
    if plot_whale_trajectories and whale_trajectories:
        if verbose:
            print(f"\tPlot whale trajectories {extension}")
        for whale_id, traj in whale_trajectories.items():
            fig.add_trace(go.Scattergeo(
                lon=[p[1] for p in traj],
                lat=[p[0] for p in traj],
                mode="lines",
                line=dict(width=1, color="orange"),
                name=f"Whale {whale_id} path",
                showlegend=False  # avoids hundreds of legend entries
            ))

    if verbose:
        print(f"\tPlot footprints {extension}")

    nSats_tot = nPlanes * nSats

    t_start = time.time()
    for i, fov in enumerate(all_fov_polygons):
        lats = list(fov[:, 0]) + [fov[0, 0]]
        lons = list(fov[:, 1]) + [fov[0, 1]]

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(color="dodgerblue", width=1),
            showlegend=False
        ))

        n = i % nSats_tot
        if verbose and n > 0 and n % 10000 == 0:
            t_end = time.time()
            hours, rem = divmod(t_end - t_start, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"\t\t {n} Added footprint | Time: {int(hours)}h {int(minutes)}m {seconds:.0f}s")
            t_start = time.time()

    # Targets not observed
    if verbose:
        print(f"\tPlot targets {extension}")

    if all_targets:
        observed_ids = set(observed_targets.keys()) if observed_targets else set()
        unobserved = [w for idx, w in all_targets.items() if idx not in observed_ids]

        if unobserved:
            fig.add_trace(go.Scattergeo(
                lon=[w.lon for w in unobserved],
                lat=[w.lat for w in unobserved],
                mode="markers",
                marker=dict(color="red", size=5),
                name="Unobserved targets"
            ))

    # Observed targets
    if observed_targets:
        fig.add_trace(go.Scattergeo(
            lon=[w.lon for w in observed_targets.values()],
            lat=[w.lat for w in observed_targets.values()],
            mode="markers",
            marker=dict(color="green", size=6),
            name="Observed"
        ))

    fig.update_layout(
        title="Satellite Footprints",
        geo=dict(
            projection_type="equirectangular",
            showland=True,
            landcolor="rgb(230,230,230)",
            showocean=True,
            oceancolor="rgb(200,220,255)",
            showcountries=True,
            countrycolor="black"
        )
    )

    html_path = f"footprints_{extension}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def plot_offnadir_distribution(excel_file, bin_size_deg=5):
    try:
        df = pd.read_excel(excel_file, sheet_name="Cue")
        if "offnadir_deg" not in df.columns:
            print("offnadir_deg column not found in Cue")
            return
        angles = df["offnadir_deg"].dropna()
        if angles.empty:
            print("No off-nadir data to plot")
            return

        max_angle = int(np.ceil(angles.max() / bin_size_deg) * bin_size_deg)
        bins = np.arange(0, max_angle + bin_size_deg, bin_size_deg)

        plt.figure(figsize=(8, 5))
        counts, _, _ = plt.hist(angles, bins=bins, edgecolor="black", color="tab:blue")
        plt.xlabel("Off-nadir angle (degrees)")
        plt.ylabel("Count")
        # plt.title(f"Off-nadir Angle Distribution ({bin_size_deg}° bins)")
        plt.xticks(bins*2)

        ymax = counts.max()
        if ymax <= 1:
            plt.yticks([0, 1])
        else:
            plt.gca().yaxis.set_major_locator(MultipleLocator(2))

       # plt.gca().xaxis.set_major_locator(MultipleLocator(bin_size_deg / 2.0))
       # plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.1f"))



        plt.grid(False)                 # clear all grid lines
        plt.grid(axis="y", alpha=0.5)   # only horizontal grid
        plt.tight_layout()

        plot_path = os.path.join(f"offnadir.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        # print(f"Saved off-nadir distribution plot -> {plot_path.replace(os.sep, '/')}")

    except Exception as e:
        print(f"Could not generate off-nadir distribution plot: {e}")



def plot_latency_distribution(excel_file, latency_col, bin_size_sec=30):
    try:
        df = pd.read_excel(excel_file, sheet_name="Cue")
        if latency_col not in df.columns:
            print(f"{latency_col} column not found in Cue")
            return
        latency = df[latency_col].dropna()
        if latency.empty:
            print("No latency data to plot")
            return

        # Use seconds for binning
        latency_sec = latency
        max_latency = int(np.ceil(latency_sec.max() / bin_size_sec) * bin_size_sec)
        bins = np.arange(0, max_latency + bin_size_sec, bin_size_sec)

        plt.figure(figsize=(8, 5))
        counts, _, _ = plt.hist(latency_sec, bins=bins, edgecolor="black", color="tab:green")

        # Axis labels
        plt.xlabel("Latency (minutes:seconds)")
        plt.ylabel("Count")
        # plt.title(f"Latency Distribution({bin_size_sec}s bins)")

        # Format x ticks as MM:SS
        def format_mmss(x, _):
            minutes = int(x // 60)
            seconds = int(x % 60)
            return f"{minutes:d}:{seconds:02d}"

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(format_mmss))
        ax.xaxis.set_major_locator(MultipleLocator(bin_size_sec*2))  # tick every 60s (1 min)

        ymax = counts.max()
        if ymax <= 1:
            plt.yticks([0, 1])
        else:
            ax.yaxis.set_major_locator(MultipleLocator(2))

        plt.grid(False)
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(f"{latency_col}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        # print(f"Saved latency distribution plot to {plot_path.replace(os.sep, '/')}")

    except Exception as e:
        print(f"Could not generate latency distribution plot: {e}")


def plot_viewing_time_distribution(excel_file, viewing_time_col, bin_size_sec=30):
    try:
        df = pd.read_excel(excel_file, sheet_name="Cue")
        if viewing_time_col not in df.columns:
            print(f"{viewing_time_col} column not found in Cue")
            return
        viewing_time = df[viewing_time_col].dropna()
        if viewing_time.empty:
            print("No viewing time data to plot")
            return

        # Use seconds for binning
        viewing_time_sec = viewing_time
        max_viewing_time = int(np.ceil(viewing_time_sec.max() / bin_size_sec) * bin_size_sec)
        bins = np.arange(0, max_viewing_time + bin_size_sec, bin_size_sec)

        plt.figure(figsize=(8, 5))
        counts, _, _ = plt.hist(viewing_time_sec, bins=bins, edgecolor="black", color="tab:orange")

        # Axis labels
        plt.xlabel("Viewing time (minutes:seconds)")
        plt.ylabel("Count")
        # plt.title(f"Viewing time Distribution({bin_size_sec}s bins)")

        # Format x ticks as MM:SS
        def format_mmss(x, _):
            minutes = int(x // 60)
            seconds = int(x % 60)
            return f"{minutes:d}:{seconds:02d}"

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(format_mmss))
        ax.xaxis.set_major_locator(MultipleLocator(bin_size_sec*2))  # tick every 60s (1 min)

        ymax = counts.max()
        if ymax <= 1:
            plt.yticks([0, 1])
        else:
            ax.yaxis.set_major_locator(MultipleLocator(2))

        plt.grid(False)
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(f"{viewing_time_col}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        # print(f"Saved viewing time distribution plot to {plot_path.replace(os.sep, '/')}")

    except Exception as e:
        print(f"Could not generate viewing time distribution plot: {e}")

