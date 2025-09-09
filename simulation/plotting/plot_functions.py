import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pykep as pk

from simulation.constellation import analyze_keplerian_constellation

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from PIL import Image
import os


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

def plot_all_fov_footprints(all_fov_polygons, known_targets):

    # Setup figure and map background (same as in your function)
    Image.MAX_IMAGE_PIXELS = None
    fig, ax_map = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    folder_worldmap = os.path.dirname(os.path.realpath(__file__))
    img = mpimg.imread(os.path.join(folder_worldmap, "WorldMap2.jpg"))

    ax_map.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.BORDERS, linestyle=":")
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
    ax_map.set_global()


    # Now draw all stored footprints
    for intersections in all_fov_polygons:
        if intersections.shape[1] != 2:
            raise ValueError("Input Matrix should have 2 columns (lat, lon, alt)")

        n_points = intersections.shape[0]
        if n_points == 4:
            fovs = [intersections]
        elif n_points == 8:
            fovs = [intersections[:4, :], intersections[4:, :]]
        else:
            raise ValueError("Edge-points of the footprint should be 4 or 8.")

        for fov in fovs:
            latitudes = fov[:, 0]
            longitudes = fov[:, 1]

            # close the polygon
            latitudes = np.append(latitudes, latitudes[0])
            longitudes = np.append(longitudes, longitudes[0])

            # Plot and fill
            ax_map.plot(longitudes, latitudes, color='red', linestyle='-', transform=ccrs.PlateCarree())
            ax_map.fill(longitudes, latitudes, color='red', alpha=0.1, transform=ccrs.PlateCarree())
            ax_map.scatter(longitudes, latitudes, color='black', s=0.001, transform=ccrs.PlateCarree())

    # Legend once

    for target_geodetic in known_targets:
        ax_map.plot(target_geodetic[1], target_geodetic[0], marker='o', color='green', markersize=4,
                    transform=ccrs.PlateCarree())
        # ax_map.text(target_geodetic[1] - 7.5, target_geodetic[0] - 7.5, "Target", color='green', transform=ccrs.PlateCarree())

    fov_line = Line2D([0], [0], color='red', lw=2, label='Field of View')
    ax_map.legend(handles=[fov_line], loc='lower left')

    ax_map.set_title("All FOV Footprints")
    plt.show()

