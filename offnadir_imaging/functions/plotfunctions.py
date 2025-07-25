import numpy as np
import pyvista
from pyvista import examples

from paseos.utils.point_transformation import Point_ECEF2Geodetic

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_earth_with_pyvista(satellite, feature, sun, R_earth):
    light = pyvista.Light()

    P_sun = Point_ECEF2Geodetic(sun[0], sun[1], sun[2])
    light.set_direction_angle(-P_sun[0] + 180,P_sun[1])

    earth = examples.planets.load_earth(radius=R_earth)
    earth_texture = examples.load_globe_texture()

    satellite_copy = satellite.copy()
    feature_copy = feature.copy()
    sun_copy = sun.copy()

    pl = pyvista.Plotter(shape=(1, 1), lighting='none')
    cubemap = examples.download_cubemap_space_4k()
    _ = pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, True)
    pl.add_light(light)

    pl.subplot(0, 0)
    pl.add_text("3D View", font_size = 12)
    pl.add_mesh(earth, texture=earth_texture, smooth_shading=True)
    pl.link_views()

    satellite_copy[0] = -satellite_copy[0]
    satellite_copy[1] = -satellite_copy[1]
    feature_copy[0] = -feature_copy[0]
    feature_copy[1] = -feature_copy[1]
    sun_copy[0] = -sun_copy[0]
    sun_copy[1] = -sun_copy[1]

    # Plot satellite (represented as a point)
    pl.add_points(satellite_copy, color="teal", point_size=16, render_points_as_spheres=True, label = 'Satellite')
    pl.add_points(feature_copy, color="violet", point_size=16, render_points_as_spheres=True, label = 'Feature')

    vector = sun_copy - feature_copy
    distance = np.linalg.norm(vector)
    direction = vector / distance
    short_vector = direction * (distance / 20000)

    pl.add_lines(np.array([satellite_copy, feature_copy]), color='black', width=3)
    pl.add_lines(np.array([feature_copy, feature_copy+short_vector]), color='peachpuff', width=3)
    pl.add_lines(np.array([[0,0,0], sun_copy/10000]), color='peachpuff', width=3)

    # pl.add_lines(np.array([np.array([0, 0, 0]), np.array([sun_copy[0], 0, 0] )/ 10000]), color='green', width=3)   # x axis

    # Set view options
    # pl.set_background('black')
    pl.show_axes()
    pl.view_isometric()
    pl.add_legend(bcolor='w', face='circle', size = (0.12, 0.12))

    pl.show(cpos="xy")

def plot_earth_slice_with_sun(satellite, feature, sun_direction, R_earth, ax2d):
    satellite_vector = np.array(satellite)
    feature_vector = np.array(feature)
    sun_vector = np.array(sun_direction)

    # Calculate the normal to the plane formed by the Earth center, satellite, and feature
    normal_vector = np.cross(satellite_vector, feature_vector)
    normal_vector /= np.linalg.norm(normal_vector)

    # Plot Earth as a circle in 2D projection on the plane
    earth_circle = plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5, label='Earth')
    ax2d.add_patch(earth_circle)

    # Compute rotation to align plane normal with Z-axis
    def rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        ux, uy, uz = axis
        return np.array([
            [cos_a + ux**2 * (1 - cos_a),      ux*uy*(1 - cos_a) - uz*sin_a, ux*uz*(1 - cos_a) + uy*sin_a],
            [uy*ux*(1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a),      uy*uz*(1 - cos_a) - ux*sin_a],
            [uz*ux*(1 - cos_a) - uy*sin_a, uz*uy*(1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
        ])

    rotation_axis = np.cross(normal_vector, [0, 0, 1])
    if np.linalg.norm(rotation_axis) < 1e-8:
        rotation_mat = np.eye(3)  # Already aligned
    else:
        rotation_angle = np.arccos(np.clip(np.dot(normal_vector, [0, 0, 1]), -1.0, 1.0))
        rotation_mat = rotation_matrix(rotation_axis, rotation_angle)

    # Rotate vectors
    sat_rot = rotation_mat @ satellite_vector
    feat_rot = rotation_mat @ feature_vector
    sun_rot = rotation_mat @ sun_vector

    x_s, y_s, _ = sat_rot
    x_f, y_f, _ = feat_rot
    x_sun, y_sun, _ = sun_rot

    # Plot line of sight and points
    ax2d.plot([x_s, x_f], [y_s, y_f], color='black', label='Line of Sight')
    ax2d.scatter(x_s, y_s, color='teal', s=100, label='Satellite')
    ax2d.scatter(x_f, y_f, color='violet', s=100, label='Feature')

    # Normalize and scale sun direction
    sun_dir_2d = np.array([x_sun, y_sun])
    sun_dir_2d = sun_dir_2d / np.linalg.norm(sun_dir_2d) * R_earth * 1.2

    # From Sun to Earth (arrow pointing toward Earth center)
    arrow_start = sun_dir_2d * R_earth * 1.2
    arrow_vec = -sun_dir_2d * R_earth * 1.2

    ax2d.arrow(
        arrow_start[0], arrow_start[1],
        arrow_vec[0], arrow_vec[1],
        head_width=R_earth * 0.07,
        color='orange',
        label='Sun Direction',
        length_includes_head=True
    )

    # Axes and plot settings
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Projection of Earth Slice with Sun Direction')
    ax2d.set_aspect('equal', 'box')

    distance = np.sqrt(x_s**2 + y_s**2)
    margin = 1.5 * (distance - R_earth)
    ax2d.set_xlim(-R_earth - margin, R_earth + margin)
    ax2d.set_ylim(-R_earth - margin, R_earth + margin)
    ax2d.legend()



def plot_target_perspective(satellite, feature, sun_direction, ax2d):
    satellite_vector = np.array(satellite)
    feature_vector = np.array(feature)
    sun_vector = np.array(sun_direction)

    print('sun', sun_vector , 'sat', satellite, 'feature', feature)

    if not (feature_vector==np.zeros_like(feature_vector)).all():
        # Calculate the normal to the plane formed by the Earth center, satellite, and feature
        normal_vector = np.cross(satellite_vector, feature_vector)
        normal_vector /= np.linalg.norm(normal_vector)


        # Compute rotation to align plane normal with Z-axis
        def rotation_matrix(axis, angle):
            axis = axis / np.linalg.norm(axis)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            ux, uy, uz = axis
            return np.array([
                [cos_a + ux**2 * (1 - cos_a),      ux*uy*(1 - cos_a) - uz*sin_a, ux*uz*(1 - cos_a) + uy*sin_a],
                [uy*ux*(1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a),      uy*uz*(1 - cos_a) - ux*sin_a],
                [uz*ux*(1 - cos_a) - uy*sin_a, uz*uy*(1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
            ])

        rotation_axis = np.cross(normal_vector, [0, 0, 1])
        if np.linalg.norm(rotation_axis) < 1e-8:
            rotation_mat = np.eye(3)  # Already aligned
        else:
            rotation_angle = np.arccos(np.clip(np.dot(normal_vector, [0, 0, 1]), -1.0, 1.0))
            rotation_mat = rotation_matrix(rotation_axis, rotation_angle)

        # Rotate vectors
        sat_rot = rotation_mat @ satellite_vector
        feat_rot = rotation_mat @ feature_vector
        sun_rot = rotation_mat @ sun_vector

        x_s, y_s, _ = sat_rot
        x_f, y_f, _ = feat_rot
        x_sun, y_sun, _ = sun_rot

        x_s_target = x_s - x_f
        y_s_target = y_s - y_f
        x_f_target = x_f - x_f
        y_f_target = y_f - y_f
        x_sun_target = x_sun - x_f
        y_sun_target = y_sun - y_f

    print('sun', [x_sun_target, y_sun_target], 'sat', [x_s_target, y_s_target], 'feature', [x_f_target, y_f_target])
    # Normalize and scale sun direction
    sun_dir_2d_target = np.array([x_sun_target, y_sun_target])
    sun_dir_2d_target = sun_dir_2d_target / np.linalg.norm(sun_dir_2d_target)

    # Plot line of sight and points
    ax2d.plot([x_s_target, x_f_target], [y_s_target, y_f_target], color='black', label='Line of Sight')
    ax2d.scatter(x_s_target, y_s_target, color='teal', s=100, label='Satellite')
    ax2d.scatter(x_f_target, y_f_target, color='violet', s=100, label='Feature')

    scaling = np.linalg.norm(x_s_target)
    print(scaling)
    # From Sun to Earth (arrow pointing toward Earth center)
    arrow_start = sun_dir_2d_target  * scaling * 1.2
    arrow_vec = -sun_dir_2d_target  * scaling * 1.2

    ax2d.arrow(
        arrow_start[0], arrow_start[1],
        arrow_vec[0], arrow_vec[1],
        color='orange',
        label='Sun Direction',
        length_includes_head=True
    )

    # Axes and plot settings
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Projection of Earth Slice with Sun Direction')
    ax2d.set_aspect('equal', 'box')
    ax2d.legend()

    distance = np.sqrt(x_s_target**2 + x_s_target**2)
    margin = 1.5 * (distance - scaling)
    ax2d.set_xlim(-scaling  - margin, scaling + margin)
    ax2d.set_ylim(-scaling - margin, scaling + margin)
    ax2d.legend()

def get_rgb(bmp):
    return np.power(np.clip(np.array(bmp), 0.0, 1.0), 1 / 2.2)
