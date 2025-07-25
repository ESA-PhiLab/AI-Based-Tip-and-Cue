import numpy as np
import rasterio
from rasterio.enums import Resampling
import trimesh
from mitsuba import Bitmap
import matplotlib.pyplot as plt
# import meshio



def get_obj_bounds(filepath):
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    bounds = vertices.min(axis=0), vertices.max(axis=0)
    return center, bounds

def convert_DEM(img_path, dem_path, obj_output_path, GSD, scale_km = True, print_output=False, plot_DEM = False):
    # === Load Satellite Image to Get Dimensions ===
    bmp = Bitmap(img_path)
    img_np = np.array(bmp)
    img_height, img_width = img_np.shape[:2]

    if print_output:
        print(f"Image resolution: {img_width} x {img_height}")

    # === Load and Resample DEM to Match Image Resolution ===
    with rasterio.open(dem_path) as src:
        dem_resampled = src.read(
            1,
            out_shape=(1, img_height, img_width),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        # Clean and normalize elevation
        dem_resampled[np.isnan(dem_resampled)] = 0.0
       #  dem_resampled -= dem_resampled.min()  # normalize so min is zero
       # dem_resampled = dem_resampled * h_max

        if print_output:
            print(f"DEM stats after resampling: min={dem_resampled.min()}, max={dem_resampled.max()}, mean={dem_resampled.mean()}")

    if plot_DEM:
        # === Optional: Visualize the DEM for debugging ===
        plt.imshow(dem_resampled, cmap='terrain')
        plt.colorbar(label='Elevation (m)')
        plt.title('Resampled DEM (normalized)')
        plt.show()

    if scale_km:
        scale_xy = GSD / 1000.0  # convert x/y from meters to kilometers
        scale_z = 1.0 / 1000.0  # convert elevation from meters to kilometers
    else:
        scale_xy = GSD  # convert x/y from meters to kilometers
        scale_z = 1.0   # convert elevation from meters to kilometers

    # === Create Grid Using Physical Dimensions ===
    x = np.linspace(-img_width / 2, img_width / 2, img_width) * scale_xy
    y = np.linspace(-img_height / 2, img_height / 2, img_height) * scale_xy
    xx, yy = np.meshgrid(x, y)
    zz = dem_resampled * scale_z # apply z scaling

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    # === Create Triangle Faces ===
    faces = []
    for i in range(img_height - 1):
        for j in range(img_width - 1):
            i0 = i * img_width + j
            i1 = i0 + 1
            i2 = i0 + img_width
            i3 = i2 + 1
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    faces = np.array(faces, dtype=np.uint32)

    # Compute UVs linearly from grid [0, 1]
    u = np.linspace(0, 1, img_width)
    v = np.linspace(0, 1, img_height)[::-1]  # Flip vertically
    uu, vv = np.meshgrid(u, v)
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    # Create Trimesh object with UVs
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()

    mesh.visual.uv = uvs

    # Manual OBJ export with vertices, UVs, normals, and faces
    with open(obj_output_path, 'w') as f:
        # Write vertex positions
        for v in mesh.vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')

        # Write UV coordinates
        for uv in mesh.visual.uv:
            f.write(f'vt {uv[0]} {uv[1]}\n')

        # Write vertex normals
        for n in mesh.vertex_normals:
            f.write(f'vn {n[0]*-1} {n[1]*-1} {n[2]*-1}\n')

        # Write faces with vertex, UV, and normal indices
        for face in mesh.faces:
            v1, v2, v3 = face + 1  # OBJ is 1-based
            f.write(f'f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n')

    if print_output:
        print(f"Mesh exported to {obj_output_path}")
        print(f"Mesh shape: vertices={vertices.shape[0]}, faces={faces.shape[0]}")

        center, (vmin, vmax) = get_obj_bounds(obj_output_path)
        print("Mesh center:", center)
        print("Vertex bounds (min):", vmin)
        print("Vertex bounds (max):", vmax)

if __name__ == "__main__":

    # === CONFIGURATION ===
    dem_path = '../input_dem_WV.tiff'
    img_path = '../input_img_WV.PNG'
    obj_path = "../dem_mesh_WV.obj"

    GSD = 0.37  # meters per pixel

    convert_DEM(img_path, dem_path, obj_path, GSD, scale_km=True, print_output=True, plot_DEM = True)


