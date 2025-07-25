import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt



def make_waves(X, Y, num_waves=50, wave_min=0.05, wave_max=0.3):
    """Generate a realistic random ocean wave field."""
    Z = np.zeros_like(X, dtype=np.float32)

    for _ in range(num_waves):
        wavelength = np.random.uniform(5, 50)  # meters
        amplitude = np.random.uniform(wave_min, wave_max)
        angle = np.random.uniform(0, 2 * np.pi)

        kx = np.cos(angle) * 2 * np.pi / wavelength
        ky = np.sin(angle) * 2 * np.pi / wavelength
        phase = np.random.uniform(0, 2 * np.pi)

        Z += amplitude * np.sin(kx * X + ky * Y + phase)

    # Normalize to [0, 1] and rescale
    Z = Z - Z.min()

    if Z.max() > 0:
        Z = Z / Z.max()
    Z = wave_min + Z * (wave_max - wave_min)  # ~0.1–1.1 m

    return Z.astype(np.float32)


def add_curvature(X, Y, Z, R=6371000.0):
    """Subtract exact Earth curvature sagitta from elevation."""
    D = np.sqrt(X**2 + Y**2)
    curvature = R - np.sqrt(R**2 - D**2)

   #  curvature = curvature * 150000
    return Z - curvature

def get_DEM(input_path, output_path, GSD, wave_properties, random_seed, waves, curvature, plot_DEM):

    np.random.seed(random_seed)

    num_waves = wave_properties['num_waves']
    wave_min = wave_properties['wave_min']
    wave_max = wave_properties['wave_max']

    # === Load image to get pixel dimensions ===
    image = Image.open(input_path)
    width, height = image.size

    # === Define spatial bounds ===
    physical_width = width * GSD
    physical_height = height * GSD

    # === Define GeoTIFF transform and CRS ===
    transform = from_origin(0.0, physical_height, GSD, -GSD)
    crs = "EPSG:32633"  # Example CRS (UTM Zone 33N)

    # === Generate coordinates in meters, centered ===
    x = np.linspace(-physical_width / 2, physical_width / 2, width)
    y = np.linspace(-physical_height / 2, physical_height / 2, height)
    X, Y = np.meshgrid(x, y)

    # === Initialize elevation ===
    Z = np.zeros((height, width), dtype=np.float32)

    # === Apply waves if enabled ===
    if waves:
        Z = make_waves(X, Y, num_waves=num_waves, wave_min=wave_min, wave_max=wave_max)

    # === Apply curvature if enabled ===
    if curvature:
        Z = add_curvature(X, Y, Z)

    # === Metadata ===
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw"
    }

    # === Save to GeoTIFF ===
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(Z, 1)


    if plot_DEM:
        # === Preview ===
        plt.imshow(Z, cmap="viridis")
        plt.colorbar(label="Elevation (m)")
        plt.title(f"Synthetic DEM (waves={waves}, curvature={curvature})")
        plt.show()

if __name__ == "__main__":

    random_seed = 42

    GSD = 0.37  # m/pixel

    num_waves = 50
    wave_min = 0.05  # m
    wave_max = 0.5  # m

    waves = True
    curvature = True

    img_path = "../input_img_WV.PNG"
    dem_path = "../input_dem_WV.tiff"

    get_DEM(img_path, dem_path, GSD, num_waves, wave_min, wave_max, random_seed=42, waves=True, curvature=True, plot_DEM=True)
    print(f"Saved synthetic DEM to {dem_path}\n")
