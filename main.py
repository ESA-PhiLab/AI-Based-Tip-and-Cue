from offnadir_imaging.rendering import generate_image
from settings import *

DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(img_path, satellite,
                                                                                           satellite_lat, satellite_lon,
                                                                                           satellite_alt, target_lat,
                                                                                           target_lon, target_alt,
                                                                                           datetime_utc,
                                                                                           sensor_characteristics,
                                                                                           wave_properties, bools, dem_seed)

