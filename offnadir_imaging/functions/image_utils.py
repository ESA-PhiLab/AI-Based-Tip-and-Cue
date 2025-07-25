import numpy as np

def stack_rgb_img(DN_key, band_data):
    R = np.squeeze(band_data['red'][DN_key])
    G = np.squeeze(band_data['green'][DN_key])
    B = np.squeeze(band_data['blue'][DN_key])

    # Stack into (H, W, 3)
    DN_rgb_image = np.stack([R, G, B], axis=-1)

    return DN_rgb_image

def crop_black_border_image(img_array: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = np.mean(img_array, axis=2)
    mask = gray > threshold

    # Find rows and columns where content exists
    valid_rows = np.where(np.any(mask, axis=1))[0]
    valid_cols = np.where(np.any(mask, axis=0))[0]

    if valid_rows.size == 0 or valid_cols.size == 0:
        return img_array  # image is all black

    y0, y1 = valid_rows[0], valid_rows[-1] + 1
    x0, x1 = valid_cols[0], valid_cols[-1] + 1

    return img_array[y0:y1, x0:x1]

def radiance_to_DN2047(radiance_values, gain, offset, eff_bw, abs_cal_factor):
    DN = (radiance_values - offset) / gain / (abs_cal_factor / eff_bw)
    DN = np.array(DN)
    return DN.astype(int)

def DN2047_to_radiance(DN, gain, offset, eff_bw, abs_cal_factor):
    L = gain * DN * (abs_cal_factor / eff_bw) + offset
    return L

def DN255_to_linear(img_DN):
    img = img_DN / 255.0
    img_linear = np.power(img, 2.2)
    return img_linear

def linear_to_DN255(img_linear):
    img = np.power(img_linear, 1/2.2)
    img_DN = img * 255
    img_DN = np.array(img_DN)
    img_DN[img_DN>=255] = 255
    return img_DN.astype(int)

def DN2047_to_linear(img_DN):
    img = img_DN / 2047.0
    img_linear = np.power(img, 2.2)
    img_linear[img_linear>=1] = 1
    return img_linear

def linear_to_DN2047(img_linear):
    img = np.power(img_linear, 1/2.2)
    img_DN = img * 2047
    img_DN = np.array(img_DN)
    img_DN[img_DN>=2047] = 2047
    return img_DN.astype(int)

def DN255_to_DN2047(img_DN255):
    img_DN2047 = img_DN255 / 255 * 2047
    return img_DN2047.astype(int)

def DN2047_to_DN255(img_DN2047):
    img_DN255 = img_DN2047 / 2047 * 255
    return img_DN255.astype(int)