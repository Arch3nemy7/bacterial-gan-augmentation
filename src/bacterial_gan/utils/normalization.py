"""Color normalization utilities for bacterial images."""

import numpy as np
from PIL import Image


def normalize_macenko(img: np.ndarray, Io=240, alpha=1, beta=0.15):
    """
    Implementation of Macenko color normalization.
    Takes NumPy array (H, W, 3) and returns normalized NumPy array
    with stain invariant color normalization.

    Args:
        img (np.ndarray): Input image.
        Io (int): Light intensity (background).
        alpha (float): Percentile for angle calculation.
        beta (float): Threshold for optical density.

    Returns:
        np.ndarray: Color-normalized image.
    """
    # Reference stain matrix (can be adjusted from ideal target image)
    stain_ref = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])

    # Maximum stain concentration reference
    max_conc_ref = np.array([1.9705, 1.0308])

    # Convert to appropriate data type
    img = img.astype(np.float64)

    # 1. Convert RGB to Optical Density (OD)
    h, w, c = img.shape
    img_reshaped = img.reshape((-1, 3))

    # Avoid log of zero by adding small epsilon (fixed formula)
    OD = -np.log10((img_reshaped + 1e-8) / Io)

    # 2. Remove background pixels (filter low OD pixels)
    # Keep only pixels with sufficient optical density
    OD_thresh = beta
    mask = np.all(OD > OD_thresh, axis=1)

    # If not enough foreground pixels, return original image
    if np.sum(mask) < 2:
        return img.astype(np.uint8)

    OD_hat = OD[mask]

    # 3. Calculate SVD decomposition
    try:
        _, _, V = np.linalg.svd(OD_hat, full_matrices=False)
    except np.linalg.LinAlgError:
        # If SVD fails, return original image
        return img.astype(np.uint8)

    # 4. Project to first two singular vectors
    V = V[0:2, :].T
    proj = np.dot(OD_hat, V)

    # 5. Determine extreme angles
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    # 6. Reconstruct stain matrix
    v_min = np.dot(V, np.array([np.cos(min_phi), np.sin(min_phi)]))
    v_max = np.dot(V, np.array([np.cos(max_phi), np.sin(max_phi)]))

    # Check which vector has higher first component to determine order
    if v_min[0] > v_max[0]:
        stain_matrix = np.array([v_max, v_min]).T
    else:
        stain_matrix = np.array([v_min, v_max]).T

    # 7. Project onto stain matrix
    # Separate stain concentration from original image
    try:
        conc, _, _, _ = np.linalg.lstsq(stain_matrix, OD.T, rcond=None)
    except np.linalg.LinAlgError:
        # If lstsq fails, return original image
        return img.astype(np.uint8)

    # Normalize stain concentration
    max_conc = np.percentile(conc, 99, axis=1)

    # Avoid division by zero
    max_conc[max_conc == 0] = 1e-8

    conc *= (max_conc_ref / max_conc)[:, np.newaxis]

    # Reconstruct normalized image
    norm_img = Io * np.exp(-np.dot(stain_ref, conc))
    norm_img = np.clip(norm_img, 0, 255)
    norm_img = norm_img.T.reshape((h, w, 3)).astype(np.uint8)

    return norm_img


# Wrapper for use with torchvision.transforms
class MacenkoNormalize(object):
    def __call__(self, img):
        # Convert from PIL Image to NumPy array
        np_img = np.array(img)
        # Apply normalization
        norm_img = normalize_macenko(np_img)
        # Convert back to PIL Image
        return Image.fromarray(norm_img)
