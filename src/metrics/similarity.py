"""
Similarity metrics for image registration.
"""

import numpy as np


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Normalized Cross Correlation.
    """
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return float(np.mean(img1_norm * img2_norm))


def compute_mi(img1: np.ndarray, img2: np.ndarray, bins: int = 50) -> float:
    """
    Compute Mutual Information using histogram binning.
    """
    img1_f = img1.flatten()
    img2_f = img2.flatten()

    hist_2d, _, _ = np.histogram2d(img1_f, img2_f, bins=bins)
    p_xy = hist_2d / (np.sum(hist_2d) + 1e-8)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
    h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
    h_xy = -np.sum(p_xy[p_xy > 0] * np.log2(p_xy[p_xy > 0]))

    return float(h_x + h_y - h_xy)
