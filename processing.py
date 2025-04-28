import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from scipy.stats import zscore

from skimage.feature import hog, canny, local_binary_pattern
from skimage.transform import rotate
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin


# DYI: added hog feature extraction function
def hog_extract(images):
    hog_feature_list = []
    for image in images:
        image = image.reshape(int(len(image)**(0.5)), int(len(image)**(0.5)))
        # HOG features
        hog_feature = hog(image, pixels_per_cell=(16, 16), cells_per_block=(4, 4))
        hog_feature_list.append(hog_feature)
    
    return np.array(hog_feature_list)

# DYI: added canny feature extraction function
def canny_extract(images):
    canny_feature_list = []
    for image in images:
        image = image.reshape(int(len(image)**(0.5)), int(len(image)**(0.5)))

        # Canny features
        canny_edges = canny(image)
        edge_density = np.sum(canny_edges) / canny_edges.size
        canny_feature = edge_density.flatten()
        canny_feature_list.append(canny_feature)
    
    return np.array(canny_feature_list)


# DYI: added function to crop the center of the image
def crop_center(image, crop_size=(150, 150)):
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_crop_x, half_crop_y = crop_size[1] // 2, crop_size[0] // 2
    return image[center_y - half_crop_y:center_y + half_crop_y, center_x - half_crop_x:center_x + half_crop_x]

# DYI: added function to change the noise of the image
def augment_noise(image, mean=0, std=0.01):
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 255)

# DYI: added function to extract LBP features
def extract_lbp_features(images, P=8, R=1):
    lbp_features = []
    for image in images:
        image = image.reshape(int(len(image)**0.5), int(len(image)**0.5))  # Reshape to 2D
        lbp = local_binary_pattern(image, P=P, R=R, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize histogram
        lbp_features.append(lbp_hist)
    return np.array(lbp_features)

#DYI: added function to filter outliers based on Z-Score
def filter_outliers_zscore(features, targets, threshold=3):

    z_scores = zscore(targets)
    mask = np.abs(z_scores) < threshold
    return features[mask], targets[mask]