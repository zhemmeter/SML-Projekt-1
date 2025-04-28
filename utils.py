"""Utility functions for project 1."""
import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler



IMAGE_SIZE = (300, 300)



def load_config():
    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = Path(config["data_dir"])

    if config["load_rgb"] is None or config["downsample_factor"] is None:
        raise NotImplementedError("Make sure to set load_rgb and downsample_factor!")

    print(f"[INFO]: Configs are loaded with: \n {config}")
    return config

# DYI: load best parameters from yaml file
def load_params():
    with open("./best_params.yaml", "r") as file:
        params = yaml.safe_load(file)

    return params

def load_dataset(config, split="train"):
    labels = pd.read_csv(
        config["data_dir"] / f"{split}_labels.csv", dtype={"ID": str}
    )

    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim
    images = np.zeros((len(labels), feature_dim))

    idx = 0
    for _, row in labels.iterrows():
        image = Image.open(
            config["data_dir"] / f"{split}_images" / f"{row['ID']}.png"
        )
        if not config["load_rgb"]:
            image = image.convert("L")
        image = image.resize(
            (
                IMAGE_SIZE[0] // config["downsample_factor"],
                IMAGE_SIZE[1] // config["downsample_factor"],
            ),
            resample=Image.BILINEAR,
        )
        image = np.asarray(image).reshape(-1)
        images[idx] = image
        idx += 1

    distances = labels["distance"].to_numpy()
    return images, distances

# DYI: create validation dataset, essentially equal to load_dataset()
def load_validation_dataset(config, split="val"):
    labels = pd.read_csv(
        config["data_dir"] / f"{split}_labels.csv", dtype={"ID": str}
    )

    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = np.zeros((len(labels), feature_dim))

    idx = 0
    for _, row in labels.iterrows():
        image = Image.open(
            config["data_dir"] / f"{split}_images" / f"{row['ID']}.png"
        )
        if not config["load_rgb"]:
            image = image.convert("L")
        image = image.resize(
            (
                IMAGE_SIZE[0] // config["downsample_factor"],
                IMAGE_SIZE[1] // config["downsample_factor"],
            ),
            resample=Image.BILINEAR,
        )
        image = np.asarray(image).reshape(-1)
        images[idx] = image
        idx += 1

    distances = labels["distance"].to_numpy()
    return images, distances

def load_test_dataset(config):
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = []
    img_root = os.path.join(config["data_dir"], "test_images")

    for img_file in sorted(os.listdir(img_root)):
        if img_file.endswith(".png"):
            image = Image.open(os.path.join(img_root, img_file))
            if not config["load_rgb"]:
                image = image.convert("L")
            image = image.resize(
                (
                    IMAGE_SIZE[0] // config["downsample_factor"],
                    IMAGE_SIZE[1] // config["downsample_factor"],
                ),
                resample=Image.BILINEAR,
            )
            image = np.asarray(image).reshape(-1)
            images.append(image)

    return images

def print_results(gt, pred):
    print(f"MAE: {round(mean_absolute_error(gt, pred)*100, 3)}")
    print(f"R2: {round(r2_score(gt, pred)*100, 3)}")
    #DYI: added Grade metrics according to FS25 course
    print(f"Grade: {grade(mean_absolute_error(gt, pred))}")

def save_results(pred):
    text = "ID,Distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open("prediction.csv", 'w') as f: 
        f.write(text)

# DYI: added grade interpolation function according to FS25 course 
def grade(error):
    if error <= 0.18:
        # Linear interpolation between 4 (at 0.18) and 6 (at 0.08)
        return 4 + (6 - 4) / (0.08 - 0.18) * (error - 0.18)
    elif error <= 0.35:
        # Linear interpolation between 1 (at 0.35) and 4 (at 0.18)
        return 1 + (4 - 1) / (0.18 - 0.35) * (error - 0.35)
    else:
        return 0

# DYI: added Randomsearch function 
def RS(regressor, iterations, images, distances):
    
    #create the random grid
    random_grid = {
        'regressor__n_neighbors': [1,2],
        'regressor__weights': ['distance'],
        'regressor__algorithm': ['kd_tree'],
        'regressor__leaf_size': [50, 55],
        'regressor__p': [1],
        'regressor__metric': ['manhattan'],
    }
    
    rdm_regr = RandomizedSearchCV(estimator = regressor, 
                                  param_distributions = random_grid, 
                                  n_iter = iterations, 
                                  cv = 3, 
                                  verbose = 3, 
                                  random_state = 42, 
                                  n_jobs = -1
                                  )
    
    rdm_regr.fit(images, distances)

    return rdm_regr.best_params_

# DYI: added Gridsearch function
def GS(regressor, images, distances):
    # Refined parameter grid
    param_grid = {
        'regressor__loss': ['squared_error'], 
        'regressor__learning_rate': [0.01, 0.02],  # Spread around 0.05
        'regressor__n_estimators': [600],  # Spread around 600
        'regressor__subsample': [0.7],  # Spread around 0.7
        'regressor__min_samples_split': [7],  # Spread around 6, 7, 8
        'regressor__min_samples_leaf': [2],  # Spread around 2
        'regressor__min_weight_fraction_leaf': [0.0],  # Keep as is
        'regressor__max_depth': [14],  # Spread around 14, 15, 16
        'regressor__min_impurity_decrease': [0.0],  # Keep as is
        'regressor__max_features': ['sqrt'],  # Add more options
        'regressor__alpha': [0.9999],  # Spread around 0.9999
        'regressor__max_leaf_nodes': [30]  # Spread around 25, 30, 35
    }

    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=2,
        verbose=3,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )

    grid_search.fit(images, distances)

    return grid_search.best_params_