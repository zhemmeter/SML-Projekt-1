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

IMAGE_SIZE = (300, 300)


def load_config():
    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = Path(config["data_dir"])

    if config["load_rgb"] is None or config["downsample_factor"] is None:
        raise NotImplementedError("Make sure to set load_rgb and downsample_factor!")

    print(f"[INFO]: Configs are loaded with: \n {config}")
    return config

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
    print(len(labels))
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


def save_results(pred):
    text = "ID,Distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open("prediction.csv", 'w') as f: 
        f.write(text)
        
def grade(error):
    if error <= 0.18:
        # Linear interpolation between 4 (at 0.18) and 6 (at 0.08)
        return 4 + (6 - 4) / (0.08 - 0.18) * (error - 0.18)
    elif error <= 0.35:
        # Linear interpolation between 1 (at 0.35) and 4 (at 0.18)
        return 1 + (4 - 1) / (0.18 - 0.35) * (error - 0.35)
    else:
        return 0
    
def RS(regressor, iterations, images, distances):
    
    #number of trees
    n_estimators = [int(x) for x in np.linspace(start = 400, stop = 500, num = 10)]
    # Criterion to measure the quality of a split
    criterion = ['squared_error']
    #number of features to consider at every split
    max_features = ['log2', 'sqrt', None, 40, 50, 60, 80]
    #maximum number levels in a tree
    max_depth = [int(x) for x in np.linspace(30,60,num=10)]
    max_depth.append(None)
    #minimum number of samples required to split a node
    min_samples_split = [2]
    #minimum number of samples required at each leaf node
    min_samples_leaf = [1]
    #method of selecting samples
    bootstrap = [False]

    #create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    rdm_regr = RandomizedSearchCV(estimator = regressor, 
                                  param_distributions = random_grid, 
                                  n_iter = iterations, 
                                  cv = 3, 
                                  verbose = 1, 
                                  random_state = 42, 
                                  n_jobs = -1
                                  )
    
    rdm_regr.fit(images, distances)

    return rdm_regr.best_params_

def GS(regressor, images, distances):
    # Refined parameter grid
    param_grid = {
        'n_estimators': [474],
        'criterion': ['squared_error'],
        'max_features': [60],
        'max_depth': [30],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [False]
    }

    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=param_grid,
                               cv=3,
                               verbose=1,
                               n_jobs=-1,
                               scoring='neg_mean_absolute_error')

    grid_search.fit(images, distances)

    return grid_search.best_params_