from utils import load_config, load_dataset, load_test_dataset, load_validation_dataset, print_results, save_results, grade, RS, GS, load_params, hog_extract, canny_extract

# sklearn imports...

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel

def main():
    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    # Load test dataset for validation
    val_im, val_d = load_validation_dataset(config)
    print(f"[INFO]: Validation dataset loaded with {len(val_im)} samples.")



    # preprocessing
    pipe = Pipeline([
        ('standardscaler', StandardScaler()),
        ('minmaxscaler', MinMaxScaler()),
        ('selector', SelectFromModel(GradientBoostingRegressor())),
        ('regressor', GradientBoostingRegressor())
    ])

    # Parameter Matrix
    params = load_params()
    for key, value in params.items():
        pipe.set_params(**{f"selector__estimator__{key}": value})
        pipe.set_params(**{f"regressor__{key}": value})    
    
    # Feature Selection
    images_features = np.hstack((hog_extract(images), canny_extract(images)))
    val_im_features = np.hstack((hog_extract(val_im), canny_extract(val_im)))

    # Model Fitting
    pipe.fit(images_features,distances)
    pred = pipe.predict(val_im_features)

    print_results(val_d, pred)


def train():
    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    # Define the Regressor
    pipe = Pipeline([
        ('standardscaler', StandardScaler()),
        ('minmaxscaler', MinMaxScaler()),
        ('selector', SelectFromModel(GradientBoostingRegressor())),
        ('regressor', GradientBoostingRegressor())
    ])
    
    # Feature Selection
    images_features = np.hstack((hog_extract(images), canny_extract(images)))

    # Set selector parameters
    pipe.set_params(selector__estimator__n_estimators=550)
    pipe.set_params(selector__estimator__max_depth=14)
    pipe.set_params(selector__estimator__min_samples_split=7)
    pipe.set_params(selector__estimator__min_samples_leaf=2)
    pipe.set_params(selector__estimator__max_features='sqrt')
    pipe.set_params(selector__estimator__loss='squared_error')
    pipe.set_params(selector__estimator__learning_rate=0.05)
    pipe.set_params(selector__estimator__subsample=0.7)
    pipe.set_params(selector__estimator__min_weight_fraction_leaf=0.0)
    pipe.set_params(selector__estimator__min_impurity_decrease=0.0)
    pipe.set_params(selector__estimator__alpha=0.9999)
    pipe.set_params(selector__estimator__max_leaf_nodes=35)


    # Call the RS to perform tuning
    best_params = GS(pipe, images, distances)

    # Print the best parameters
    print(f"Best parameters: {best_params}")
    
    # Save the best parameters to a file
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
    # train()
    main()

