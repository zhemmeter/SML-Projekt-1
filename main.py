from utils import load_config, load_dataset, load_test_dataset, load_validation_dataset, print_results, save_results, grade, RS, GS, load_params
from processing import  filter_outliers_zscore, augment_noise, crop_center, extract_lbp_features

# sklearn imports...

import numpy as np
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel

def main():
    kaggle = input("[PROMPT]: Kaggle Mode? (Y/N)")

    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    # Load test dataset for validation
    val_im, val_d = load_validation_dataset(config)
    print(f"[INFO]: Validation dataset loaded with {len(val_im)} samples.")

    
    if kaggle == "Y":
        # Combine datasets for hand-in
        images = np.concatenate((images, val_im), axis=0)
        distances = np.concatenate((distances, val_d), axis=0)

        # Load test dataset for hand-in
        test_im = load_test_dataset(config)
        print(f"[INFO]: Test dataset loaded with {len(test_im)} samples.")


    # Preprocessing pipeline
    pipe = Pipeline([
        ('standardscaler', StandardScaler()),
        ('minmaxscaler', MinMaxScaler()),
        ('regressor', KNeighborsRegressor())  # Use VotingRegressor as the final step
    ])

    # Parameter Matrix
    params = load_params()
    for key, value in params.items():
        print(f"[INFO]: Setting parameter: {key} = {value}")
        pipe.set_params(**{key : value})        

    # Filter dataset
    images, distances = filter_outliers_zscore(images, distances, threshold=3)
    print(f"[INFO]: Dataset after outlier removal: {len(images)} samples.")

    # Model Fitting
    pipe.fit(images, distances)

    
    if kaggle == "Y":
        # Private Set Prediction
        pred = pipe.predict(test_im)
        # Save predictions to CSV
        save_results(pred)


    
    else:
        # Validation Set Prediction
        pred = pipe.predict(val_im)

        # Save predictions, ground truth, and differences to CSV
        results_df = pd.DataFrame({
            'Ground Truth': val_d,
            'Predictions': pred,
            'Difference': np.abs(val_d - pred)
        })

        # Plot of the differences
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['Difference'], kde=True, bins=30, color='blue')
        plt.title("Verteilung der Differenzen zwischen Ground Truth und Predictions", fontsize=16)
        plt.xlabel("Differenz", fontsize=14)
        plt.ylabel("Häufigkeit", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Print MAE & grade approximation
        print_results(val_d, pred)


def train():
    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # Define the base regressors
    knn = KNeighborsRegressor()
    #rf = RandomForestRegressor(random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)

    # Define the VotingRegressor
    voting_regressor = VotingRegressor([
        ('knn', knn),
        #('rf', rf),
        ('gbr', gbr)
    ])

    # Preprocessing pipeline
    pipe = Pipeline([
        ('standardscaler', StandardScaler()),
        ('minmaxscaler', MinMaxScaler()),
        ('regressor', voting_regressor)  # Use VotingRegressor as the final step
    ])

    # Filter dataset
    images, distances = filter_outliers_zscore(images, distances, threshold=3)
    print(f"[INFO]: Dataset after outlier removal: {len(images)} samples.")
    # Call the RS to perform tuning
    best_params = RS(pipe, 500, images, distances)

    # Print the best parameters
    print(f"Best parameters: {best_params}")
    
    # Save the best parameters to a file
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
    if input("[PROMPT]: Hyperparameter Search? (Y/N)") == "Y":
        train()
    main()

