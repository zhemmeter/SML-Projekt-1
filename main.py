from utils import load_config, load_dataset, load_test_dataset, load_validation_dataset, print_results, save_results, grade, RS, GS, load_params

# sklearn imports...

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


#def isincenter():
    #return

#def isobject():
    #return
    

#def splitrange():
    #return


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
    pipe = Pipeline([('standardscaler', StandardScaler()),
                     ('regressor',RandomForestRegressor())])

    #Random Forest best parameters (RandomSearchCV)
    params = load_params()
    
    for key, value in params.items():
        pipe.set_params(**{f"regressor__{key}": value})

    print(f"[INFO]: Model is fitted with the following parameters: {params}")
    
    pipe.fit(images,distances)
    
    
    print("Model Trained")
    pred= pipe.predict(val_im)
    print("Model Predicted")
    print(f"MAE: {mean_absolute_error(val_d, pred)} Grade: {grade(mean_absolute_error(val_d, pred))}")

    
    # Save the results
    #prv_images = load_private_test_dataset(config)
    #save_results(model.predict(prv_images))
    
    # save_results(private_test_pred)

def train():
    # Load configs from "config.yaml"
    config = load_config()
    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    # Define the Regressor
    regressor = RandomForestRegressor(random_state = 42)
    

    # Call the RS to perform tuning
    best_params = GS(regressor, images, distances)

    # Print the best parameters
    print(f"Best parameters: {best_params}")
    
    # Save the best parameters to a file
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
   main()


#covariance: matrix for multidimensional parameters :-> r,g,b | distance
#linear model of parameters and unknown variance -> estimate of distance

#look at Ordinary least-squares errors and maximum likelihood estimator again