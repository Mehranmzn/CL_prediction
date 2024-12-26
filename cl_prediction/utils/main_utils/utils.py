import yaml
import os,sys
import numpy as np  
import pandas as pd
from cl_prediction.exception.exception import CLPredictionException
from cl_prediction.logging.logger import logging
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from cl_prediction.constants.training_testing_pipeline import (
    DATA_WINDOW, 
    ROLLING_FEATURES,
      CUMULATIVE_FEATURES, 
      DATA_GROUPING_COLUMN,
      CAP_FEATURES
)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import BaseEstimator, TransformerMixin
from cl_prediction.utils.ml_utils.metric.classification_metrics import get_classification_score
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import json


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CLPredictionException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CLPredictionException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CLPredictionException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CLPredictionException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CLPredictionException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj,  allow_pickle=True)
    except Exception as e:
        raise CLPredictionException(e, sys) from e
    


def sub_track_mlflow(best_model, classification_metric, classification_report, model_name, aux_info, x1,x2):
        mlflow.set_registry_uri("https://dagshub.com/mehran1414/CL_prediction.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # Log metrics
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision)
            mlflow.log_metric("recall", classification_metric.recall)
            signature = infer_signature(x1, x2)

            mlflow.sklearn.log_model(best_model, "model_name", signature=signature, input_example=x1[:5])

            # Log model
            
            # Save classification report as a text artifact
            report_json = json.dumps(classification_report, indent=2)
            mlflow.log_text(report_json, "classification_report.json")
            
            # Set tags
            mlflow.set_tag("model_name", model_name)
            for key, value in aux_info.items():
                mlflow.set_tag(key, value)
            
            # Model registry
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=model_name, signature=signature, input_example=x1[:5])
            else:
                mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=x1[:5])


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates classification models using GridSearchCV, computes evaluation metrics, and applies calibration.

    Args:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - models: Dictionary of models to evaluate.
    - param: Dictionary of hyperparameters for each model.

    Returns:
    - report: Dictionary with model names and their best test accuracy and corresponding threshold.
    - classification_reports: Dictionary with the classification report for the best threshold for each model.
    """
  

    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            # Use GridSearchCV to find the best parameters
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring="f1", verbose=1)
            grid_search.fit(X_train, y_train)

            # Get the best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters for {model_name}: {best_params}")

            # Train the final model using the best parameters
            final_model = model.set_params(**best_params)
            final_model.fit(X_train, y_train)

            # # Wrap the model with calibration
            # calibrated_model = CalibratedClassifierCV(base_estimator=final_model)
            # calibrated_model.fit(X_test, y_test)

            # Evaluate across multiple thresholds
            best_f1= 0
            best_threshold = 0
            best_report = None
            train_preds = None

            for threshold in [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                # Predict calibrated probabilities
                y_pred_proba_calibrated = final_model.predict_proba(X_test)[:, 1]
                
                # Adjust the prediction threshold
                y_pred = (y_pred_proba_calibrated > threshold).astype(int)

                f1 = f1_score(y_test, y_pred, average='macro')
                print(f"{model_name} Macro F1 score at threshold {threshold}: {f1:.4f}")

                # Store the best F1 score and corresponding threshold
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_report = classification_report(y_test, y_pred, output_dict=True)
                    train_preds = get_classification_score(y_true=y_test,y_pred=y_pred)

            print(f"{model_name} Best Macro F1 score: {best_f1:.4f} at Threshold: {best_threshold}")

            save_object(f"final_model/trained_model_{model_name}.pkl", final_model)

        
            ## Auxiliary information
            aux_info = {
                "experiment_id": "sub_exp",
                "dataset": "TEST",
                "description": "Experiment Test with diff models and diff threhsolds on test data",
                "threhsold": best_threshold,

            }

            # Call the function
            sub_track_mlflow(best_model=final_model,
                            classification_metric=train_preds,
                            classification_report=best_report,
                            model_name= model_name,
                            aux_info= aux_info,
                            x1 = X_test,
                            x2 = final_model.predict(X_test[:5]))


            # Store the results
            report[model_name] = {
                "best_f1": best_f1,
                "best_threshold": best_threshold
            }
            #classification_reports[model_name] = best_report

        print("Evaluation completed")
        return report#, classification_reports

    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")



class FeatureEngineering():
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.group_col = 'client_nr'
        self.cap_features = CAP_FEATURES

    def log_transform(self, log_features):

        for col in log_features:
            self.data[col] = np.log1p(self.data[col])  # Use log(1+x) to avoid log(0)

        for col in self.cap_features:
            self.data[col] = self.data[col].clip(upper=self.data[col].quantile(0.99))


        self.data['transaction_balance_ratio'] = self.data['total_nr_trx'] / self.data['max_balance']
        self.data['debit_credit_ratio'] = self.data['nr_debit_trx'] / self.data['nr_credit_trx']

        # Create a binary flag for negative balances
        self.data['min_balance_negative_flag'] = (self.data['min_balance'] < 0).astype(int)

        # Take the absolute value for transformation
        self.data['min_balance_abs'] = self.data['min_balance'].abs()

        # Apply log1p transformation to the absolute value
        self.data['min_balance_abs'] = np.log1p(self.data['min_balance_abs'])


        # Create a binary flag for negative balances
        self.data['max_balance_negative_flag'] = (self.data['max_balance'] < 0).astype(int)

        # Take the absolute value for transformation
        self.data['max_balance_abs'] = self.data['max_balance'].abs()

        # Apply log1p transformation to the absolute value
        self.data['max_balance_abs'] = np.log1p(self.data['max_balance_abs'])


    def crg_impute(self):
        # Identify customers with all CRG missing
        clients_with_all_na_crg = (
            self.data.groupby(self.group_col)['CRG']
            .apply(lambda x: x.isna().all())
            .reset_index()
        )
        clients_with_all_na_crg = clients_with_all_na_crg[clients_with_all_na_crg['CRG'] == True][self.group_col].values

        # Create a reference dataset for customers with valid CRG
        valid_customers = (
            self.data[~self.data['CRG'].isna()]
            .groupby(self.group_col)
            .median()
            .reset_index()
        )

        # Impute missing CRG values for customers with all NaNs
        for client in clients_with_all_na_crg:
            # Extract the median `volume_debit_trx` for this client
            client_data = (
                self.data[self.data[self.group_col] == client][['volume_debit_trx']]
                .median()
                .values
                .reshape(1, -1)
            )

            # Compute distances to all valid customers
            valid_data = valid_customers[['volume_debit_trx']].values
            distances = euclidean_distances(client_data, valid_data).flatten()

            # Find the nearest neighbor
            nearest_idx = np.argmin(distances)
            nearest_client = valid_customers.iloc[nearest_idx][self.group_col]

            # Impute CRG using the nearest customer's median CRG
            nearest_crg = (
                self.data[self.data[self.group_col] == nearest_client]['CRG']
                .median()
            )
            self.data.loc[self.data[self.group_col] == client, 'CRG'] = nearest_crg
            
    def crg_classification(self):
        # Interaction Features with CRG
        self.data['CRG_total_nr_trx'] = self.data['CRG'] * self.data['total_nr_trx']
        self.data['CRG_nr_debit_trx'] = self.data['CRG'] * self.data['nr_debit_trx']
        self.data['CRG_nr_credit_trx'] = self.data['CRG'] * self.data['nr_credit_trx']
        self.data['CRG_volume_debit_trx'] = self.data['CRG'] * self.data['volume_debit_trx']
        self.data['CRG_volume_credit_trx'] = self.data['CRG'] * self.data['volume_credit_trx']

        # Binning CRG into categories (example: quartiles)
        # Dynamically adjust labels based on the number of bins created
        labels = ['Low', 'Medium-Low', 'Medium-High', 'High']  # Define initial labels
        self.data['CRG_binned'] = pd.qcut(
            self.data['CRG'], 
            q=5,  # Initial number of quantiles
            labels=labels[:4],  # Ensure labels are one fewer than the resulting bins
            duplicates='drop'  # Drop duplicate edges
        )


        # Optional: Convert the binned CRG feature into dummy variables
        self.data = pd.get_dummies(self.data, columns=['CRG_binned'], drop_first=False)
        

        

    def create_features(self, group_col, time_col, rolling_features, window=3):
        """
        Creates rolling window features, recency, and cumulative features for the dataset.

        Args:
        - group_col (str): Column for grouping (e.g., client_nr).
        - time_col (str): Column representing time (e.g., yearmonth).
        - rolling_features (list): List of features to compute rolling statistics.
        - cumulative_features (list): List of features to compute cumulative sums.
        - window (int): Size of the rolling window.

        Returns:
        - DataFrame with new rolling, recency, and cumulative features.
        """
        self.data = self.data.sort_values(by=[group_col, time_col])

        # Rolling window features
        for feature in rolling_features:
            self.data[f'{feature}_rolling_mean_{window}'] = self.data.groupby(group_col)[feature].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            self.data[f'{feature}_rolling_std_{window}'] = self.data.groupby(group_col)[feature].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            self.data[f'{feature}_rolling_sum_{window}'] = self.data.groupby(group_col)[feature].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )

        # Recency feature: Check if there was a credit application in the past 6 months (excluding the current month)
        def calculate_past_6_months_flag_excluding_current(x):
            return x.shift().rolling(window=6, min_periods=1).max()

        self.data['recency_6_months'] = self.data.groupby(group_col)['credit_application'].transform(
            calculate_past_6_months_flag_excluding_current
        )

        # Number of credit applications in the past 6 months (excluding the current month)
        def calculate_past_6_months_count_excluding_current(x):
            return x.shift().rolling(window=6, min_periods=1).sum()

        self.data['nr_applications_6_months'] = self.data.groupby(group_col)['nr_credit_applications'].transform(
            calculate_past_6_months_count_excluding_current
        )

        # Handle missing values with backward fill and median replacement
        self.data = self.data.groupby(group_col).apply(lambda group: group.fillna(method='bfill'))
        self.data.fillna(self.data.median(), inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        return self.data

    def add_lagged_and_statistical_features(self, group_col, features):
        """
        Adds lagged features (1 to 6 months) and min, max, mean, and std for the last 2 to 6 months.

        Args:
        - group_col (str): Column for grouping (e.g., client_nr).
        - features (list): List of features to generate lagged and statistical features.

        Returns:
        - DataFrame with lagged and statistical features added.
        """
        # Lagged features from 1 to 6 months
        for lag in range(1, 7):
            for feature in features:
                self.data[f'{feature}_lag_{lag}'] = self.data.groupby(group_col)[feature].shift(lag)

        # Min, Max, Mean, and Std for the previous 2, 3, 4, 5, 6 months
        for window_size in range(2, 7):
            for feature in features:
                self.data[f'{feature}_min_last_{window_size}'] = self.data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).min()
                )
                self.data[f'{feature}_max_last_{window_size}'] = self.data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).max()
                )
                self.data[f'{feature}_mean_last_{window_size}'] = self.data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).mean()
                )
                self.data[f'{feature}_std_last_{window_size}'] = self.data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).std()
                )

        # Handle missing values with backward fill and median replacement
        self.data = self.data.groupby(group_col).apply(lambda group: group.fillna(method='bfill'))
        self.data.fillna(self.data.median(), inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        return self.data

    def generate_features(self, date_column: str, group_column: str,rolling_features: list ,  log_features: list ,features: list):
        self.log_transform(log_features = log_features)
        self.crg_impute()
        self.crg_classification()
        self.create_features(
            group_col=group_column,
            time_col=date_column,
            rolling_features= rolling_features,
            window=DATA_WINDOW
        )
        self.add_lagged_and_statistical_features(
            group_col=group_column,
            features= features
        )


    def get_dataframe(self):
        return self.data
    


 
