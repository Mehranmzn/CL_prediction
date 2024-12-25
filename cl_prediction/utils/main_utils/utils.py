import yaml
import os,sys
import numpy as np  
import pandas as pd
from cl_prediction.exception.exception import CLPredictionException
from cl_prediction.logging.logger import logging
import pickle
from sklearn.metrics import mean_squared_error, accuracy_score
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
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates classification models using GridSearchCV and computes evaluation metrics.

    Args:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - models: Dictionary of models to evaluate.
    - param: Dictionary of hyperparameters for each model.

    Returns:
    - report: Dictionary with model names and their test accuracies.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            # Use GridSearchCV to find the best parameters
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring="accuracy", verbose=1)
            grid_search.fit(X_train, y_train)

            # Get the best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters for {model_name}: {best_params}")

            # Train the final model using the best parameters
            if model_name == 'LightGBM':
                final_model = lgb.LGBMClassifier(**best_params)
            elif model_name == 'XGBoost':
                final_model = xgb.XGBClassifier(**best_params)
            elif model_name == 'CatBoost':
                final_model = cb.CatBoostClassifier(**best_params, verbose=0)
            else:
                final_model = model.set_params(**best_params)

            final_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = final_model.predict(X_train)
            y_test_pred = final_model.predict(X_test)

            # Calculate evaluation metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            print(f"{model_name} Train Accuracy: {train_accuracy:.4f}")
            print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")

            # Store the test accuracy in the report
            report[model_name] = test_accuracy

        print("Evaluation completed")
        return report

    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, data):
        self.data = data.copy()
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
        data = self.data.sort_values(by=[group_col, time_col])

        # Rolling window features
        for feature in rolling_features:
            data[f'{feature}_rolling_mean_{window}'] = data.groupby(group_col)[feature].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            data[f'{feature}_rolling_std_{window}'] = data.groupby(group_col)[feature].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            data[f'{feature}_rolling_sum_{window}'] = data.groupby(group_col)[feature].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )

        # Recency feature: Check if there was a credit application in the past 6 months (excluding the current month)
        def calculate_past_6_months_flag_excluding_current(x):
            return x.shift().rolling(window=6, min_periods=1).max()

        data['recency_6_months'] = data.groupby(group_col)['credit_application'].transform(
            calculate_past_6_months_flag_excluding_current
        )

        # Number of credit applications in the past 6 months (excluding the current month)
        def calculate_past_6_months_count_excluding_current(x):
            return x.shift().rolling(window=6, min_periods=1).sum()

        data['nr_applications_6_months'] = data.groupby(group_col)['nr_credit_applications'].transform(
            calculate_past_6_months_count_excluding_current
        )

        # Handle missing values with backward fill and median replacement
        data = data.groupby(group_col).apply(lambda group: group.fillna(method='bfill'))
        data.fillna(data.median(), inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

    def add_lagged_and_statistical_features(self, group_col, features):
        """
        Adds lagged features (1 to 6 months) and min, max, mean, and std for the last 2 to 6 months.

        Args:
        - group_col (str): Column for grouping (e.g., client_nr).
        - features (list): List of features to generate lagged and statistical features.

        Returns:
        - DataFrame with lagged and statistical features added.
        """
        data = self.data

        # Lagged features from 1 to 6 months
        for lag in range(1, 7):
            for feature in features:
                data[f'{feature}_lag_{lag}'] = data.groupby(group_col)[feature].shift(lag)

        # Min, Max, Mean, and Std for the previous 2, 3, 4, 5, 6 months
        for window_size in range(2, 7):
            for feature in features:
                data[f'{feature}_min_last_{window_size}'] = data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).min()
                )
                data[f'{feature}_max_last_{window_size}'] = data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).max()
                )
                data[f'{feature}_mean_last_{window_size}'] = data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).mean()
                )
                data[f'{feature}_std_last_{window_size}'] = data.groupby(group_col)[feature].transform(
                    lambda x: x.shift().rolling(window=window_size, min_periods=1).std()
                )

        # Handle missing values with backward fill and median replacement
        data = data.groupby(group_col).apply(lambda group: group.fillna(method='bfill'))
        data.fillna(data.median(), inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

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
        return self.dataframe
    


 
