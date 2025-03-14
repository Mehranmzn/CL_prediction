import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from cl_prediction.constants.training_testing_pipeline import (
    TARGET_COLUMN, FEATURES, DATA_GROUPING_COLUMN, 
    DATA_DATE_COLUMN,
    HIGHLY_CORRELATED_FEATURES,
    LOG_FEATURES,
    CAP_FEATURES, SCHEMA_FILE_PATH
)
from cl_prediction.constants.training_testing_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from cl_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from cl_prediction.entity.config_entity import DataTransformationConfig
from cl_prediction.exception.exception import CLPredictionException 
from cl_prediction.logging.logger import logging
from cl_prediction.utils.main_utils.utils import save_numpy_array_data,save_object, read_yaml_file

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise CLPredictionException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CLPredictionException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Initialize the data transformer pipeline with KNNImputer and OneHotEncoder for categorical columns.
        """
        logging.info(
            "Entered get_data_transformer_object method of Transformation class"
        )
        try:
           # Initialize the 
            imputer = SimpleImputer(strategy='median')
            #or KNNImputer
            # imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            
            logging.info(
                f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            schema = read_yaml_file(SCHEMA_FILE_PATH)

            numerical_columns = [
                column["name"]
                for column in schema["columns"]
                if column["type"] in ["INTEGER", "FLOAT"]
            ]

            # Combine the transformations using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[

                    ("imputer", imputer, numerical_columns),  # Apply imputer to all numeric columns
                ],
                remainder="passthrough"  # Keep the remaining columns as they are
            )
            return Pipeline(steps=[('preprocessor', preprocessor)])
        except Exception as e:
            raise CLPredictionException(e, sys)

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(
                columns=[DATA_GROUPING_COLUMN,TARGET_COLUMN],
                axis=1
            )
            
            target_feature_train_df = train_df[TARGET_COLUMN]

            #testing dataframe
            input_feature_test_df = test_df.drop(
                columns=[DATA_GROUPING_COLUMN, TARGET_COLUMN], 
                axis=1
            )
            target_feature_test_df = test_df[TARGET_COLUMN]


                      
            preprocessor=self.get_data_transformer_object()

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

             
            # Convert transformed data back to DataFrame for column manipulation
            train_transformed_df = pd.DataFrame(transformed_input_train_feature, columns=train_df.drop(columns=[TARGET_COLUMN]).columns)
            test_transformed_df = pd.DataFrame(transformed_input_test_feature, columns=test_df.drop(columns=[TARGET_COLUMN]).columns)

            # Remove unwanted columns
            train_transformed_df = train_transformed_df.drop(columns=[DATA_DATE_COLUMN, DATA_GROUPING_COLUMN, DATA_DATE_COLUMN, HIGHLY_CORRELATED_FEATURES], errors="ignore")
            test_transformed_df = test_transformed_df.drop(columns=[DATA_DATE_COLUMN, DATA_GROUPING_COLUMN, DATA_DATE_COLUMN, HIGHLY_CORRELATED_FEATURES], errors="ignore")

            # Convert to numpy arrays and append target column
            train_arr = np.c_[train_transformed_df.to_numpy(), target_feature_train_df.to_numpy()]
            test_arr = np.c_[test_transformed_df.to_numpy(), target_feature_test_df.to_numpy()]

            # train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            # test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise CLPredictionException(e,sys)
