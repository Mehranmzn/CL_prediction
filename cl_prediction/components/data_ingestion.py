from cl_prediction.exception.exception import CLPredictionException
from cl_prediction.logging.logger import logging
from cl_prediction.entity.config_entity import DataIngestionConfig
from cl_prediction.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
from typing import List
from dotenv import load_dotenv
import snowflake.connector
from cl_prediction.utils.main_utils.utils import read_yaml_file
from cl_prediction.constants.training_testing_pipeline import SCHEMA_FILE_PATH




load_dotenv()



class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CLPredictionException(e, sys)

    def export_table_as_dataframe(self):
        """
        Read data from Snowflake table
        """
        try:
            # Snowflake connection details
            conn = snowflake.connector.connect(
                user=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PASSWORD"),
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                warehouse=self.data_ingestion_config.collection_name,
                database=self.data_ingestion_config.database_name,
                schema="RAW",  # The schema within the database
                role="transform",  # The role you granted
                ocsp_fail_open=True,
                insecure_mode=True  # Disable SSL verification for debugging

            )
            
            # SQL query to fetch data
            table_name = self.data_ingestion_config.table_name
            query = f"SELECT * FROM {table_name};"

            # Execute query and fetch data
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            columns = [col[0] for col in cursor.description]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            df["TSDATE"] = pd.to_datetime(df["TSDATE"])
            current_cols = df.columns
            schema = read_yaml_file(SCHEMA_FILE_PATH)

            target_columns = [column['name'] for column in schema['columns']]
            column_mapping = dict(zip(current_cols, target_columns))

            #rename columns
            df.rename(columns=column_mapping, inplace=True)

            # Close connection
            cursor.close()
            conn.close()
            
            return df

        except Exception as e:
            raise CLPredictionException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise CLPredictionException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            # Calculate the split index
            # Define date ranges
           
            # Split the data based on date ranges
            train_set = dataframe[
                (dataframe["TSDate"] >= self.data_ingestion_config.train_start_date) & (dataframe["TSDate"] <= self.data_ingestion_config.train_end_date)
            ]
            
            test_set = dataframe[
                (dataframe["TSDate"] >= self.data_ingestion_config.test_start_date) & (dataframe["TSDate"] <= self.data_ingestion_config.test_end_date)
            ]

            logging.info("Split the dataframe into train, validation, and test sets based on date ranges.")

            # Create the directory for saving the files if it doesn't exist
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train, validation, and test datasets to file paths.")

            # Save the train, validation, and test datasets to the specified file paths
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info("Successfully exported train, validation, and test datasets to file paths.")
        except Exception as e:
            raise CLPredictionException(e, sys)
       

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_table_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact

        except Exception as e:
            raise CLPredictionException(e, sys)