import os
import sys

from cl_prediction.constants.training_testing_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
from cl_prediction.exception.exception import CLPredictionException
from cl_prediction.logging.logger import logging

class CLPredictionEstimator:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CLPredictionException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CLPredictionException(e,sys)