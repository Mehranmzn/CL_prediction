import os
import sys
from cl_prediction.exception.exception import CLPredictionException 
from cl_prediction.logging.logger import logging
from cl_prediction.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from cl_prediction.entity.config_entity import ModelTrainerConfig
from cl_prediction.utils.ml_utils.model.estimator import CLPredictionEstimator
from cl_prediction.utils.main_utils.utils import save_object,load_object
from cl_prediction.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from cl_prediction.utils.ml_utils.metric.classification_metrics import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from lightgbm import LGBMClassifier
from io import StringIO
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
from sklearn.metrics import classification_report
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from xgboost import XGBClassifier
import dagshub
dagshub.init(repo_owner='mehran1414', repo_name='CL_prediction', mlflow=True)




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CLPredictionException(e,sys)
        
    def track_mlflow(self, best_model, classification_metric, classification_report, model_name, aux_info, x1,x2):
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
            report_buf = StringIO()
            report_buf.write(classification_report)
            report_buf.seek(0)
            mlflow.log_text(report_buf.getvalue(), "classification_report.txt")
            report_buf.close()
            
            # Set tags
            mlflow.set_tag("model_name", model_name)
            for key, value in aux_info.items():
                mlflow.set_tag(key, value)
            
            # Model registry
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=model_name, signature=signature, input_example=x1[:5])
            else:
                mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=x1[:5])


        
    def train_model(self,X_train,y_train,x_test,y_test):
        class_weight = {0:1, 1:30}
        base_estimator = DecisionTreeClassifier(max_depth=50, class_weight=class_weight)
        models = {
                "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=30),
                "Easy Ensemble": EasyEnsembleClassifier(verbose=1),
                "Random Forest": RandomForestClassifier(verbose=1, class_weight="balanced"),
                "LightGBM": LGBMClassifier(random_state=42, class_weight="balanced"),
                "AdaBoost": AdaBoostClassifier(random_state=42, base_estimator=base_estimator),
               #"HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
               
            }
        params = {
                "Random Forest": {
                    "n_estimators": [ 300 ],
                    "max_depth": [ 50],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False]
                },
                "LightGBM": {
                    "n_estimators": [ 300],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [50],
                    "lambda_l1": [ 5],
                    'boosting_type': ['dart'],
                },
                "AdaBoost": {
                    "n_estimators": [300],
                    "learning_rate": [0.05,0.1],
                    "algorithm": ["SAMME", "SAMME.R"]
                },
                "Balanced Random Forest": {
                    "n_estimators": [300],
                    "max_depth": [50],
                    "bootstrap": [True, False]
                },
                "Easy Ensemble": {
                    "n_estimators": [10],
                },
                "XGBoost": {
                    "n_estimators": [300],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [50],
                    "min_child_weight": [1, 5],
                    "alpha": [10],
                    "scale_pos_weight": [30]
                },
            #      "HistGradientBoosting": {
            #                 'learning_rate': [0.05, 0.1],
            #                 'max_iter': [300],
            #                 'max_depth': [50],
            #                 'l2_regularization': [ 10],

            # }
        }
        model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                                       models=models, param=params)

        # To get best model score from dict
        best_model_name = None
        best_model_score = 0
        best_model_threshold = 0

        for model_name, metrics in model_report.items():
            if metrics['best_f1'] > best_model_score:
                best_model_score = metrics['best_f1']
                best_model_name = model_name
                best_model_threshold = metrics['best_threshold']

        # Retrieve the best model
        best_model = models[best_model_name]

        # Print the best model details
        print(f"Best Model: {best_model_name}")
        print(f"Best macro f1: {best_model_score:.4f} at Threshold: {best_model_threshold}")
          # Use the first 5 rows of your training data as an example

        #load the .pkkl model file from the path
        with open(f"final_model/trained_model_{best_model_name}.pkl", "rb") as f:
            best_model = pickle.load(f)

        y_train_pred=best_model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_pred > best_model_threshold).astype(int)

        #preds = best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        cs_report = classification_report(y_train,  y_train_pred)
        ## Auxiliary information
        aux_info = {
            "experiment_id": "exp_01",
            "dataset": "train",
            "description": "Experiment with calibration and Feature engineering ",
            "threhsold": best_model_threshold,
            
        }

        # Call the function
        self.track_mlflow(best_model=best_model,
                        classification_metric=classification_train_metric,
                        classification_report=cs_report,
                        model_name= model_name,
                        aux_info= aux_info,
                        x1 = X_train,
                        x2 = best_model.predict(X_train[:5]))

        

        y_test_pred=best_model.predict_proba(x_test)[:, 1]
        #preds = best_model.predict(x_test)
        y_test_pred = (y_test_pred > best_model_threshold).astype(int)

        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)


        cs_report_test = classification_report(y_test,  y_test_pred)
        ## Auxiliary information
        aux_info = {
            "experiment_id": "exp_01",
            "dataset": "test",
            "description": "Experiment Test with calibration and Feature engineering ",
            "threhsold": best_model_threshold,

        }

        # Call the function
        self.track_mlflow(best_model=best_model,
                        classification_metric=classification_test_metric,
                        classification_report=cs_report_test,
                        model_name= model_name,
                        aux_info= aux_info,
                        x1 = x_test,
                        x2 = best_model.predict(x_test[:5]))
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        CL_Model=CLPredictionEstimator(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=CL_Model)
        #model pusher
        save_object("final_model/final_model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise CLPredictionException(e,sys)