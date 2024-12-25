from cl_prediction.entity.artifact_entity import ClassificationMetricArtifact
from cl_prediction.exception.exception import CLPredictionException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import numpy as np

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics and return a ClassificationMetricArtifact object.

    Args:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - ClassificationMetricArtifact: Object containing accuracy, precision, recall, and F1-score.
    """
    try:
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Create a ClassificationMetricArtifact object (assuming it exists in your codebase)
        classification_metric = ClassificationMetricArtifact(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
        )
        return classification_metric

    except Exception as e:
        raise Exception(f"An error occurred while calculating classification metrics: {str(e)}")

