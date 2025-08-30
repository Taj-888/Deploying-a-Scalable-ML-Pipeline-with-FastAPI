import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # simple baseline
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train.ravel() if y_train.ndim > 1 else y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model:Any, X: np.ndarray)-> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function
    return model.predict(X)

def save_model(model: Any, path: str)-> None:
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """

    p= Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str) -> Any:
    """ Loads pickle file from `path` and returns it."""
    # TO DO: implement the function
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
        df,
        column_name: str,
        slice_value,
        *,
        encoder,
        lb,
        model,
        categorical_features,
        label: str = "salary",
) -> Tuple[float, float, float]:
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function

    slice_df = df[df[column_name] == slice_value].copy()
    if slice_df.empty:
        return float("nan"), float("nan"), float("nan")

    # Process with the trained encoder/label binarizer
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,  # <-- string label, not lb
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice) # your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
