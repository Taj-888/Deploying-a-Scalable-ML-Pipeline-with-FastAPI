import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
#  --------- Paths ----------
# # Try to infer the project root so the script works from anywhere.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(THIS_DIR, "data", "census.csv"),
    os.path.join(THIS_DIR, "..", "data", "census.csv"),
    os.path.join(os.getcwd(), "data", "census.csv"),
]

data_path = None
for p in CANDIDATES:
    if os.path.exists(p):
        data_path = os.path.abspath(p)
        break

project_path = os.path.dirname(os.path.dirname(data_path))  # root that contains /data and /model

data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

# --------- 1) Load data ----------
data = pd.read_csv(data_path)

#--------- 2) Train/Test split ----------
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["salary"] if "salary" in data.columns else None,
)
# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# --------- 3) Process data ----------
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model,X_test) # your code here

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
slice_path = os.path.join(project_path, "slice_output.txt")
with open(slice_path, "w") as f:
    f.write("")  # clear
#iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            encoder=encoder,
            lb=lb,
            model=model,
            categorical_features=cat_features,
            label="salary",
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
