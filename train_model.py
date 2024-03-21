import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
model = RandomForestClassifier()
# TODO: load the census.csv data
project_path = "/home/nlaffra/ml-pipeline-reset"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data,
                               test_size=0.20
                              )
print("LINE 26: ", end ="|")
print(train)
print("|")
print("LINE 29: ", end ="|")
print(test)
print("|")

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

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    X = train,
    categorical_features=cat_features,
    label="salary",
    training=True
    )

print("Train parameters: ", end="|")
print("X_train: ")
print(X_train)
print(", y_train: ")
print(y_train)
print(", encoder: ")
print(encoder)
print(", lb: ")
print(lb)
print("|")

X_test, y_test, _, _ = process_data(
    X = test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

print("Test parameters: ", end="|")
print("X_test: ")
print(X_test)
print(", y_test: ")
print(y_test)
print("|")

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

print("model: ", end="|")
print(model)
print("|")

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
print("model_path: ", end="|")
print(model_path)
print("|")

save_model(model, model_path)

print("model (updated): ", end="|")
print(model)
print("|")

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
print("encoder_path: ", end="|")
print(encoder_path)
print("|")

save_model(encoder, encoder_path)

print("encoder (updated): ", end="|")
print(encoder)
print("|")

# load the model
load_model(model_path)

print("model (loaded): ", end="|")
print(model)
print("|")

print("X_test again")
print(X_test)

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

print("preds: ", end="|")
print(preds)
print("|")

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data = test, 
            column_name = col, 
            slice_value = slicevalue,
            categorical_features = cat_features,
            label = "salary",
            encoder = encoder,
            lb = lb, 
            model = model
            training = False
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
