import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load data
# data = pd.read_csv("path/to/data.csv")

# 修改为本地数据文件
data = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_229.csv")

X = data.drop("target_column", axis=1)
y = data["target_column"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(1)
])

# Compile model
model.compile(optimizer="adam", loss="mse")

# Train model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Output validation results
output = model.evaluate(X_test, y_test)

# # Call method for validation
# method()

# # Method definition
# def method():
#     print("Validation Loss:", output)