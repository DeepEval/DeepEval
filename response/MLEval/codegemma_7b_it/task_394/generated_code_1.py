import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the training data
# train_data = pd.read_csv("train_data.csv")

# 修改为本地数据文件
train_data = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_394.csv")

# Extract the features and labels from the training data
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Return the fitted model
output = model
