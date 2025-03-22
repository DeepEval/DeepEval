import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
# data = pd.read_csv("data.csv")

# 修改为本地数据文件
data = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_400.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42)

# Create a LinearRegression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
r2 = model.score(X_test, y_test)

print("R-squared value:", r2)