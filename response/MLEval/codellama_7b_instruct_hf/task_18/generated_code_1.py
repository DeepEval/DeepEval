import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
# df = pd.read_csv("data.csv")

# 修改为本地数据文件
df = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_18.csv")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42)

# Create a LinearRegression model instance
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)

# Evaluate the model using the test data
y_pred = model.predict(X_test)

# Print the mean squared error (MSE)
print("MSE:", mean_squared_error(y_test, y_pred))

# Return the final output
output = y_pred