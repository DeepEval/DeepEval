import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
# data = pd.read_csv("your_data.csv")

# 修改为本地数据文件
data = pd.read_csv("evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_533.csv")

# Identify outliers and high leverage points
z_score = StandardScaler().fit_transform(data)
outliers = z_score[abs(z_score) > 2]
high_leverage_points = data[abs(z_score) > 1.5]

# Remove outliers and high leverage points
data = data.drop(outliers.index)
data = data.drop(high_leverage_points.index)

# Re-run the regression
X = data.drop("target", axis=1)
y = data["target"]
X_scaled = StandardScaler().fit_transform(X)
model = LinearRegression().fit(X_scaled, y)

# Print the results
print("R-squared value:", model.score(X_scaled, y))
print("Coefficients:", model.coef_)