import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
# df = pd.read_csv('data.csv')

# 修改为本地数据文件
df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codellama_7b_instruct_hf/testcases/task_102.csv')

# Split the data into features and target variable
X = df[['X1', 'X2', 'X3']]
y = df['y']

# Create a LinearRegression model and fit the data
model = LinearRegression()
model.fit(X, y)

# Get the coefficients of the regression line
coefs = model.coef_

# Print the coefficients
print(coefs)