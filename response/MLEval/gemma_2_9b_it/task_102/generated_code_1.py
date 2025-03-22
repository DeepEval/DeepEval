import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Load your data (replace 'your_data.csv' with your actual file)
    # df = pd.read_csv('your_data.csv') 

    # 修改为本地数据文件
    df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_102.csv')

    # Assuming 'X1' is your feature and 'y' is your target variable
    X = df[['X1']]  
    y = df['y']

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients
    coefficient_mom = model.coef_[0]  
    coefficient_dad = model.intercept_  

    # Interpretation
    output = f"The regression line equation is: y = {coefficient_mom:.2f} * X1 + {coefficient_dad:.2f}\n"
    output += f"Interpretation:\n"
    output += f"- Coefficient of X1 ({coefficient_mom:.2f}): For every one-unit increase in X1, the target variable (y) is predicted to increase by {coefficient_mom:.2f} units.\n"
    output += f"- Intercept ({coefficient_dad:.2f}): This is the predicted value of y when X1 is zero."

    return output

# Call the method and print the output
print(method())