import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def method():
    # Load your dataset (replace 'your_dataset.csv' with your actual file)
    # data = pd.read_csv('your_dataset.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_294.csv')

    # Assuming your columns are named 'age', 'positive_nodes', 'operation_year', and 'survival'
    X = data[['age', 'positive_nodes', 'operation_year']] 
    y = data['survival']

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Get the coefficients (slopes) for each feature
    coefficients = model.coef_

    # Interpret the coefficients
    output =  "The model shows the following relationships:\n"
    output += f"- Age:  Coefficient = {coefficients[0]:.2f} (Negative correlation with survival)\n"
    output += f"- Positive Nodes: Coefficient = {coefficients[1]:.2f} (Negative correlation with survival)\n"
    output += f"- Operation Year: Coefficient = {coefficients[2]:.2f} (Positive correlation with survival)\n"

    return output

# Call the method and print the output
print(method())