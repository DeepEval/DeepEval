import pandas as pd
from sklearn.linear_model import LinearRegression

def method():
    # Load the dataset
    # data = pd.read_csv('cpu_data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_308.csv')

    # Define the features and target variable
    X = data[['Total Request Rate']]
    y = data['CPU']

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict CPU usage for a given total request rate
    request_rate = 1000
    predicted_cpu = model.predict([[request_rate]])

    # Return the predicted CPU usage
    return predicted_cpu

# Call the method for validation
predicted_cpu = method()
print("Predicted CPU usage:", predicted_cpu)