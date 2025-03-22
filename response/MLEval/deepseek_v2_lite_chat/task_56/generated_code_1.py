import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Load your data here. Replace 'your_data.csv' with the path to your dataset.
    # data = pd.read_csv('your_data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_56.csv')

    # Define the features (X) and the target (y)
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the name of your target variable
    y = data['target_column']

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train your model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)

    # The output is the mean squared error of the predictions
    output = mse

    return output

# Call the method for validation
result = method()
print("Mean Squared Error:", result)