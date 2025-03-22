import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to fit the model
def method():
    # Load your dataset
    # Replace 'data.csv' with your actual dataset file path
    # df = pd.read_csv('data.csv')

    # 修改为本地数据文件
    df = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_308.csv')

    # Assuming 'request_rate' and 'cpu_usage' are the columns for total request rate and CPU usage
    X = df[['request_rate', 'cpu']]  # Features
    y = df['target']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Visualize the predictions
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.show()

    # Return the model or the evaluation results
    return model, mse

# Call the method for validation
model, mse = method()