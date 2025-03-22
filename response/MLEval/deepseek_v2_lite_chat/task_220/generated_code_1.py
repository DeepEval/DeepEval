import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def method():
    # Load a sample dataset (replace this with your actual dataset)
    # data = pd.read_csv('sample_data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_220.csv')

    # Assume the dataset has columns: 'Age', 'Price' and other relevant features
    # For simplicity, let's assume 'Price' is the target variable we want to predict
    X = data[['Age']]  # Features are only age
    y = data['Price']  # Target variable is the price

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Plot the actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()

    # Return the final output (predicted prices)
    return y_pred

# Call the method for validation
output = method()