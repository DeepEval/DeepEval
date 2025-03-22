import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def method():
    # Load your training data (replace 'data.csv' with your actual file)
    # data = pd.read_csv('data.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_18.csv')
    
    # Assume 'Size' and 'Location' are columns in the dataset that you want to predict
    X = data[['Size', 'Location']]
    y = data['Price']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features to speed up convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the test set results
    y_pred = model.predict(X_test)
    
    # Evaluate the model (optional)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Return the predictions or the model itself, depending on the task
    return y_pred

# Call the method for validation
output = method()