import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def method():
    # Load your dataset
    # data = pd.read_csv('your_dataset.csv')

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_533.csv')
    
    # Assuming the target column is the last column in the dataset
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolate outliers using isolation forest
    clf = IsolationForest(contamination=0.01)  # 1% contamination threshold
    outliers = clf.fit_predict(X_scaled)
    
    # Identify rows with outlier predictions
    outlier_indices = np.where(outliers == -1)[0]
    
    # Remove outliers and high leverage points
    robust_X = X.iloc[~np.isin(range(len(X)), outlier_indices), :]
    robust_y = y[~np.isin(range(len(y)), outlier_indices)]
    
    # Retrain the model
    model = LinearRegression()
    model.fit(robust_X, robust_y)
    
    # Predict using the model
    predictions = model.predict(robust_X)
    
    # Calculate the Mean Squared Error (MSE) of the predictions
    mse = mean_squared_error(robust_y, predictions)
    
    # Output the results
    output = {
        'Model MSE': mse,
        'Number of Outliers Removed': len(outlier_indices)
    }
    
    return output

# Call the method for validation
output = method()
print(output)