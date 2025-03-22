import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def method():
    # Sample data for demonstration
    data = {
        'total_request_rate': [100, 200, 300, 400, 500],
        'cpu': [10, 20, 30, 40, 50]
    }
    
    # Creating a DataFrame
    df = pd.DataFrame(data)
    
    # Defining feature and target variable
    X = df[['total_request_rate']]
    y = df['cpu']
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model.predict(X_test)
    
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Returning the coefficients and evaluation metrics
    output = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'mean_squared_error': mse,
        'r2_score': r2
    }
    
    return output

# Call the method for validation
output = method()
print(output)