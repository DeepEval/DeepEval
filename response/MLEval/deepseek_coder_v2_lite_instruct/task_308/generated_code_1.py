import pandas as pd
from sklearn.linear_model import LinearRegression

def method():
    # Sample data
    data = {
        'request_rate': [100, 200, 300, 400, 500],
        'cpu_usage': [20, 30, 40, 50, 60]
    }
    df = pd.DataFrame(data)
    
    # Features and target
    X = df[['request_rate']]
    y = df['cpu_usage']
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the CPU usage for a new request rate
    new_request_rate = 250
    predicted_cpu = model.predict([[new_request_rate]])
    
    # Prepare the output
    output = {
        'predicted_cpu': predicted_cpu[0]
    }
    
    return output

# Call the method for validation
output = method()
print(output)