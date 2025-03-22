import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def method():
    # Sample data creation
    data = {
        'age': np.random.randint(18, 80, size=100),  # Random ages between 18 and 80
        'income': np.random.randint(20000, 100000, size=100),  # Random income values
        'education_level': np.random.randint(1, 5, size=100),  # Random education level (1-4)
        'experience': np.random.randint(0, 40, size=100),  # Years of experience
        'target': np.random.randint(0, 100, size=100)  # Target variable to predict
    }
    
    df = pd.DataFrame(data)
    
    # Define features and target
    X = df[['age', 'income', 'education_level', 'experience']]
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    
    # Output
    output = {
        'predictions': predictions,
        'mean_squared_error': mse
    }
    
    return output

# Call the method to validate
result = method()
print("Predictions:", result['predictions'])
print("Mean Squared Error:", result['mean_squared_error'])