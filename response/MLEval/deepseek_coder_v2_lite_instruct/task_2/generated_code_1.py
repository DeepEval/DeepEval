import numpy as np
from sklearn.preprocessing import StandardScaler

def method():
    # Example dataset: a 2D array where each row is a feature vector
    X = np.array([[0, 15], [1, -10], [2, 30], [5, -20]])
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler to the features
    scaler.fit(X)
    
    # Transform the features
    X_scaled = scaler.transform(X)
    
    # For demonstration, let's return the scaled features
    output = X_scaled
    
    return output

# Call the method for validation
output = method()
print(output)