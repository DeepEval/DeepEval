import numpy as np
from sklearn.preprocessing import StandardScaler

def method():
    # Sample data: features
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the features
    scaler.fit(X)

    # Transform the features
    scaled_X = scaler.transform(X)

    # Return the scaled output
    return scaled_X

# Call the method for validation
output = method()
print(output)