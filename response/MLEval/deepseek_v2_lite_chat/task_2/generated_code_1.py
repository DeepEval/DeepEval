import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def method():
    # Generate some dummy data for demonstration
    data = np.array([[2, -1], [-1, 2], [0, 0], [1, 1], [-1, -1]])
    
    # Create a StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the data with the scaler
    scaled_data = scaler.fit_transform(data)
    
    # Convert the transformed data back to a numpy array and return
    return np.array(scaled_data)

# Call the method for validation
output = method()
print("Scaled Data:", output)