# Import necessary packages
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Generate some sample data
np.random.seed(0)
data = {
    'Feature1': np.random.rand(10),
    'Feature2': np.random.rand(10),
    'Feature3': np.random.rand(10)
}
df = pd.DataFrame(data)

def method():
    """
    Fit scaler to the features.

    Returns:
        scaler (StandardScaler): The fitted StandardScaler instance.
    """
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the features
    scaler.fit(df.drop('Feature3', axis=1))  # Fit to all features except 'Feature3'
    
    return scaler

# Call the method for validation
scaler = method()

# To scale the data, you can use the transform method
scaled_data = scaler.transform(df.drop('Feature3', axis=1))
print(scaled_data)

# If you need to use the scaler later, you can save it
# scaler.save('scaler.joblib')

# To load the scaler, you can use the load method
# scaler = joblib.load('scaler.joblib')