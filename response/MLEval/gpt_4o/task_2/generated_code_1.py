import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def method(features):
    # Create a StandardScaler instance
    scaler = StandardScaler()
    
    # Fit the scaler to the features
    scaler.fit(features)
    
    # Transform the features using the fitted scaler
    scaled_features = scaler.transform(features)
    
    return scaled_features

# Sample data for validation
def validate():
    # Creating a sample DataFrame with features
    data = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    features_df = pd.DataFrame(data)
    
    # Calling the method and printing the output
    scaled_features = method(features_df)
    print("Scaled features:\n", scaled_features)

validate()