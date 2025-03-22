import sklearn.preprocessing

def method():

    # Load the dataset (assuming it's available)
    X = ...  # Features

    # Fit a scaler to the features
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X)

    # Return the scaler object (optional)
    return scaler

# Call the method for validation
scaler = method()

# Print the scaler object (optional)
print(scaler)