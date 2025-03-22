from sklearn.preprocessing import StandardScaler

def method():
    # Load the data
    X = # your data

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the data
    scaler.fit(X)

    # Return the scaled data
    return scaler.transform(X)