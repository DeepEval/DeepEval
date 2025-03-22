import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def method():
    # Create a synthetic dataset
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    Y = np.sin(X).ravel() + np.random.randn(100) * 3
    
    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale the features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define and train the KNN regressor
    n_neighbors = 30
    knn = KNeighborsRegressor(n_neighbors, weights='distance')
    knn.fit(X_train_scaled, Y_train)
    
    # Predict and evaluate
    Y_pred = knn.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_pred)
    
    print(f"KNN Regressor MSE: {mse}")
    
    # Plotting the results
    plt.scatter(X_test, Y_test, color='black')
    plt.plot(X_test, Y_pred, color='blue', linewidth=3)
    plt.show()
    
    # Return the mean squared error as output
    return mse

# Call the method to validate the results
output = method()