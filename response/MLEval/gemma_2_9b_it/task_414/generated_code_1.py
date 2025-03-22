from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def method():
    # Load your data (replace with your actual data loading)
    X = np.random.rand(100, 5)  # Example: 100 samples, 5 features
    y = np.random.rand(100)      # Example: 100 target values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Lasso model with k = 30
    lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42, solver='saga') 
    lasso.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lasso.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (k=30): {mse}")

    # Visualize the model performance (optional)
    # ... Your code to visualize the model's performance ...

    return mse 

# Call the method and print the output
output = method()
print("Final Output:", output)