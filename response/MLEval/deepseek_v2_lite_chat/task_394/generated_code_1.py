from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def method():
    # Load your training data (replace 'data.csv' with your actual file)
    data = pd.read_csv('data.csv')

    # Assuming the target column is 'target' and the features are in remaining columns
    X = data.drop('target', axis=1)
    y = data['target']

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Output the trained model if needed
    return model

# Call the method for validation
output = method()