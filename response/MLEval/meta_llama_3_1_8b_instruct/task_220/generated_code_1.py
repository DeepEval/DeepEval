# Import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some sample data
np.random.seed(0)
data = {
    'age': np.random.randint(18, 65, 100),
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
}
df = pd.DataFrame(data)

# Define the method
def method():
    # Split the data into training and testing sets
    X = df[['feature1', 'feature2']]
    y = df['age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Return the predicted values
    return predictions

# Call the method and store the output
output = method()

# Print the output
print(output)

# To return the final 'output' if needed
# return output