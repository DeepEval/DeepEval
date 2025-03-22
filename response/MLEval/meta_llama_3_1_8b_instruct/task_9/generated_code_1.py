# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Generate some sample data
np.random.seed(0)
data = {
    'Days': np.random.randint(1, 100, 100),
    'Morale': np.random.randint(1, 100, 100)
}
df = pd.DataFrame(data)

# Define the method function
def method():
    # Split data into features (X) and target (y)
    X = df[['Days']]
    y = df['Morale']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, predictions)

    # Print the mean squared error
    print(f'Mean Squared Error: {mse}')

    # Return the output
    return mse

# Call the method function for validation
output = method()
print('Final Output:', output)