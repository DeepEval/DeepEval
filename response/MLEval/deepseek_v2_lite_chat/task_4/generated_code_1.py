from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Scale the features for better performance (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Ridge regression model
ridge = Ridge(alpha=0.5)

# Train the model using the training set
ridge.fit(X_train, y_train)

# Make predictions using the trained model and the test set
y_pred = ridge.predict(X_test)

# Evaluate the performance of the model (using mean squared error as an example)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Return the final output (you can add any additional processing or calculations here)
output = y_pred

# Call the method for validation
print("Output:", method())