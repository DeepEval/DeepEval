from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load the Boston housing dataset
boston = load_boston()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3)

# Create a LinearRegression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance using mean squared error
mse = model.score(X_test, y_test)
print(f"Mean squared error: {mse}")