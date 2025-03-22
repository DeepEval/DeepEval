import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def method():
    # Load training data
    X_train = # load feature data
    y_train = # load target data

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    score = model.score(X_val, y_val)

    # Return the final output
    return score

# Call the method for validation
score = method()
print(score)