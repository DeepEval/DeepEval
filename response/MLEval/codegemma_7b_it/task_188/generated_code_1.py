from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def method():
    # Load data
    X = ...  # Input data
    Y = ...  # Target data

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create and fit the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Evaluate the model on the test set
    output = model.score(X_test, Y_test)

    return output

# Call the method
output = method()

# Print the output
print(output)