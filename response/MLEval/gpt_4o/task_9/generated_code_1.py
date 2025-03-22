import numpy as np
from sklearn.linear_model import LinearRegression

def method():
    # Example data: Days and corresponding morale scores
    days = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    morale = np.array([10, 12, 15, 18, 20, 21, 22, 23, 25, 27])

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(days, morale)

    # Predict morale for a given number of days (e.g., 11 days)
    days_to_predict = np.array([[11]])
    predicted_morale = model.predict(days_to_predict)

    # Prepare the output
    output = predicted_morale[0]
    return output

# Call the method for validation
predicted_morale = method()
print(f"Predicted morale for 11 days: {predicted_morale}")