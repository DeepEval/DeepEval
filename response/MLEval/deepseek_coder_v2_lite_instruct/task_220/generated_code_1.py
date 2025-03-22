import numpy as np
from sklearn.linear_model import LinearRegression

def method():
    # Example data: age and corresponding outcome
    age = np.array([25, 35, 45, 55, 65]).reshape(-1, 1)
    outcome = np.array([50, 60, 70, 80, 90])

    # Create a linear regression model
    model = LinearRegression()
    model.fit(age, outcome)

    # Predict the outcome for a new age
    new_age = np.array([30, 40, 50]).reshape(-1, 1)
    predicted_outcome = model.predict(new_age)

    # Prepare the output
    output = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'predicted_outcome': predicted_outcome
    }

    return output

# Call the method for validation
output = method()
print(output)