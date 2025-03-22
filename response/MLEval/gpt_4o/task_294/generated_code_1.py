import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def method():
    # Creating a synthetic dataset
    np.random.seed(0)
    n_samples = 100

    # Generate synthetic data
    age = np.random.randint(30, 80, n_samples)
    positive_nodes = np.random.randint(0, 20, n_samples)
    operation_year = np.random.randint(1950, 2000, n_samples)

    # Simulate survival using a linear combination (for demonstration purposes)
    survival = 100 - 0.5 * age - 1.5 * positive_nodes + 0.3 * operation_year + np.random.normal(0, 5, n_samples)

    # Create a DataFrame
    data = pd.DataFrame({
        'age': age,
        'positive_nodes': positive_nodes,
        'operation_year': operation_year,
        'survival': survival
    })

    # Define features and target
    X = data[['age', 'positive_nodes', 'operation_year']]
    y = data['survival']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    output = {
        'R2 Score': r2_score(y_test, y_pred),
        'Coefficients': model.coef_,
        'Intercept': model.intercept_
    }

    return output

# Call the method for validation
results = method()
print("R2 Score:", results['R2 Score'])
print("Coefficients:", results['Coefficients'])
print("Intercept:", results['Intercept'])