# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Generate a sample dataset for demonstration purposes
np.random.seed(0)
data = {
    'Age': np.random.randint(18, 100, 100),
    'Positive Nodes': np.random.randint(0, 100, 100),
    'Operation Year': np.random.randint(1990, 2022, 100),
    'Survival': np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

# Define the method function
def method():
    # Split the data into features and target
    X = df[['Age', 'Positive Nodes', 'Operation Year']]
    y = df['Survival']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the R-squared score for each feature
    r2_age = r2_score(y_test, model.coef_[0]*X_test['Age'] + model.intercept_)
    r2_positive_nodes = r2_score(y_test, model.coef_[1]*X_test['Positive Nodes'] + model.intercept_)
    r2_operation_year = r2_score(y_test, model.coef_[2]*X_test['Operation Year'] + model.intercept_)

    # Print the R-squared scores
    print(f'R-squared score for Age: {r2_age}')
    print(f'R-squared score for Positive Nodes: {r2_positive_nodes}')
    print(f'R-squared score for Operation Year: {r2_operation_year}')

    # Plot the data
    plt.scatter(X_test['Age'], y_test)
    plt.title('Age vs Survival')
    plt.xlabel('Age')
    plt.ylabel('Survival')
    plt.show()

    plt.scatter(X_test['Positive Nodes'], y_test)
    plt.title('Positive Nodes vs Survival')
    plt.xlabel('Positive Nodes')
    plt.ylabel('Survival')
    plt.show()

    plt.scatter(X_test['Operation Year'], y_test)
    plt.title('Operation Year vs Survival')
    plt.xlabel('Operation Year')
    plt.ylabel('Survival')
    plt.show()

    # Return the R-squared scores
    return {
        'Age': r2_age,
        'Positive Nodes': r2_positive_nodes,
        'Operation Year': r2_operation_year
    }

# Call the method function
output = method()
print(output)