# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Create a sample dataset
np.random.seed(0)
X1 = np.random.rand(100, 1)
X2 = np.random.rand(100, 1)
y = 2 + 3 * X1 + 4 * X2 + np.random.randn(100, 1)

# Create a DataFrame
df = pd.DataFrame(np.hstack((X1, X2, y)), columns=['X1', 'X2', 'y'])

# Define the method function
def method():
    # Add a constant to the independent value
    X = sm.add_constant(df[['X1', 'X2']])
    
    # Fit the model
    model = sm.OLS(df['y'], X).fit()
    
    # Print the summary of the model
    print(model.summary())
    
    # Extract coefficients
    coeff = model.params
    
    # Print the coefficients
    print("Coefficients:")
    print(f"MOMMY (X1): {coeff['X1']}")
    print(f"DADMY (X2): {coeff['X2']}")
    
    # Return the coefficients
    return {"MOMMY": coeff['X1'], "DADMY": coeff['X2']}

# Call the method function
output = method()
print("Final Output:")
print(output)