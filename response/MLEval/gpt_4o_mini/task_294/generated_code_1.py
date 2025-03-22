import pandas as pd
import numpy as np
import statsmodels.api as sm

def method():
    # Sample DataFrame creation for demonstration purposes
    # In practice, you would load your dataset here
    data = {
        'age': np.random.randint(30, 80, 100),
        'positive_nodes': np.random.randint(0, 10, 100),
        'operation_year': np.random.randint(2000, 2021, 100),
        'survival': np.random.randint(0, 2, 100)  # Binary survival (1: survived, 0: not survived)
    }
    
    df = pd.DataFrame(data)
    
    # Define the independent variables (features)
    X = df[['age', 'positive_nodes', 'operation_year']]
    
    # Add a constant to the model (intercept)
    X = sm.add_constant(X)
    
    # Define the dependent variable (target)
    y = df['survival']
    
    # Fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()
    
    # Get the summary of the model
    output = result.summary()
    
    return output

# Call the method to validate
output = method()
print(output)