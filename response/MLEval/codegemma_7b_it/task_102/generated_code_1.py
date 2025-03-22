import statsmodels.api as sm

def method():

    # Load the data (assuming it's available in a variable called 'data')
    X = data['X1']
    y = data['y']

    # Add a constant to X for the intercept
    X = sm.add_constant(X)

    # Create a regression model
    model = sm.OLS(y, X).fit()

    # Print the regression results
    print(model.summary())

    # Return the output if needed
    output = model.summary()
    return output

# Call the method to generate the regression results
method()