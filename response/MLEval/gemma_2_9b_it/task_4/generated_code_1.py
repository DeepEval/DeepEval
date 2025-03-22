from sklearn.linear_model import Ridge

def method():
    # Initialize a Ridge regressor with alpha = 1.0
    ridge_reg = Ridge(alpha=1.0) 

    return ridge_reg

# Call the method and store the result
output = method()

# You can now use the 'output' object (ridge_reg) to fit your data and make predictions.
# For example:
# X_train, X_test, y_train, y_test = ...  # Load your training and testing data
# output.fit(X_train, y_train)  
# predictions = output.predict(X_test)