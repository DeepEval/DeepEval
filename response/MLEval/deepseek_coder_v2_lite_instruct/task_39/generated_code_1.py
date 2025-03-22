def method():
    # Example data for y_true and y_pred
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    
    # Calculate the mean of y_true
    y_mean_true = sum(y_true) / len(y_true)
    
    # Calculate the numerator and denominator of the R-squared formula
    numerator = sum((y - y_pred[i]) ** 2 for i, y in enumerate(y_true))
    denominator = sum((y - y_mean_true) ** 2 for y in y_true)
    
    # Calculate the R-squared value
    r_squared = 1 - (numerator / denominator)
    
    # Return the R-squared value
    return r_squared

# Call the method and print the result
output = method()
print("R-squared:", output)