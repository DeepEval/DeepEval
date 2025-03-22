import numpy as np

def method():
    # Example dataset (replace this with your actual dataset)
    X = np.array([1, 2, 3, 4, 5])  # Features
    y = np.array([2, 3, 4, 5, 6])  # Target variable

    # Calculate the mean of the target variable
    mean_y = np.mean(y)

    # Return the mean as the output
    output = mean_y

    return output

# Call the method for validation
result = method()
print(result)