import numpy as np

def method(data):
    # Calculate the mean of the outcome variable
    output = np.mean(data)
    return output

# Example usage:
if __name__ == "__main__":
    # Example data
    outcome_variable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Call the method function
    predicted_mean = method(outcome_variable)
    # Print the predicted mean
    print(f"The predicted mean of the outcome variable is: {predicted_mean}")