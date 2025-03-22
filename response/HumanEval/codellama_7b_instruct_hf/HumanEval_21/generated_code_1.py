from sklearn.preprocessing import MinMaxScaler

def method():
    # Given list of numbers
    nums = [10, 5, 8, 0, 2, 15]

    # Create a MinMaxScaler object
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler to the input list
    scaler.fit(nums.reshape(-1, 1))

    # Transform the input list using the scaler
    transformed_nums = scaler.transform(nums.reshape(-1, 1))

    # Reshape the transformed list to a flat array
    output = transformed_nums.flatten()

    return output

# Test case
nums = [10, 5, 8, 0, 2, 15]
output = method()
print(output)