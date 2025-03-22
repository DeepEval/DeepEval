def method(numbers):
    # Check if the input list has at least two elements
    if len(numbers) < 2:
        raise ValueError("The list must contain at least two elements.")
    
    # Find the minimum and maximum values in the list
    min_value = min(numbers)
    max_value = max(numbers)
    
    # Apply the linear transformation
    transformed_values = [(x - min_value) / (max_value - min_value) for x in numbers]
    
    return transformed_values

# Test case for validation
if __name__ == "__main__":
    test_numbers = [10, 20, 30, 40, 50]
    output = method(test_numbers)
    print("Transformed values:", output)  # Expected output: [0.0, 0.25, 0.5, 0.75, 1.0]