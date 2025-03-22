def method():
    # Example list of numbers
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    
    # Check if the list has at least two elements
    if len(numbers) < 2:
        raise ValueError("The list must contain at least two elements.")
    
    # Find the minimum and maximum values in the list
    min_value = min(numbers)
    max_value = max(numbers)
    
    # Apply the linear transformation
    transformed_numbers = [(num - min_value) / (max_value - min_value) for num in numbers]
    
    # Return the transformed list
    return transformed_numbers

# Test case
output = method()
print("Transformed list:", output)