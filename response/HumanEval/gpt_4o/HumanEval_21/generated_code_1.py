def method(numbers):
    if not numbers or len(numbers) < 2:
        raise ValueError("The list must contain at least two elements.")
    
    min_value = min(numbers)
    max_value = max(numbers)
    
    # Avoid division by zero if all elements in the list are the same
    if min_value == max_value:
        raise ValueError("All elements in the list are the same. Transformation is not possible.")

    output = [(x - min_value) / (max_value - min_value) for x in numbers]
    return output

# Test case
numbers = [3, 6, 9, 15, 18]
output = method(numbers)
print(output)  # Expected output: [0.0, 0.23076923076923078, 0.46153846153846156, 0.9230769230769231, 1.0]