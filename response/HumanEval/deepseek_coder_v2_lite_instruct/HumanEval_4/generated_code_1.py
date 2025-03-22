def method(numbers):
    if not numbers:
        return 0  # If the list is empty, return 0 as MAD is not defined.

    mean = sum(numbers) / len(numbers)
    absolute_differences = [abs(x - mean) for x in numbers]
    mad = sum(absolute_differences) / len(numbers)
    
    return mad

# Test case
input_numbers = [1, 2, 3, 4, 5]
output = method(input_numbers)
print(f"Mean Absolute Deviation: {output}")  # Expected output: Mean Absolute Deviation: 1.0