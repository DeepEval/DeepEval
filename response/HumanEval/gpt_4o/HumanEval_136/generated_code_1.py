def method(lst):
    # Initialize variables to hold the largest negative and smallest positive integers
    largest_negative = None
    smallest_positive = None

    # Iterate over each number in the list
    for num in lst:
        # Check for negative numbers
        if num < 0:
            if largest_negative is None or num > largest_negative:
                largest_negative = num
        # Check for positive numbers
        elif num > 0:
            if smallest_positive is None or num < smallest_positive:
                smallest_positive = num

    # Create the output tuple
    output = (largest_negative, smallest_positive)
    return output

# Test case for validation
test_list = [-10, -20, 5, 3, 0, 15, -1, 2]
result = method(test_list)
print(result)  # Output should be (-1, 2)