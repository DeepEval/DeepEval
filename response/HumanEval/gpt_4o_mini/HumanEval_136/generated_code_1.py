def method(input_list):
    # Initialize variables to hold the largest negative and smallest positive integers
    largest_negative = None
    smallest_positive = None
    
    # Iterate through the input list to find the required values
    for num in input_list:
        if num < 0:  # Check for negative integers
            if largest_negative is None or num > largest_negative:
                largest_negative = num
        elif num > 0:  # Check for positive integers
            if smallest_positive is None or num < smallest_positive:
                smallest_positive = num
    
    # Return a tuple with the results
    return (largest_negative, smallest_positive)

# Test case for validation
input_list = [-5, -1, 3, 4, -3, 2, -4]
output = method(input_list)
print(output)  # Expected output: (-1, 2)