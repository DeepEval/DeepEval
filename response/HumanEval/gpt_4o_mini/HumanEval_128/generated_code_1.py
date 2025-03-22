def method(arr):
    # Check if the array is empty and return None if it is
    if not arr:
        return None

    # Initialize the product of signs
    sign_product = 1
    sum_magnitudes = 0
    
    for num in arr:
        # Determine the sign of the number
        if num > 0:
            sign_product *= 1  # positive
            sum_magnitudes += num  # add magnitude
        elif num < 0:
            sign_product *= -1  # negative
            sum_magnitudes += -num  # add magnitude (positive form)
        else:
            # If the number is zero, the product of signs becomes zero
            sign_product = 0
            break  # No need to check further as product will remain zero

    # Return the final output based on the product of signs
    return sum_magnitudes * sign_product

# Test case for validation
test_case = [-3, 2, -5, 0]
output = method(test_case)
print(output)  # Expected output is 0 because of the presence of 0 in the array

# Another test case
test_case_2 = [3, -2, 5]
output_2 = method(test_case_2)
print(output_2)  # Expected output is (3 + 2 + 5) * (-1) = -10