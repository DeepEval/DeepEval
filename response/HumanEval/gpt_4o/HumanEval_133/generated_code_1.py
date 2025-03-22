import math

def method(numbers):
    # Round each number to the nearest upper integer
    rounded_numbers = [math.ceil(num) for num in numbers]
    # Calculate the sum of squares of the rounded numbers
    output = sum(num ** 2 for num in rounded_numbers)
    return output

# Test case for validation
test_numbers = [1.2, 2.5, 3.7, 4.1]
expected_output = 4**2 + 3**2 + 4**2 + 5**2  # 16 + 9 + 16 + 25 = 66
# assert method(test_numbers) == expected_output, "Test case failed!"
method(test_numbers)

# print("Test case passed!")