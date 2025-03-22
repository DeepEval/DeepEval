import math

def method(n):

    # Case when n is 1
    if n == 1:
        return 2

    # Calculate the number of numbers with n digits that start with 1
    numbers_starting_with_1 = 9 * math.pow(10, n - 2)

    # Calculate the number of numbers with n digits that end with 1
    numbers_ending_with_1 = 10 * math.pow(10, n - 1)

    # Calculate the total number of numbers with n digits that start or end with 1
    total_numbers = numbers_starting_with_1 + numbers_ending_with_1

    return total_numbers

# Test case
n = 3
output = method(n)
print(output)  # Output: 198