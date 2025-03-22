import math

def double_the_difference(nums):
    sum_of_squares = 0

    for num in nums:
        if isinstance(num, int) and num > 0 and num % 2 != 0:
            sum_of_squares += math.pow(num, 2)

    return sum_of_squares

# Test case
nums = [1, 3, 2, 0]
print(double_the_difference(nums))  # Output: 10

nums = [-1, -2, 0]
print(double_the_difference(nums))  # Output: 0

nums = [9, -2]
print(double_the_difference(nums))  # Output: 81

nums = [0]
print(double_the_difference(nums))  # Output: 0