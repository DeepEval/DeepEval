import math

def count_nums(nums):

    count = 0
    for num in nums:
        sum_of_digits = 0
        abs_num = abs(num)

        while abs_num > 0:
            digit = abs_num % 10
            sum_of_digits += digit
            abs_num //= 10

        if sum_of_digits > 0:
            count += 1

    return count


# Test Case
nums = [123, -456, 789, -1, -12, -100, -10]
result = count_nums(nums)
print(result)  # Output: 4