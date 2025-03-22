def method(nums):
    # Sum of odd elements at even indices
    total = sum(num for i, num in enumerate(nums) if i % 2 == 0 and num % 2 != 0)
    return total

# Test case
test_nums = [10, 3, 5, 7, 9, 11, 2, 14, 6]
output = method(test_nums)
print(output)  # Expected output: 14 (5 from index 2 and 9 from index 4)