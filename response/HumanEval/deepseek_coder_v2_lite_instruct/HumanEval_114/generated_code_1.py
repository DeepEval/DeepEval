def method():
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    min_sum = float('inf')
    current_min = 0

    for num in nums:
        current_min = min(current_min + num, num)
        min_sum = min(min_sum, current_min)

    return min_sum

# Test case
print(method())  # Expected output: -5