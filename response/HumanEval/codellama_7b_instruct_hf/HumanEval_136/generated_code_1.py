import math

def method():
    nums = [3, 5, 7, -1, 2, -3, -7, 9, 1]

    if not nums:
        return None, None

    max_neg = float('-inf')
    min_pos = float('inf')

    for num in nums:
        if num < 0 and num > max_neg:
            max_neg = num
        elif num > 0 and num < min_pos:
            min_pos = num

    if max_neg == float('-inf') or min_pos == float('inf'):
        return None, None

    return max_neg, min_pos


# Test case
output = method()
print(output)  # Output: (-1, 1)