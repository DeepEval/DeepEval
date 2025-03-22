from sys import maxsize

def method(nums):
    min_sum = maxsize
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            min_sum = min(min_sum, current_sum)
    return min_sum

nums = [3, 4, -1, -2, 5, 2]
output = method(nums)
print(output)  # Output: -2