from heapq import nsmallest

def next_smallest(nums):
    if not nums or len(nums) < 2:
        return None
    second_smallest = nsmallest(2, nums)[1]
    return second_smallest

# Test case
nums = [5, 1, 4, 3, 2]
output = next_smallest(nums)
print(output) # Output: 2