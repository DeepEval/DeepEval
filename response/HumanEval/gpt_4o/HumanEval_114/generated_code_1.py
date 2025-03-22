def method(nums):
    if not nums:
        return None  # Handle edge case if nums is empty
    
    # Initialize the minimum sum with a large positive value
    min_sum = float('inf')
    current_sum = 0
    
    for num in nums:
        # Update the current sum to be the smallest sum possible ending at this position
        current_sum = min(current_sum + num, num)
        
        # Update the minimum sum found so far
        min_sum = min(min_sum, current_sum)
        
    return min_sum

# Test case for validation
nums = [3, -4, 2, -3, -1, 7, -5]
print(method(nums))  # Expected output: -6, which is the sum of the sub-array [-4, 2, -3, -1]