def method(nums):
    nums.sort()  # Sort the list to make it easier to avoid duplicates and use two pointers
    n = len(nums)
    
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # Skip duplicate elements to avoid duplicate triplets
        
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                return True
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return False

# Test case
print(method([-1, 0, 1, 2, -1, -4]))  # This should return True because (-1, 0, 1) is a valid triplet.