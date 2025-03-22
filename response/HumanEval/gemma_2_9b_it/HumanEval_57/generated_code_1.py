def method(nums):
    """
    Return True if list elements are monotonically increasing or decreasing.
    """

    increasing = decreasing = True
    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1]:
            increasing = False
        if nums[i] > nums[i - 1]:
            decreasing = False
    return increasing or decreasing

# Test Case
nums1 = [1, 2, 3, 4, 5]
nums2 = [5, 4, 3, 2, 1]
nums3 = [1, 3, 2, 4]

print(f"Is {nums1} monotonically increasing or decreasing? {method(nums1)}")  # Output: True
print(f"Is {nums2} monotonically increasing or decreasing? {method(nums2)}")  # Output: True
print(f"Is {nums3} monotonically increasing or decreasing? {method(nums3)}")  # Output: False