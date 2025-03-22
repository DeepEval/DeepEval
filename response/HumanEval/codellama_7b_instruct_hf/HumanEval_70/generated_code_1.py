# Import necessary packages
import heapq

def method(nums):
    """
    Given list of integers, return list in strange order. Strange sorting, is when you start with the minimum value, then maximum of the remaining integers, then minimum and so on.

    Args:
        nums: List of integers.

    Returns:
        List in strange order.
    """

    # Sort the list in ascending order
    nums.sort()

    # Create an empty output list
    output = []

    # Iterate over the sorted list
    while nums:
        # Get the minimum value and add it to the output list
        output.append(nums[0])
        nums.pop(0)

        # Get the maximum value and add it to the output list
        output.append(nums[-1])
        nums.pop(-1)

    # Return the output list
    return output


# Test case
nums = [1, 2, 3, 4, 5]
output = method(nums)
print(output)  # Output: [1, 5, 2, 4, 3]