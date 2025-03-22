def rolling_maximum(nums):
    """
    Generate a list of rolling maximum element found until the given moment in the sequence.

    Args:
        nums (list): A list of integers.

    Returns:
        list: A list of rolling maximum elements.
    """
    if not nums:
        return []

    output = [nums[0]]
    max_so_far = nums[0]

    for num in nums[1:]:
        if num >= max_so_far:
            max_so_far = num
            output.append(num)
        else:
            output.append(max_so_far)

    return output


def method():
    nums = [3, 2, -1, -3, 6, 4, -1, 3, 2, 0]
    return rolling_maximum(nums)


output = method()
print(output)