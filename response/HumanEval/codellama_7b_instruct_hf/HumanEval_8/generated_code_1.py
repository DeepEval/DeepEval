from functools import reduce

def method(nums):
    """
    Returns a tuple consisting of a sum and a product of all the integers in a list.

    Empty sum should be equal to 0 and empty product should be equal to 1.
    """

    # Handle empty list case
    if not nums:
        return 0, 1

    # Calculate sum using reduce function
    sum_nums = reduce(lambda x, y: x + y, nums)

    # Calculate product using reduce function
    product_nums = reduce(lambda x, y: x * y, nums)

    return sum_nums, product_nums

# Test case
nums = [1, 2, 3, 4, 5]
print(method(nums))  # Output: (15, 120)

nums = []
print(method(nums))  # Output: (0, 1)