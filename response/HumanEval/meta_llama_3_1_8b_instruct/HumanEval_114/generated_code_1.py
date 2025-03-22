def method():
    """
    Given an array of integers, find the minimum sum of any non-empty subarray.

    Args:
    None

    Returns:
    int: The minimum sum of any non-empty subarray.

    """
    def kadane_algorithm(nums):
        """
        Kadane's algorithm to find the maximum sum of a subarray.

        Args:
        nums (list): A list of integers.

        Returns:
        int: The maximum sum of a subarray.

        """
        max_sum = float('-inf')
        current_sum = 0

        for num in nums:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)

        return max_sum

    def min_sum_subarray(nums):
        """
        Find the minimum sum of any non-empty subarray.

        Args:
        nums (list): A list of integers.

        Returns:
        int: The minimum sum of any non-empty subarray.

        """
        n = len(nums)
        prefix_sum = [0] * (n + 1)

        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        min_sum = float('inf')
        for i in range(n):
            for j in range(i, n):
                subarray_sum = prefix_sum[j + 1] - prefix_sum[i]
                min_sum = min(min_sum, subarray_sum)

        return min_sum

    # Test case
    nums = [3, 2, -6, 4, 0]
    print("Minimum sum of any non-empty subarray:", min_sum_subarray(nums))

    return min_sum_subarray(nums)

output = method()
print("Final output:", output)