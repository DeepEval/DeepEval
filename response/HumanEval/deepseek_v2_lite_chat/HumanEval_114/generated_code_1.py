def method():
    def min_sum_subarray(nums):
        max_ending_here = max_so_far = float('-inf')
        for num in nums:
            max_ending_here = max(num, max_ending_here + num)
            max_so_far = max(max_so_far, max_ending_here)
        return max_so_far

    return min_sum_subarray(nums)

# Example usage:
# nums = [1, -1, 5, 7, -6, 8, -6, 1]
# print(method(nums))