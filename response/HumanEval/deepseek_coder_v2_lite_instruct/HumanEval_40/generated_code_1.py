def method():
    def triples_sum_to_zero(nums):
        nums.sort()
        n = len(nums)
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
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
    test_case = [1, -1, 0, 2, -2, -1]
    output = triples_sum_to_zero(test_case)
    return output

# Call the method and print the output
print(method())