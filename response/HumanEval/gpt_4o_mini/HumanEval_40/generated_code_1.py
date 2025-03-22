def method():
    def triples_sum_to_zero(nums):
        nums.sort()  # Sort the list first
        n = len(nums)
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:  # Skip duplicates
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

    # Example test case
    test_case = [-1, 0, 1, 2, -1, -4]
    output = triples_sum_to_zero(test_case)
    return output

# Execute the method and print the output
output = method()
print(output)  # Should print True as there are triples (-1, 0, 1) that sum to zero.