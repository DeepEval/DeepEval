def method(nums):
    output = []
    while nums:
        min_val = min(nums)
        output.append(min_val)
        nums.remove(min_val)
        if nums:
            max_val = max(nums)
            output.append(max_val)
            nums.remove(max_val)
    return output

# Test Case
nums = [3, 1, 2, 4, 5]
sorted_nums = method(nums)
print(f"Original list: {nums}")
print(f"Strange sorted list: {sorted_nums}")