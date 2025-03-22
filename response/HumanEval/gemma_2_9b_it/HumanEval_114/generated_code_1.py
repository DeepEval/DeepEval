def method():
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    min_sum = nums[0]
    current_sum = 0
    for num in nums:
        current_sum += num
        min_sum = min(min_sum, current_sum)
        if current_sum < 0:
            current_sum = 0
    return min_sum

output = method()
print(output)