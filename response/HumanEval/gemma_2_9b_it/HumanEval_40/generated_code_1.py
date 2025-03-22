def method():
    nums = [1, -1, -2, 2, 4, -3]  

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    return True
    return False

output = method()
print(output)  # Output: True