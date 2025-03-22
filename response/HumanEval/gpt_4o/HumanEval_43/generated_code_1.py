def method(lst):
    seen = set()
    for num in lst:
        if -num in seen:
            return True
        seen.add(num)
    return False

# Test case for validation
test_list = [3, 1, -1, 4, 2]
output = method(test_list)
print(output)  # This should print True because 1 and -1 sum to zero.