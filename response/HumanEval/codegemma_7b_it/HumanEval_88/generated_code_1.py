from copy import deepcopy

def method(arr):
    output = deepcopy(arr)
    sum_first_last = sum(arr[0], arr[-1])

    if sum_first_last % 2 != 0:
        output.sort()
    else:
        output.sort(reverse=True)

    return output

# Test Case
arr = [1, 5, 8, 3, 6]
expected_output = [1, 3, 5, 6, 8]

output = method(arr)

print(output)