def method(arr):
    if not arr:  # Check for empty array
        return True

    n = len(arr)
    breakpoints = 0

    for i in range(n):
        if arr[i] > arr[(i + 1) % n]:  # Compare with next element, wrapping around
            breakpoints += 1

    # If there's more than one breakpoint, we cannot sort the array by right shifts
    return breakpoints <= 1

# Test case
test_array = [3, 4, 5, 1, 2]  # This can be sorted by one right shift
output = method(test_array)
print(output)  # Should print True