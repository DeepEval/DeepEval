def method(lst):
    if not lst:  # Check if the list is empty
        return True

    increasing = True
    decreasing = True

    for i in range(1, len(lst)):
        if lst[i] > lst[i - 1]:
            decreasing = False
        elif lst[i] < lst[i - 1]:
            increasing = False

    return increasing or decreasing

# Test case for validation
test_case = [1, 2, 2, 3, 4]  # Example of a monotonically increasing list
output = method(test_case)
print(f"Test case {test_case}: {output}")  # Expected: True

test_case2 = [5, 4, 4, 3, 1]  # Example of a monotonically decreasing list
output2 = method(test_case2)
print(f"Test case {test_case2}: {output2}")  # Expected: True

test_case3 = [1, 3, 2]  # Example of a non-monotonic list
output3 = method(test_case3)
print(f"Test case {test_case3}: {output3}")  # Expected: False

test_case4 = []  # Example of an empty list
output4 = method(test_case4)
print(f"Test case {test_case4}: {output4}")  # Expected: True